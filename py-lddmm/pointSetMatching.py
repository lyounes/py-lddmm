import os
import loggingUtils
import numpy as np
import scipy.linalg as la
import logging
import conjugateGradient as cg
import pointSets
import kernelFunctions as kfun
import pointEvolution as evol
from affineBasis import AffineBasis, getExponential, gradExponential
import matplotlib
matplotlib.use("TKAgg")
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as stats
import matplotlib.pyplot as plt
import pygraph
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv
from mnist import MNIST

class TestErrors:
    def __init__(self):
        self.knn = 1.
        self.linSVM = 1.
        self.SVM = 1.
        self.RF = 1.
        self.logistic = 1
    def __repr__(self):
        rates = 'logistic: {0:f}, linSVM: {1:f}, SVM: {2:f}, RF: {3:f}, kNN: {4:f}\n'.format(self.logistic,
                                        self.linSVM, self.SVM, self.RF, self.knn)
        return rates


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class PointSetMatchingParam:
    def __init__(self, timeStep = .1, KparDiff = None, sigmaKernel = 6.5,
                 sigmaError = 1.0, errorType = 'L2'):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.sigmaError = sigmaError
        self.errorType = errorType
        if errorType == 'L2':
            self.fun_obj0 = pointSets.L2Norm0
            self.fun_obj = pointSets.L2NormDef
            self.fun_objGrad = pointSets.L2NormGradient
        elif errorType == 'classification':
            self.fun_obj0 = None
            self.fun_obj = None
            self.fun_objGrad = None            
        else:
            logging.error('Unknown error Type: ' + self.errorType)
        if KparDiff == None:
            self.KparDiff = kfun.Kernel(name = 'gauss', sigma = self.sigmaKernel)
        else:
            self.KparDiff = KparDiff

class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []


## Main class for surface matching
#        Template: surface class (from surface.py); if not specified, opens fileTemp
#        Target: surface class (from surface.py); if not specified, opens fileTarg
#        param: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        affineWeight: multiplicative constant on affine regularization
#        rotWeight: multiplicative constant on affine regularization (supercedes affineWeight)
#        scaleWeight: multiplicative constant on scale regularization (supercedes affineWeight)
#        transWeight: multiplicative constant on translation regularization (supercedes affineWeight)
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'similitude', 'euclidean', 'translation' or 'none'
#        maxIter: max iterations in conjugate gradient
class PointSetMatching(object):

    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, verb=True, testSet = None, addDim = 0, intercept=True,
                 u0 = None, normalizeInput = True, l1Cost = 0, relearnRate = 1,
                 rotWeight = None, scaleWeight = None, transWeight = None,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):
        if param is None:
            self.param = PointSetMatchingParam()
        else:
            self.param = param

        if Template is None:
            if fileTempl is None:
                logging.error('Please provide a template surface')
                return
            else:
                self.fv0 = pointSets.loadlmk(fileTempl)[0]
        else:
            self.fv0 = np.copy(Template)
            
        if Target is None:
            if fileTarg is None:
                logging.error('Please provide a target surface')
                return
            else:
                if self.param.errorType == 'classification':
                    self.fv1 = pointSets.loadlmk(fileTempl)[1]
                else:
                    self.fv1 = pointSets.loadlmk(fileTarg)[0]
        else:
            self.fv1 = np.copy(Target)

            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()

        if self.param.errorType == 'classification':
            if normalizeInput:
                s = 1e-5 + np.std(self.fv0, axis=0)
                self.fv0 /= s

            if addDim > 0:
                self.fv0 = np.concatenate((self.fv0,
                                           0.01*np.random.normal(size=(self.fv0.shape[0],addDim))), axis=1)
        self.saveRate = 10
        self.relearnRate = relearnRate
        self.iter = 0
        self.setOutputDir(outputDir)
        self.dim = self.fv0.shape[1]
        self.maxIter = maxIter
        self.verb = verb
        self.saveTrajectories = saveTrajectories
        self.testGradient = testGradient
        self.regweight = regWeight
        self.affine = affine
        self.affB = AffineBasis(self.dim, affine)
        self.affineDim = self.affB.affineDim
        self.affineBasis = self.affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(self.affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[self.affB.rotComp] = rotWeight
        if (len(self.affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[self.affB.simComp] = scaleWeight
        if (len(self.affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[self.affB.transComp] = transWeight



        self.x0 = np.copy(self.fv0)
        self.x0try = np.copy(self.fv0)
        self.fvDef = np.copy(self.fv0)
        self.npt = self.fv0.shape[0]
        # self.u = np.zeros((self.dim, 1))
        # self.u[0:2] = 1/np.sqrt(2.)

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.random.normal(0, 0.01, [self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.x0.shape[0]
        self.saveFile = saveFile
        pointSets.savelmk(self.fv0, self.outputDir+'/Template.vtk')
        if not(self.param.errorType == 'classification'):
            pointSets.savelmk(self.fv1, self.outputDir+'/Target.vtk')
        self.coeffAff1 = 1.
        self.coeffAff2 = 100.
        self.coeffAff = self.coeffAff1
        self.coeffInitx = .1
        self.affBurnIn = 25
        self.pplot = pplot
        self.testSet = testSet
        self.l1Cost = l1Cost

        if self.param.errorType == 'classification':
            self.intercept = intercept
            self.testError = TestErrors()
            self.nclasses = self.fv1.max()+1
            nTr = np.zeros(self.nclasses)
            for k in range (self.nclasses):
                nTr[k] = (self.fv1==k).sum()
            #self.wtr = np.zeros(self.fv1.shape)
            self.wTr = float(self.fv1.size)/(nTr[self.fv1[:,0]]*self.nclasses)[:, np.newaxis]
            self.swTr = self.wTr.sum()
            #self.wTr *= self.swTr

            if u0 is None:
                self.u = self.learnLogistic()
            else:
                self.u = u0
            print "Non-negative coefficients", (np.fabs(self.u).sum(axis=1) > 1e-10).sum()
            if self.intercept:
                xDef1 = np.concatenate((np.ones((self.fvDef.shape[0], 1)), self.fvDef), axis=1)
            else:
                xDef1 = self.fvDef
            gu = np.argmax(np.dot(xDef1, self.u), axis=1)[:,np.newaxis]
            err = np.sum(np.not_equal(gu, self.fv1)*self.wTr)/self.swTr
            logging.info('Training Error {0:0f}'.format(err))
            if self.testSet is not None:
                if normalizeInput:
                    self.testSet = (self.testSet[0]/s, self.testSet[1])
                if addDim > 0:
                    self.testSet= (np.concatenate((self.testSet[0], np.zeros((self.testSet[0].shape[0], addDim))),
                                                  axis=1),
                                   self.testSet[1])
                nTe = np.zeros(self.nclasses)
                for k in range(self.nclasses):
                    nTe[k] = (self.testSet[1] == k).sum()
                self.wTe = float(self.testSet[1].size)/(nTe[self.testSet[1][:, 0]]*self.nclasses)[:, np.newaxis]
                self.swTe = self.wTe.sum()
                #self.wTe *= self.swTe
                testRes = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff,
                                                            withPointSet=self.testSet[0])
                if self.intercept:
                    xDef1 = np.concatenate((np.ones((self.testSet[0].shape[0], 1)), testRes[1][-1,...]), axis=1)
                else:
                    xDef1 = testRes[1][-1, ...]
                gu = np.argmax(np.dot(xDef1, self.u), axis=1)[:,np.newaxis]
                test_err = np.sum(np.not_equal(gu, self.testSet[1])*self.wTe)/self.swTe
                logging.info('Testing Error {0:0f}'.format(test_err))
            if self.pplot:
                self.colors = ['red', 'blue', 'coral', 'green', 'gold', 'maroon','magenta','olive','lime','purple']
                fig = plt.figure(2)
                fig.clf()
                for k in range(self.npt):
                    plt.plot([self.fvDef[k, 0]], [self.fvDef[k, 1]], marker='o',
                             color=self.colors[self.fv1[k,0]])
                if self.testSet is not None:
                    for k in range(self.testSet[0].shape[0]):
                        plt.plot([self.testSet[0][k, 0]], [self.testSet[0][k, 1]], marker='*',
                                 color=self.colors[self.testSet[1][k,0]])
                plt.pause(0.1)

    #     if self.pplot:
    #         fig=plt.figure(3)
    #         #fig.clf()
    #         ax = Axes3D(fig)
    #         lim0 = self.addSurfaceToPlot(self.fv1, ax, ec = 'k', fc = 'b')
    #         lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
    # #            ax.plot_trisurf(self.fv1.vertices[self.fv1.faces[:,0],:], self.fv1.vertices[self.fv1.faces[:,1],:],
    # #                           self.fv1.vertices[self.fv1.faces[:,2],:])# color=[0,0,1])
    #         #plt.axis('equal')
    #         ax.set_xlim(min(lim0[0][0],lim1[0][0]), max(lim0[0][1],lim1[0][1]))
    #         ax.set_ylim(min(lim0[1][0],lim1[1][0]), max(lim0[1][1],lim1[1][1]))
    #         ax.set_zlim(min(lim0[2][0],lim1[2][0]), max(lim0[2][1],lim1[2][1]))
    #         plt.pause(0.1)


    # def addSurfaceToPlot(self, fv1, ax, ec = [0,0,1], fc = [1,0,0], al=.5, lw=1):
    #     x = fv1.vertices[fv1.faces[:,0],:]
    #     y = fv1.vertices[fv1.faces[:,1],:]
    #     z = fv1.vertices[fv1.faces[:,2],:]
    #     a = np.concatenate([x,y,z], axis=1)
    #     poly = [ [a[i,j*3:j*3+3] for j in range(3)] for i in range(a.shape[0])]
    #     tri = Poly3DCollection(poly, alpha=al, linewidths=lw)
    #     tri.set_edgecolor(ec)
    #     tri.set_facecolor(fc)
    #     ax.add_collection3d(tri)
    #     xlim = [fv1.vertices[:,0].min(),fv1.vertices[:,0].max()]
    #     ylim = [fv1.vertices[:,1].min(),fv1.vertices[:,1].max()]
    #     zlim = [fv1.vertices[:,2].min(),fv1.vertices[:,2].max()]
    #     return [xlim, ylim, zlim]

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef, _fvInit = None):
        if self.param.errorType == 'classification':
            obj = pointSets.LogisticScore(_fvDef, self.fv1, self.u, w=self.wTr, intercept=self.intercept, l1Cost=self.l1Cost) \
                  / (self.param.sigmaError**2)
            #obj = pointSets.LogisticScore(_fvDef, self.fv1, self.u) / (self.param.sigmaError**2)
        else:
            obj = self.param.fun_obj(_fvDef, self.fv1) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, at, Afft, kernel = None, withTrajectory = False, withJacobian=False, x0 = None, regWeight = None):
        if x0 is None:
            x0 = self.x0
        if kernel is None:
            kernel = self.param.KparDiff
        #print 'x0 fun def', x0.sum()
        if regWeight is None:
            regWeight = self.regweight
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        foo = np.copy(self.fv0)
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        if withJacobian:
            (xt,Jt)  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A, withJacobian=True)
        else:
            xt  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A)

        obj=0
        obj1 = 0 
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :, :])
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            ra = kernel.applyK(z, a)
            if hasattr(self, 'v'):  
                self.v[t, :] = ra
            obj = obj + regWeight*timeStep*np.multiply(a, (ra)).sum()

            if self.affineDim > 0:
                obj1 +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        #print obj, obj+obj1
        obj += obj1
        if withJacobian:
            return obj, xt, Jt
        elif withTrajectory:
            return obj, xt
        else:
            return obj


    def objectiveFun(self):
        if self.obj == None:
            if self.param.errorType == 'classification':
                self.obj0 = 0
            else:
                self.obj0 = self.param.fun_obj0(self.fv1) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            #foo = surfaces.Surface(surf=self.fvDef)
            self.fvDef = np.copy(np.squeeze(self.xt[-1, :, :]))
            #foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft, self.x0]

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft
        x0Try = self.x0

        foo = self.objectiveFunDef(atTry, AfftTry, x0 = x0Try, withTrajectory=True)
        objTry += foo[0]
        ff = np.copy(np.squeeze(foo[1][-1, :, :]))
        objTry += self.dataTerm(ff)

        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry
            self.x0Try = x0Try
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry


    def testEndpointGradient(self):
        c0 = self.dataTerm(self.fvDef)
        ff = np.copy(self.fvDef)
        dff = np.random.normal(size=ff.shape)
        eps = 1e-6
        ff += eps*dff
        c1 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps, (grd*dff).sum()) )

    def endPointGradient(self):
        if self.param.errorType == 'classification':
            px = pointSets.LogisticScoreGradient(self.fvDef, self.fv1, self.u, w=self.wTr, intercept=self.intercept)
            #px = pointSets.LogisticScoreGradient(self.fvDef, self.fv1, self.u)
        else:
            px = self.param.fun_objGrad(self.fvDef, self.fv1)
        return px / self.param.sigmaError**2

    
    def hamiltonianCovector(self, x0, at, px1, KparDiff, regWeight, affine = None):
        N = x0.shape[0]
        dim = x0.shape[1]
        M = at.shape[0]
        timeStep = 1.0/M
        xt = evol.landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine)
        if not(affine is None):
            A0 = affine[0]
            A = np.zeros([M,dim,dim])
            for k in range(A0.shape[0]):
                A[k,...] = getExponential(timeStep*A0[k]) 
        else:
            A = np.zeros([M,dim,dim])
            for k in range(M):
                A[k,...] = np.eye(dim) 
                
        pxt = np.zeros([M+1, N, dim])
        pxt[M, :, :] = px1
        foo = np.copy(self.fv0)
        for t in range(M):
            px = np.squeeze(pxt[M-t, :, :])
            z = np.squeeze(xt[M-t-1, :, :])
            a = np.squeeze(at[M-t-1, :, :])
            foo = np.copy(z)
            v = KparDiff.applyK(z,a)
            a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*regWeight*a[np.newaxis,...]))
            a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
            zpx = KparDiff.applyDiffKT(z, a1, a2)
                
            if not (affine is None):
                pxt[M-t-1, :, :] = np.dot(px, A[M-t-1]) + timeStep * zpx
            else:
                pxt[M-t-1, :, :] = px + timeStep * zpx
        return pxt, xt
    
    def hamiltonianGradient(self, px1, kernel = None, affine=None, regWeight=None, x0=None, at=None):
        if regWeight is None:
            regWeight = self.regweight
        if x0 is None:
            x0 = self.x0
        if at is None:
            at = self.at
        if kernel is None:
            kernel  = self.param.KparDiff
        return evol.landmarkHamiltonianGradient(x0, at, px1, kernel, regWeight, affine=affine,
                                                getCovector=True)
                                                    

    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        foo = self.hamiltonianGradient(px1, affine=A)
        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            grd.aff = 2*self.affineWeight.reshape([1, self.affineDim])*self.Afft
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
               grd.aff[t] -=  dAff.reshape(grd.aff[t].shape)
            grd.aff /= (self.coeffAff*coeff*self.Tsize)
            #            dAfft[:,0:self.dim**2]/=100
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.aff = dir1.aff + beta * dir2.aff
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)
        dir.aff = np.copy(dir0.aff)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[t, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            u = self.param.KparDiff.applyK(z, gg)
            #uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            uu = g1.aff[t]
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    #print np.multiply(np.multiply(g1[1][t], gr[1][t]), self.affineWeight).shape
                    #res[ll] += np.multiply(uu, gr.aff[t]).sum() * self.coeffAff
                    res[ll] += np.multiply(uu, gr.aff[t]).sum() * self.coeffAff
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1


        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        self.x0 = np.copy(self.x0Try)
        #print self.at

    def endOfIteration(self, endP=False):
        self.iter += 1
        if self.testGradient:
            self.testEndpointGradient()

        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0 or endP) :
            logging.info('Saving surfaces...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            # if self.affineDim <=0:
            #     xtEPDiff, atEPdiff = evol.landmarkEPDiff(self.at.shape[0], self.x0,
            #                                              np.squeeze(self.at[0, :, :]), self.param.KparDiff)
            #     self.fvDef = np.copy(np.squeeze(xtEPDiff[-1, :, :]))
            #     pointSets.savelmk(self.fvDef, self.outputDir +'/'+ self.saveFile+'EPDiff.vtk')
            #     logging.info('EPDiff difference %f' %(np.fabs(self.xt[-1,:,:] - xtEPDiff[-1,:,:]).sum()) )

            self.fvDef = np.copy(np.squeeze(self.xt[-1, :, :]))
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            (xt, Jt)  = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                              withJacobian=True)
            if self.affine=='euclidean' or self.affine=='translation':
                f = np.copy(self.fv0)
                X = self.affB.integrateFlow(self.Afft)
                displ = np.zeros(self.x0.shape[0])
                dt = 1.0 /self.Tsize
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yyt = np.dot(self.xt[t,...] - X[1][t, ...], U.T)
                    f = np.copy(yyt)
                    # vf = surfaces.vtkFields() ;
                    # vf.scalars.append('Jacobian') ;
                    # vf.scalars.append(np.exp(Jt[t, :]))
                    # vf.scalars.append('displacement')
                    # vf.scalars.append(displ)
                    # vf.vectors.append('velocity') ;
                    # vf.vectors.append(vt)
                    # nu = self.fv0ori*f.computeVertexNormals()
                    pointSets.savelmk(f, self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.lmk')
                f = np.copy(self.fv1)
                yyt = np.dot(f - X[1][-1, ...], U.T)
                f = np.copy(yyt)
                pointSets.savelmk(f, self.outputDir +'/TargetCorrected.lmk')
            for kk in range(self.Tsize+1):
                fvDef = np.copy(np.squeeze(xt[kk, :, :]))
                pointSets.savelmk(fvDef, self.outputDir +'/'+ self.saveFile+str(kk)+'.lmk')
            if self.param.errorType == 'classification':
                # J1 = np.nonzero(self.fv1>0)[0]
                # J2 = np.nonzero(self.fv1<0)[0]
                # self.u = np.mean(self.fvDef[J1, :], axis=0) - np.mean(self.fvDef[J2, :], axis=0)
                # self.u = (self.u/np.sqrt((self.u**2).sum()))[:, np.newaxis]
                if self.intercept:
                    xDef1 = np.concatenate((np.ones((self.fvDef.shape[0], 1)), self.fvDef), axis=1)
                else:
                    xDef1 = self.fvDef
                gu = np.argmax(np.dot(xDef1, self.u), axis=1)[:,np.newaxis]
                err = np.sum(np.not_equal(gu, self.fv1) * self.wTr)/self.swTr
                logging.info('Training Error {0:0f}'.format(err))
                if self.nclasses < 3:
                    pcau = PCA(n_components=self.nclasses)
                else:
                    pcau = PCA(n_components=3)

                if self.intercept:
                    ofs = 1
                else:
                    ofs = 0

                U,S,V = la.svd(self.u[ofs:self.dim+ofs,:])
                xRes = np.dot(self.fvDef, U)
                if self.nclasses < 4:
                    xTr3 = xRes[:, 0:self.nclasses-1]
                    xRes = xRes[:,self.nclasses:self.dim]
                        #np.dot(np.concatenate((xTr3,np.zeros((xTr3.shape[0], self.dim-self.nclasses+1))),axis=1), U)
                    if xRes.shape[1] > 4-self.nclasses:
                        pca = PCA(4-self.nclasses)
                        xRes = pca.fit_transform(xRes)
                    xTr3 = np.concatenate((xTr3, xRes), axis=1)
                else:
                    xTr3 = xRes[:, 0:3]
                if self.testSet is not None:
                    testRes = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, withPointSet=self.testSet[0])
                    x0Tr = self.fvDef
                    x0Te = testRes[1][testRes[1].shape[0]-1,...]
                    x1Tr = self.fv1
                    x1Te = self.testSet[1]
                    self.testDef = x0Te
                    if self.intercept:
                        xDef1 = np.concatenate((np.ones((self.testSet[0].shape[0], 1)), x0Te), axis=1)
                    else:
                        xDef1 = x0Te
                    gu = np.argmax(np.dot(xDef1, self.u), axis=1)[:,np.newaxis]
                    test_err = np.sum(np.not_equal(gu, self.testSet[1])*self.wTe)/self.swTe
                    self.testError.logistic = test_err
                    logging.info('Testing Error {0:0f}'.format(test_err))
                    xRes = np.dot(x0Te, U)
                    if self.nclasses < 3:
                        xTe3 = xRes[:, 0:self.nclasses - 1]
                        if xRes.shape[1] > 4:
                            xRes = pca.transform(xRes[:, self.nclasses:xRes.shape[1]])
                        else:
                            xRes = xRes[:, self.nclasses:xRes.shape[1]]
                        xTe3 = np.concatenate((xTe3, xRes), axis=1)
                    else:
                        xTe3 = xRes[:, 0:3]
                    clf = svm.SVC(class_weight='balanced')
                    clf.fit(x0Tr, np.ravel(x1Tr))
                    yTr = clf.predict(x0Tr)
                    yTe = clf.predict(x0Te)
                    self.testError.SVM = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe
                    logging.info('SVM prediction: {0:f} {1:f}'.format(np.sum(np.not_equal(yTr, np.ravel(x1Tr)) * np.ravel(self.wTr)) / self.swTr, \
                        np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe))
                    clf = svm.LinearSVC(class_weight='balanced')
                    clf.fit(x0Tr, np.ravel(x1Tr))
                    yTe = clf.predict(x0Te)
                    self.testError.linSVM = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe
                    logging.info('Linear SVM prediction: {0:f}'.format(np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe))

                    clf = RandomForestClassifier(n_estimators=20, class_weight='balanced')
                    clf.fit(x0Tr, np.ravel(x1Tr))
                    yTr = clf.predict(x0Tr)
                    yTe = clf.predict(x0Te)
                    self.testError.RF = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe
                    logging.info('RF prediction: {0:f} {1:f}'.format(np.sum(np.not_equal(yTr, np.ravel(x1Tr)) * np.ravel(self.wTr)) / self.swTr, \
                        np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe))

                    clf = KNeighborsClassifier()
                    clf.fit(x0Tr, np.ravel(x1Tr))
                    yTr = clf.predict(x0Tr)
                    yTe = clf.predict(x0Te)
                    self.testError.knn = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe
                    logging.info('kNN prediction: {0:f} {1:f}'.format(np.sum(np.not_equal(yTr, np.ravel(x1Tr)) * np.ravel(self.wTr)) / self.swTr, \
                        np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(self.wTe)) / self.swTe))

                if self.pplot:
                    #JJ = np.argpartition(np.ravel((self.u[ofs:self.dim+ofs,:]**2).sum(axis=1)), self.dim-2)
                    fig = plt.figure(4)
                    fig.clf()
                    #i1 = JJ[self.dim-2]
                    #i2 = JJ[self.dim-1]
                    for k in range(self.npt):
                        plt.plot([xTr3[k, 0]], [xTr3[k, 1]], marker='o',
                                 color=self.colors[self.fv1[k,0]])
                    # plt.plot([self.fvDef[k, i1]], [self.fvDef[k, i2]], marker='o',
                    #          color=self.colors[self.fv1[k, 0]])
                    if self.testSet is not None:
                        for k in range(self.testSet[0].shape[0]):
                            plt.plot([xTe3[k,0]], [xTe3[k,1]], marker='*',
                                     color=self.colors[self.testSet[1][k,0]])
                    # plt.plot([testRes[1][-1, k, i1]], [testRes[1][-1, k, i2]], marker='*',
                    #          color=self.colors[self.testSet[1][k, 0]])
                    plt.pause(0.1)
                if self.intercept:
                    ofs = 1
                else:
                    ofs = 0
                # JJ = np.argpartition(np.ravel((self.u[ofs:self.dim+ofs,:]**2).sum(axis=1)), self.dim - 3)
                # I = JJ[self.dim-3:self.dim]
                for kk in range(self.Tsize + 1):
                    fvDef = np.copy(np.squeeze(xt[kk, :, :]))
                    xRes = np.dot(fvDef, U)
                    if self.nclasses < 4:
                        x3 = xRes[:, 0:self.nclasses - 1]
                        if xRes.shape[1] > 4:
                            xRes = pca.transform(xRes[:, self.nclasses:xRes.shape[1]])
                        else:
                            xRes = xRes[:, self.nclasses:xRes.shape[1]]
                        x3 = np.concatenate((x3, xRes), axis=1)
                    else:
                        x3 = xRes[:, 0:3]
                    pointSets.savePoints(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', x3,
                                         scalars=np.ravel(self.fv1))
                if self.testSet is not None:
                    for kk in range(self.Tsize + 1):
                        fvDef = np.copy(np.squeeze(testRes[1][kk, :, :]))
                        xRes = np.dot(fvDef, U)
                        if self.nclasses < 4:
                            x3 = xRes[:, 0:self.nclasses - 1]
                            if xRes.shape[1] > 4:
                                xRes = pca.transform(xRes[:, self.nclasses:xRes.shape[1]])
                            else:
                                xRes = xRes[:, self.nclasses:xRes.shape[1]]
                            x3 = np.concatenate((x3, xRes), axis=1)
                        else:
                            x3 = xRes[:, 0:3]
                        pointSets.savePoints(self.outputDir +'/'+ self.saveFile+'Test'+str(kk)+'.vtk', x3,
                                             scalars=np.ravel(self.testSet[1]))
        else:
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef = np.copy(np.squeeze(self.xt[-1, :, :]))
        if self.relearnRate > 0 and (self.iter % self.relearnRate == 0):
            u0 = self.u
            self.u = self.learnLogistic(u0=self.u)
            logging.info('Resetting weights: delta u = {0:f}, norm u = {1:f} '.format(np.sqrt(((self.u - u0) ** 2).sum()),
                                                                                      np.sqrt(((self.u) ** 2).sum())))
            self.reset = True
            # print self.obj
            # self.obj = None
            # self.obj = self.objectiveFun()
            # print self.obj

    def endOfProcedure(self):
        self.endOfIteration(endP=True)

    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        logging.info('Gradient lower bound: %f' %(self.gradEps))
        self.coeffAff = self.coeffAff1
        cg.cg(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
        #return self.at, self.xt

    def learnLogistic(self, u0=None, random = 1.0):
        return pointSets.learnLogistic(self.fvDef, self.fv1, w= self.wTr, u0=u0, l1Cost=self.l1Cost,
                                       intercept=self.intercept, random=random)

    def localMaps1D(self, d):
        KL1 = 3*np.ones(d, dtype=int)
        KL1[0] = 2
        KL1[-1] = 2
        KL0 = np.zeros(KL1.sum(), dtype=int)
        ii = 0
        for i in range(d):
            if i>0:
                KL0[ii] = i-1
                ii += 1
            KL0[ii] = i
            ii += 1
            if i<d-1:
                KL0[ii] = i+1
                ii += 1
        return (KL0, KL1)

    def localMapsNaive(self, d):
        KL1 = np.ones(d, dtype=int)
        KL0 = np.arange(d, dtype=int)
        return (KL0, KL1)

if __name__=="__main__":

    outputDir = '/Users/younes/Development/Results/Classif'
    loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput=True)

    N = 100
    d = 10
    l1Cost = 1.
    sigma = 1.
    sigError = 0.01
    addDim = 0
    #typeData = 'Dolls'
    typeData = 'helixes'
    localMaps = None
    removeNullDirs = False
    relearnRate = 1
    u0 = None

    if typeData == 'helixes':
        NTr = 1000
        NTe = 2000
        d = 20
        h = 0.25
        x0Tr = 0.05*np.random.randn(2*NTr,d)
        x0Te = 0.05*np.random.randn(2*NTe,d)
        #x1 = np.random.randn(100,2)
        x1Tr = np.ones((2*NTr,1), dtype = int)
        x1Te = np.ones((2*NTe,1), dtype=int)
        x1Tr[NTr:2*NTr] = 0
        x1Te[NTe:2*NTe] = 0
        t = 2*np.pi*np.random.rand(NTr)
        x0Tr[0:NTr,0] += np.cos(t)
        x0Tr[0:NTr,1] += np.sin(t)
        x0Tr[0:NTr,2] += h*t

        t = 2*np.pi*np.random.rand(NTe)
        x0Te[0:NTe,0] += np.cos(t)
        x0Te[0:NTe,1] += np.sin(t)
        x0Te[0:NTe,2] += h*t

        t = 2*np.pi*np.random.rand(NTr)
        x0Tr[NTr:2*NTr,0] += h*t
        x0Tr[NTr:2*NTr,1] += 1 + np.cos(t)
        x0Tr[NTr:2*NTr,2] += np.sin(t)

        t = 2*np.pi*np.random.rand(NTe)
        x0Te[NTe:2*NTe,0] += h*t
        x0Te[NTe:2*NTe,1] += 1 + np.cos(t)
        x0Te[NTe:2*NTe,2] += np.sin(t)

        x0Tr[:, 3:d] += 1.*np.random.randn(2*NTr,d-3)
        x0Te[:, 3:d] += 1.*np.random.randn(2*NTe,d-3)
        A = np.random.randn(d,d)
        R = la.expm((A-A.T)/2)
        #x0Tr = np.dot(x0Tr, R)
        #x0Te = np.dot(x0Te, R)
        sigma = 2.5
        addDim = 0
        l1Cost = 2.
        #localMaps = PointSetMatching().localMaps1D(d)
        #u0 = np.random.randn(d+1, 2)
        #removeNullDirs = False
    elif typeData == 'csv1':
        nv = -1
        X = np.genfromtxt('/Users/younes/Development/Data/Classification/BRCA1_q2_HW2.csv', delimiter=',')
        x1 = np.array(X[0,:].T, dtype=int)
        x0 = X[1:nv,:].T
        for k in range(x0.shape[0]):
            x0[k,:] = stats.rankdata(np.ravel(x0[k,:]), method='average')/float(x0.shape[1])
        I1 = np.nonzero(x1==1)[0]
        I2 = np.nonzero(x1==0)[0]
        J1 = np.random.random(I1.size)
        J2 = np.random.random(I2.size)
        x0Tr = np.concatenate((x0[I1[J1>0.33],:], x0[I2[J2>0.33],:]))
        x0Te = np.concatenate((x0[I1[J1<0.33],:], x0[I2[J2<0.33],:]))
        x1Tr = np.concatenate((x1[I1[J1>0.33]], x1[I2[J2>0.33]]))[:, np.newaxis]
        x1Te = np.concatenate((x1[I1[J1<0.33]], x1[I2[J2<0.33]]))[:, np.newaxis]
        d = x0Tr.shape[1]
        sigma = 2.5
        addDim = 0
        l1Cost = 1.
        #localMaps = PointSetMatching().localMapsNaive(d)
        relearnRate = 1
        removeNullDirs = False
        # xTmp = np.zeros((x0Tr.shape[0],2*d))
        # xTmp[:,range(0,2*d,2)] = x0Tr
        # xTmp[:,range(1,2*d,2)] = 0.1 * np.random.randn(x0Tr.shape[0],d)
        # x0Tr = xTmp
        # xTmp = np.zeros((x0Te.shape[0],2*d))
        # xTmp[:,range(0,2*d,2)] = x0Te
        # x0Te = xTmp
        # d *= 2
    elif typeData == 'RBF':
        d = 10
        d1 = 10
        nc = 20
        NTr = 1000
        NTe = 2000
        #centers = np.random.normal(0, 1, (nc, d))
        centers = np.zeros((nc, d))
        c = np.zeros(nc)
        for k in range(nc):
            centers[k,k%d] = k / float(nc)
            c[k] = (2*(k%1.5) - 0.75)
        #c = (2*np.random.random(nc) -1)
        x0Tr = np.random.normal(0,1, (NTr,d))
        K = np.exp(- ((x0Tr[:,np.newaxis,:] - centers[np.newaxis,:,:])**2).sum(axis=2)/np.sqrt(d))
        m = np.median(np.sin(np.dot(K,c)))
        x0Tr = np.random.normal(0,1, (NTr,d))
        K = np.exp(- ((x0Tr[:,np.newaxis,:] - centers[np.newaxis,:,:])**2).sum(axis=2)/np.sqrt(d))
        x1Tr = np.array((np.sin(np.dot(K,c))-m)[:, np.newaxis] > 0, dtype=int)
        x0Te = np.random.normal(0,1, (NTe,d))
        K = np.exp(- ((x0Te[:,np.newaxis,:] - centers[np.newaxis,:,:])**2).sum(axis=2)/np.sqrt(d))
        x1Te = np.array((np.sin(np.dot(K,c))-m)[:, np.newaxis] > 0, dtype=int)
        #x0Tr = np.concatenate((x0Tr,0.5*np.random.normal(0,1,(NTr,d1))), axis=1)
        #x0Te = np.concatenate((x0Te,0.5*np.random.normal(0,1,(NTe,d1))), axis=1)
        #d += d1
        l1Cost = 2.
        sigma = 2.5
        sigError = 0.01
        addDim = 1
        #localMaps = PointSetMatching().localMaps1D(d)

    elif typeData == 'MoG':
        NTr = 100
        NTe = 2000
        N = NTr + NTe
        d = 10
        Cov0 = np.eye(d)/2
        m0 = np.concatenate((np.ones(3), np.zeros(d - 3)))
        q = np.arange(0, 1, 1.0 / d)
        Cov1 = 2*np.exp(-np.abs(q[:,np.newaxis]-q[np.newaxis,:]))
        #Cov1 = np.eye(d)
        Cov2 = np.exp(-np.abs(q[:,np.newaxis]-q[np.newaxis,:])/3.)
        #Cov2 = np.eye(d)
        m1 = np.concatenate((-np.ones(3), np.zeros(d - 3)))
        m2 = np.concatenate((-np.array([1,-1,1]), np.zeros(d - 3)))
        x0Tr = np.zeros((3 * NTr, d))
        x0Te = np.zeros((3 * NTe, d))
        x0Tr[0:NTr, :] = np.random.multivariate_normal(m0, Cov0, size=NTr)
        x0Te[0:NTe, :] = np.random.multivariate_normal(m0, Cov0, size=NTe)
        x0Tr[NTr:2 * NTr, :] = np.random.multivariate_normal(m1, Cov1, size=NTr)
        x0Te[NTe:2 * NTe, :] = np.random.multivariate_normal(m1, Cov1, size=NTe)
        x0Tr[2*NTr:3 * NTr, :] = np.random.multivariate_normal(m2, Cov2, size=NTr)
        x0Te[2*NTe:3 * NTe, :] = np.random.multivariate_normal(m2, Cov2, size=NTe)
        x1Tr = np.zeros((3 * NTr, 1), dtype=int)
        x1Te = np.zeros((3 * NTe, 1), dtype=int)
        x1Tr[NTr:2 * NTr] = 1
        x1Te[NTe:2 * NTe] = 1
        x1Tr[2*NTr:3 * NTr] = 2
        x1Te[2*NTe:3 * NTe] = 2
        A = np.random.randn(d,d)
        R = la.expm((A-A.T)/2)
        x0Tr = np.dot(x0Tr, R)
        x0Te = np.dot(x0Te, R)
        l1Cost = 2.0
        sigma = 2.5
        addDim = 0
        sigError = .01
    elif typeData=='MNIST':
        mndata = MNIST('/cis/home/younes/MNIST')
        images, labels = mndata.load_training()
        imTest, labTest = mndata.load_testing()
        N0 = len(images)
        d = len(images[0])
        NTr = 1500
        NTe = 2000 #len(imTest)
        cls = [3,5,8]
        x0Tr = np.zeros((NTr,d))
        x1Tr = np.zeros((NTr,1), dtype=int)
        kk = 0
        for k in range(len(images)):
            if labels[k] in cls:
                x0Tr[kk,:] = np.array(images[k])/255.
                x1Tr[kk] = cls.index(labels[k])
                kk += 1
                if kk == NTr:
                    break
        if kk < NTr:
            NTr = kk
            x0Tr = x0Tr[0:NTr,:]
            x1Tr = x1Tr[0:NTr]
        #std = np.std(x0Tr, axis=0)
        #print sum(std > 0.05)
        #pca = PCA(n_components=0.90)
        #x0Tr = pca.fit_transform(x0Tr)
        #x0Tr = x0Tr / np.sqrt(pca.singular_values_)
        #x0Tr = x0Tr[:,std>0.05]

        x0Te = np.zeros((NTe, d))
        x1Te = np.zeros((NTe,1), dtype=int)
        kk = 0
        for k in range(len(imTest)):
            if labTest[k] in cls:
                x0Te[kk,:] = np.array(imTest[k])/255.
                x1Te[kk] = cls.index(labTest[k])
                kk += 1
                if kk == NTe:
                    break
        if kk < NTe:
            NTe = kk
            x0Te = x0Te[0:NTe,:]
            x1Te = x1Te[0:NTe]
        #x0Te = pca.transform(x0Te) #/np.sqrt(pca.singular_values_)
        #x0Te = x0Te[:,std>0.05]
        sigma = 2.5
        l1Cost = 0.5
        sigError = 0.005
        addDim = 0
        #pca.inverse_transform(x0Tr).tofile(outputDir + '/mnistOutTrain.txt')
        #pca.inverse_transform(x0Te).tofile(outputDir + '/mnistOutTest.txt')
    elif typeData == 'Dolls':
        d = 3
        NTr = 100
        NTe = 2000
        x0Tr = np.random.multivariate_normal(np.zeros(d), np.eye(d), NTr)
        x0Te = np.random.multivariate_normal(np.zeros(d), np.eye(d), NTe)
        nrm = np.sqrt((x0Tr**2).sum(axis=1))
        x1Tr = np.array(np.sign(np.cos(4*nrm))>0, dtype=int)[:, np.newaxis]
        nrm = np.sqrt((x0Te**2).sum(axis=1))
        x1Te = np.array(np.sign(np.cos(4*nrm))>0, dtype=int)[:, np.newaxis]
        sigma = 2.5
        sigError = 0.01
        l1Cost = 2.0
        addDim = 1
        removeNullDirs = False
        relearnRate = 1
    elif typeData == 'Segments':
        NTr = 500
        NTe = 2000
        d = 100
        l0 = 10
        l1 = 12
        x0Tr = np.zeros((2*NTr, d))
        x1Tr = np.zeros((2*NTr,1), dtype=int)
        x1Tr[NTr:2*NTr,0] = 1
        start = np.random.randint(0, d+1-l0, NTr)
        for k in range(NTr):
            x0Tr[k,start[k]:start[k]+l0] = 1
        start = np.random.randint(0, d+1-l1, NTr)
        for k in range(NTr):
            x0Tr[k+NTr,start[k]:start[k]+l1] = 1
        x0Te = np.zeros((2*NTe, d))
        x1Te = np.zeros((2*NTe,1), dtype=int)
        x1Te[NTe:2*NTe,0] = 1
        start = np.random.randint(0, d+1-l0, NTe)
        for k in range(NTe):
            x0Te[k,start[k]:start[k]+l0] = 1
        start = np.random.randint(0, d+1-l1, NTe)
        for k in range(NTe):
            x0Te[k+NTe,start[k]:start[k]+l1] = 1
        #x0Tr += 0.01 * np.random.randn(x0Tr.shape[0], x0Tr.shape[1])
        #x0Te += 0.01 * np.random.randn(x0Te.shape[0], x0Te.shape[1])
        x0Tr *= (1 + 0.25*np.random.randn(x0Tr.shape[0], 1))
        x0Te *= (1 + 0.25*np.random.randn(x0Te.shape[0], 1))
        addDim = 1
        sigma = 2.5
        l1Cost = 2.
        sigError = 0.01
        removeNullDirs = False

        #localMaps = PointSetMatching().localMaps1D(d)
    else:
        NTr = 100
        NTe = 1000
        N = NTr + NTe
        d = 10
        Cov0 = np.eye(d)
        m0 = np.concatenate((np.ones(3), np.zeros(d-3)))
        q = np.arange(0,1,1.0/d)
        Cov1 = 2*np.exp(-np.abs(q[:,np.newaxis]-q[np.newaxis,:]))
        #Cov1 = np.eye(d)
        m1 = np.concatenate((-np.ones(3), np.zeros(d-3)))
        x0Tr = np.zeros((2*NTr,d))
        x0Te = np.zeros((2*NTe,d))
        x0Tr[0:NTr, :] = np.random.multivariate_normal(m0, Cov0, size=NTr)
        x0Te[0:NTe, :] = np.random.multivariate_normal(m0, Cov0, size=NTe)
        x0Tr[NTr:2*NTr, :] = np.random.multivariate_normal(m1, Cov1, size=NTr)
        x0Te[NTe:2*NTe, :] = np.random.multivariate_normal(m1, Cov1, size=NTe)
        x1Tr = np.ones((2 * NTr,1), dtype=int)
        x1Te = np.ones((2 * NTe,1), dtype=int)
        x1Tr[NTr:2 * NTr] = 0
        x1Te[NTe:2 * NTe] = 0
        sigma = 2.5
        #err0 = (1 - (np.sign(x0Te*x1Te))).sum()/(4*N)

    #x0[:,2] = 0


    nclasses = x1Tr.max() + 1
    nTr = np.zeros(nclasses)
    for k in range(nclasses):
        nTr[k] = (x1Tr == k).sum()
    wTr = float(x1Tr.size) / (nTr[x1Tr[:, 0]] * nclasses)[:, np.newaxis]

    fu = pointSets.learnLogistic(x0Tr, x1Tr, w=wTr, l1Cost=l1Cost)
    xDef1 = np.concatenate((np.ones((x0Te.shape[0], 1)), x0Te), axis=1)
    gu = np.argmax(np.dot(xDef1, fu), axis=1)[:, np.newaxis]

    # for k in range(10):
    #     fu += pointSets.learnLogistic(x0Tr, x1Tr, w=wTr, l1Cost=l1Cost, random=0.75)
    # fu /= 11

    #x0Tr *= s
    #x0Te *=s

    if removeNullDirs:
        J = np.nonzero(np.fabs(fu[1:-1, :]).sum(axis=1) > 1e-8)[0]
        x0Tr0 = x0Tr[:,J]
        x0Te0 = x0Te[:,J]
    else:
        x0Tr0 = x0Tr
        x0Te0 = x0Te

    K1 = kfun.Kernel(name='laplacian', sigma=sigma, order=1)
    sm = PointSetMatchingParam(timeStep=0.1, KparDiff = K1, sigmaError=sigError*np.sqrt(NTr), errorType='classification')


    f = PointSetMatching(Template=x0Tr0, Target=x1Tr, outputDir=outputDir, param=sm, regWeight=1.,
                         saveTrajectories=True, pplot=True, testSet=(x0Te0, x1Te), addDim = addDim, u0=u0,
                         normalizeInput=False, l1Cost = l1Cost, relearnRate=relearnRate,
                         affine='none', testGradient=False, affineWeight=1e3,
                         maxIter=1000)

    testInit = TestErrors()
    testInit.logistic = np.sum(np.not_equal(gu, x1Te) * f.wTe) / f.swTe

    clf = svm.SVC(class_weight='balanced')
    clf.fit(x0Tr, np.ravel(x1Tr))
    yTr = clf.predict(x0Tr)
    yTe = clf.predict(x0Te)
    testInit.SVM = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(f.wTe)) / f.swTe
    print 'SVM prediction:', np.sum(np.not_equal(yTr, np.ravel(x1Tr))*np.ravel(f.wTr))/f.swTr, \
        np.sum(np.not_equal(yTe, np.ravel(x1Te))*np.ravel(f.wTe))/f.swTe

    clf = svm.LinearSVC(class_weight='balanced')
    clf.fit(x0Tr, np.ravel(x1Tr))
    yTe = clf.predict(x0Te)
    print 'Linear SVM prediction:', np.sum(np.not_equal(yTe, np.ravel(x1Te))*np.ravel(f.wTe))/f.swTe
    testInit.linSVM = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(f.wTe)) / f.swTe

    clf = RandomForestClassifier(n_estimators=20, class_weight='balanced')
    clf.fit(x0Tr, np.ravel(x1Tr))
    yTr = clf.predict(x0Tr)
    yTe = clf.predict(x0Te)
    testInit.RF = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(f.wTe)) / f.swTe
    print 'RF prediction:', np.sum(np.not_equal(yTr, np.ravel(x1Tr))*np.ravel(f.wTr))/f.swTr, \
        np.sum(np.not_equal(yTe, np.ravel(x1Te))*np.ravel(f.wTe))/f.swTe

    clf = KNeighborsClassifier()
    clf.fit(x0Tr, np.ravel(x1Tr))
    yTr = clf.predict(x0Tr)
    yTe = clf.predict(x0Te)
    testInit.knn = np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(f.wTe)) / f.swTe
    print 'kNN prediction:', np.sum(np.not_equal(yTr, np.ravel(x1Tr)) * np.ravel(f.wTr)) / f.swTr, \
        np.sum(np.not_equal(yTe, np.ravel(x1Te)) * np.ravel(f.wTe)) / f.swTe

    print testInit
    if localMaps is not None:
        K1.localMaps = localMaps



    #K1.localMaps = (np.zeros(3*d-2, dtype=int), 3*np.ones(d, dtype=int))
    # K1.localMaps[1][0] = 2
    # K1.localMaps[1][d-1] = 2
    # KJ = np.random.permutation(d)
    # jK=0
    # for k in range(d):
    #     if k>0:
    #         K1.localMaps[0][jK] = KJ[k-1]
    #         jK += 1
    #     K1.localMaps[0][jK] = KJ[k]
    #     jK += 1
    #     if k<d-1:
    #         K1.localMaps[0][jK] = KJ[k+1]
    #         jK += 1
        # K1.localMaps[0][4*k] = 2*k+1
        # K1.localMaps[0][4 * k+1] = 2 * k + 2
        # K1.localMaps[0][4 * k + 2] = 2 * k + 2
        # K1.localMaps[0][4 * k + 3] = 2 * k + 1

    for k in range(1):
        f.optimizeMatching()
        #sigma /= 2
        #K1 = kfun.Kernel(name='laplacian', sigma=sigma, order=1)
        #sm = PointSetMatchingParam(timeStep=0.1, KparDiff=K1, sigmaError=sigError, errorType='classification')

        # f = PointSetMatching(Template=f.fvDef, Target=x1Tr, outputDir=outputDir, param=sm, regWeight=1.,
        #                      saveTrajectories=True, pplot=True, testSet=(f.testDef, x1Te), addDim=0, u0=u0,
        #                      normalizeInput=False, l1Cost=l1Cost, relearnRate=relearnRate,
        #                      affine='none', testGradient=True, affineWeight=1e3,
        #                      maxIter=1000)
        #f.u = f.learnLogistic(u0=f.u) # + np.random.normal(0,10.0/(k+1), f.u.shape)
        # f.obj = None
        # f.iter = 0
        # f.objectiveFun()
        # f.param.KparDiff.sigma /= 2
        # #f.maxIter = 100
        #f.optimizeMatching()
    with open(outputDir+'/results.txt','a+') as fl:
        fl.write('\n'+typeData+' dim = {0:d} N = {1:d}\n'.format(d, NTr))

        fl.write('Initial: '+testInit.__repr__())
        fl.write('Final: ' +f.testError.__repr__())
    if typeData=='MNIST':
        pca.inverse_transform(f.fvDef[:,0:f.fvDef.shape[1]- addDim]).tofile(outputDir + '/mnistOutTrainDef.txt')
        pca.inverse_transform(f.testDef[:,0:f.fvDef.shape[1]- addDim]).tofile(outputDir + '/mnistOutTestDef.txt')



    plt.pause(1000)

