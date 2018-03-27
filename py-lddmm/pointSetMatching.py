import os
import loggingUtils
import numpy as np
import numpy.linalg as la
import logging
import conjugateGradient as cg
import pointSets
import kernelFunctions as kfun
import pointEvolution as evol
from affineBasis import AffineBasis, getExponential, gradExponential
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv


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
                 regWeight = 1.0, affineWeight = 1.0, verb=True, testSet = None,
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

        self.saveRate = 1
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
        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
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
        self.l1Cost = 100

        if self.param.errorType == 'classification':
            J1 = self.fv1 > 0
            J2 = self.fv1 < 0
            self.u = (np.mean(self.fv0[J1[:,0], :], axis=0) - np.mean(self.fv0[J2[:,0], :], axis=0))[:, np.newaxis]
            self.u /= np.sqrt((self.u ** 2).sum())
            for k in range(250):
                g = (self.fvDef *
                     (self.fv1 * np.exp(-(np.dot(self.fvDef, self.u) * self.fv1)))).sum(axis=0)[:, np.newaxis]
                #g -= (g * self.u).sum() * self.u
                ep = .1/self.fvDef.size
                fu = self.u + ep * g
                ll = ep*self.l1Cost
                fu = np.sign(fu) * np.maximum(np.fabs(fu) - ll, 0)
                # fu = (fu / np.sqrt((fu ** 2).sum()))
                # cl0 = pointSets.classScore(self.fvDef, self.fv1, self.u)
                # while ep > 1e-10 and pointSets.classScore(self.fvDef, self.fv1, fu) > cl0:
                #     ep *= 0.5
                #     fu = self.u + ep * g
                #     fu = fu / np.sqrt((fu ** 2).sum())
                self.u = fu
            if self.pplot:
                fig = plt.figure(2)
                fig.clf()
                for k in range(self.npt):
                    if self.fv1[k] > 0:
                        plt.plot([self.fvDef[k, 0]], [self.fvDef[k, 1]], 'ro')
                    else:
                        plt.plot([self.fvDef[k, 0]], [self.fvDef[k, 1]], 'bo')
                if self.testSet is not None:
                    for k in range(self.testSet[0].shape[0]):
                        if self.testSet[1][k] > 0:
                            plt.plot([self.testSet[0][k, 0]], [self.testSet[0][k, 1]], 'r*')
                        else:
                            plt.plot([self.testSet[0][k, 0]], [self.testSet[0][k, 1]], 'b*')
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
            obj = pointSets.classScore(_fvDef, self.fv1, u=self.u) / (self.param.sigmaError**2)
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
            px = pointSets.classScoreGradient(self.fvDef, self.fv1, u= self.u)
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

    def endOfIteration(self):
        self.iter += 1
        if self.testGradient:
            self.testEndpointGradient()

        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0) :
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
                    pointSets.savelmk(f, self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.vtk')
                f = np.copy(self.fv1)
                yyt = np.dot(f - X[1][-1, ...], U.T)
                f = np.copy(yyt)
                pointSets.savelmk(f, self.outputDir +'/TargetCorrected.vtk')
            for kk in range(self.Tsize+1):
                fvDef = np.copy(np.squeeze(xt[kk, :, :]))
                pointSets.savelmk(fvDef, self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
            if self.param.errorType == 'classification':
                # J1 = np.nonzero(self.fv1>0)[0]
                # J2 = np.nonzero(self.fv1<0)[0]
                # self.u = np.mean(self.fvDef[J1, :], axis=0) - np.mean(self.fvDef[J2, :], axis=0)
                # self.u = (self.u/np.sqrt((self.u**2).sum()))[:, np.newaxis]
                err = (1 - np.sign(np.dot(self.fvDef, self.u) * self.fv1)).sum() / (2*self.npt)
                logging.info('Training Error {0:0f}'.format(err))
                if self.testSet is not None:
                    testRes = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, withPointSet=self.testSet[0])
                    test_err = (1 - np.sign(np.dot(testRes[1][-1,...], self.u) * self.testSet[1])).sum() \
                               / (2*self.testSet[0].shape[0])
                    logging.info('Testing Error {0:0f}'.format(test_err))
                if self.pplot:
                    JJ = np.argpartition(np.ravel(self.u), self.dim-2)
                    fig = plt.figure(4)
                    fig.clf()
                    i1 = JJ[self.dim-2]
                    i2 = JJ[self.dim-1]
                    for k in range(self.npt):
                        if self.fv1[k] > 0:
                            plt.plot([self.fvDef[k, i1]], [self.fvDef[k, i2]], 'ro')
                        else:
                            plt.plot([self.fvDef[k, i1]], [self.fvDef[k, i2]], 'bo')
                    if self.testSet is not None:
                        for k in range(self.testSet[0].shape[0]):
                            if self.testSet[1][k] > 0:
                                plt.plot([testRes[1][-1, k, i1]], [testRes[1][-1, k, i2]], 'r*')
                            else:
                                plt.plot([testRes[1][-1, k, i1]], [testRes[1][-1, k, i2]], 'b*')
                    plt.pause(0.1)

        else:
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef = np.copy(np.squeeze(self.xt[-1, :, :]))



    def endOfProcedure(self):
        self.endOfIteration()

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


if __name__=="__main__":

    outputDir = '/Users/younes/Development/Results/Classif'
    loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput=True)

    N = 100
    d = 10
    #typeData = 'helixes'
    typeData = 'csv1'

    if typeData == 'helixes':
        h = 0
        x0Tr = 0.05*np.random.randn(2*N,d)
        x0Te = 0.05*np.random.randn(2*N,d)
        #x1 = np.random.randn(100,2)
        x1Tr = np.ones((2*N,1))
        x1Te = np.ones((2*N,1))
        x1Tr[N:2*N] = -1
        x1Te[N:2*N] = -1
        t = 2*np.pi*np.random.rand(N)
        x0Tr[0:N,2] += np.cos(t)
        x0Tr[0:N,3] += np.sin(t)
        x0Tr[0:N,4] += h*t

        t = 2*np.pi*np.random.rand(N)
        x0Te[0:N,2] += np.cos(t)
        x0Te[0:N,3] += np.sin(t)
        x0Te[0:N,4] += h*t

        t = 2*np.pi*np.random.rand(N)
        x0Tr[N:2*N,2] += h*t
        x0Tr[N:2*N,3] += 1 + np.cos(t)
        x0Tr[N:2*N,4] += np.sin(t)

        t = 2*np.pi*np.random.rand(N)
        x0Te[N:2*N,2] += h*t
        x0Te[N:2*N,3] += 1 + np.cos(t)
        x0Te[N:2*N,4] += np.sin(t)
    elif typeData == 'csv1':
        nv = -1
        X = np.genfromtxt('/Users/younes/Development/Data/Classification/BRCA1_q2_HW2.csv', delimiter=',')
        x1 = 2*X[0,:].T -1
        x0 = X[1:nv,:].T
        J = np.random.random(x0.shape[0])
        x0Tr = x0[J>0.5,:]
        x1Tr = x1[J>0.5, np.newaxis]
        s = np.sqrt((x0Tr**2).sum(axis=0))
        x0Tr /= s
        x0Te = x0[J<=0.5,:]/s
        x1Te = x1[J<=0.5, np.newaxis]
        d = x0Tr.shape[1]
        # xTmp = np.zeros((x0Tr.shape[0],2*d))
        # xTmp[:,range(0,2*d,2)] = x0Tr
        # xTmp[:,range(1,2*d,2)] = 0.1 * np.random.randn(x0Tr.shape[0],d)
        # x0Tr = xTmp
        # xTmp = np.zeros((x0Te.shape[0],2*d))
        # xTmp[:,range(0,2*d,2)] = x0Te
        # x0Te = xTmp
        # d *= 2
    else:
        Cov0 = np.eye(d)
        m0 = np.concatenate((np.ones(3), np.zeros(d-3)))
        q = np.arange(0,1,1.0/d)
        Cov1 = 2*np.exp(-np.abs(q[:,np.newaxis]-q[np.newaxis,:]))
        #Cov1 = np.eye(d)
        m1 = np.concatenate((-np.ones(3), np.zeros(d-3)))
        x0Tr = np.zeros((2*N,d))
        x0Te = np.zeros((2*N,d))
        x0Tr[0:N, :] = np.random.multivariate_normal(m0, Cov0, size=N)
        x0Te[0:N, :] = np.random.multivariate_normal(m0, Cov0, size=N)
        x0Tr[N:2*N, :] = np.random.multivariate_normal(m1, Cov1, size=N)
        x0Te[N:2*N, :] = np.random.multivariate_normal(m1, Cov1, size=N)
        x1Tr = np.ones((2 * N,1))
        x1Te = np.ones((2 * N,1))
        x1Tr[N:2 * N] = -1
        x1Te[N:2 * N] = -1
        #err0 = (1 - (np.sign(x0Te*x1Te))).sum()/(4*N)
        u = np.mean(x0Tr[0:N,:], axis=0) - np.mean(x0Tr[N:2*N, :], axis=0)
        err1 = (1 - np.sign(np.dot(x0Te, u[:, np.newaxis]) * x1Te)).sum() / (4*N)
        print 'LDA Error: {0:2f}'.format(err1)

    #x0[:,2] = 0

    K1 = kfun.Kernel(name='laplacian', sigma=.25, order=3)
    sm = PointSetMatchingParam(timeStep=0.1, KparDiff = K1, sigmaError=.5, errorType='classification')

    f = PointSetMatching(Template=x0Tr, Target=x1Tr, outputDir=outputDir, param=sm, regWeight=1.,
                        saveTrajectories=True, pplot=True, testSet=(x0Te, x1Te),
                        affine='none', testGradient=True, affineWeight=1e3,
                        maxIter=5)
    K1.localMaps = (np.zeros(3*d-2, dtype=int), 3*np.ones(d, dtype=int))
    K1.localMaps[1][0] = 2
    K1.localMaps[1][d-1] = 2
    KJ = np.random.permutation(d)
    jK=0
    for k in range(d):
        if k>0:
            K1.localMaps[0][jK] = KJ[k-1]
            jK += 1
        K1.localMaps[0][jK] = KJ[k]
        jK += 1
        if k<d-1:
            K1.localMaps[0][jK] = KJ[k+1]
            jK += 1
        # K1.localMaps[0][4*k] = 2*k+1
        # K1.localMaps[0][4 * k+1] = 2 * k + 2
        # K1.localMaps[0][4 * k + 2] = 2 * k + 2
        # K1.localMaps[0][4 * k + 3] = 2 * k + 1

    for k in range(10):
        f.optimizeMatching()
        J1 = x1Tr > 0
        J2 = x1Tr < 0
        for k in range(250):
            g = (f.fvDef *
                 (f.fv1 * np.exp(-(np.dot(f.fvDef, f.u) * f.fv1)))).sum(axis=0)[:, np.newaxis]
            # g -= (g * self.u).sum() * self.u
            ep = .1 / f.fvDef.size
            fu = f.u + ep * g
            ll = ep * f.l1Cost
            fu = np.sign(fu) * np.maximum(np.fabs(fu) - ll, 0)
            # fu = (fu / np.sqrt((fu ** 2).sum()))
            # cl0 = pointSets.classScore(self.fvDef, self.fv1, self.u)
            # while ep > 1e-10 and pointSets.classScore(self.fvDef, self.fv1, fu) > cl0:
            #     ep *= 0.5
            #     fu = self.u + ep * g
            #     fu = fu / np.sqrt((fu ** 2).sum())
            f.u = fu
        # g = (f.fvDef *(x1Tr*np.exp(-(np.dot(f.fvDef,f.u)*x1Tr)))).sum(axis = 0)[:, np.newaxis]
        # g -= (g*f.u).sum()*f.u
        # ep = 1.
        # fu = f.u + ep * g
        # fu = (fu / np.sqrt((fu**2).sum()))
        # cl0 = pointSets.classScore(f.fvDef, x1Tr, f.u)
        # while ep >1e-10 and pointSets.classScore(f.fvDef, x1Tr, fu) > cl0:
        #     ep *= 0.5
        #     fu = f.u + ep * g
        #     fu = fu / np.sqrt((fu**2).sum())
        # f.u = fu
        print 'ep = ', ep
        #f.u = (np.mean(f.fvDef[J1[:,0], :], axis=0) - np.mean(f.fvDef[J2[:,0], :], axis=0))[:, np.newaxis]
        f.obj = None
        f.objectiveFun()
    plt.pause(1000)

