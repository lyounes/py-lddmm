import os
import numpy as np
import numpy.linalg as la
import logging
from base import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, bfgs
import base.surfaces as surfaces
from base import pointSets
from base.affineBasis import AffineBasis, getExponential, gradExponential
from functools import partial
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParam:
    def __init__(self, timeStep = .1, algorithm='bfgs', Wolfe=True, KparDiff = None, KparDist = None, sigmaKernel = 6.5, sigmaDist = 2.5,
                 sigmaError = 1.0, errorType = 'measure',  typeKernel='gauss', internalCost = None):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.sigmaDist = sigmaDist
        self.sigmaError = sigmaError
        self.typeKernel = typeKernel
        self.errorType = errorType
        self.algorithm = algorithm
        self.wolfe = Wolfe
        if KparDiff is None:
            self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        else:
            self.KparDiff = KparDiff
        if KparDist is None:
            self.KparDist = kfun.Kernel(name = 'gauss', sigma = self.sigmaDist)
        else:
            self.KparDist = KparDist
        self.internalCost = internalCost

class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []
        self.initx = []


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
class SurfaceMatching(object):

    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
                 subsampleTargetSize=-1,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):
        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        if self.param.algorithm == 'bfgs':
             self.euclideanGradient = True
        else:
            self.euclideanGradient = False

        self.set_fun(self.param.errorType)


        if Template==None:
            if fileTempl==None:
                logging.error('Please provide a template surface')
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileTempl)
        else:
            self.fv0 = surfaces.Surface(surf=Template)

        if self.param.errorType != 'currentMagnitude':
            if Target==None:
                if fileTarg==None:
                    logging.error('Please provide a target surface')
                    return
                else:
                    if self.param.errorType == 'L2Norm':
                        self.fv1 = surfaces.Surface()
                        self.fv1.readFromImage(fileTarg)
                    else:
                        self.fv1 = surfaces.Surface(filename=fileTarg)
            else:
                if self.param.errorType == 'L2Norm':
                    self.fv1 = surfaces.Surface()
                    self.fv1.initFromImage(fileTarg)
                else:
                    self.fv1 = surfaces.Surface(surf=Target)
        else:
            self.fv1 = None

            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()

        self.saveRate = 10
        self.randomInit = True
        self.iter = 0
        self.setOutputDir(outputDir)
        self.dim = self.fv0.vertices.shape[1]
        self.maxIter = maxIter
        self.verb = verb
        self.saveTrajectories = saveTrajectories
        self.symmetric = symmetric
        self.testGradient = testGradient
        self.internalWeight = internalWeight
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


                    

        if self.param.internalCost == 'h1':
            self.internalCost = surfaces.normGrad
            self.internalCostGrad = surfaces.diffNormGrad
#        elif self.param.internalCost == 'h1Invariant':
#            if self.fv0.vertices.shape[1] == 2:
#                self.internalCost = curves.normGradInvariant 
#                self.internalCostGrad = curves.diffNormGradInvariant
#            else:
#                self.internalCost = curves.normGradInvariant3D
#                self.internalCostGrad = curves.diffNormGradInvariant3D
        else:
            self.internalCost = None

        self.fvInit = surfaces.Surface(surf=self.fv0)
        if self.fv1:
            self.fv0.getEdges()
            self.fv1.getEdges()
            self.closed = self.fv0.bdry.max() == 0 and self.fv1.bdry.max() == 0
            if self.closed:
                v0 = self.fv0.surfVolume()
                if self.param.errorType == 'L2Norm' and v0 < 0:
                    self.fv0.flipFaces()
                    v0 = -v0
                v1 = self.fv1.surfVolume()
                if (v0*v1 < 0):
                    self.fv1.flipFaces()
            if self.closed:
                z= self.fvInit.surfVolume()
                if (z < 0):
                    self.fv0ori = -1
                else:
                    self.fv0ori = 1

                z= self.fv1.surfVolume()
                if (z < 0):
                    self.fv1ori = -1
                else:
                    self.fv1ori = 1
            else:
                self.fv0ori = 1
                self.fv1ori = 1
        else:
            self.fv0ori = 1
            self.fv1ori = 1
        #self.fv0Fine = surfaces.Surface(surf=self.fv0)
        if (subsampleTargetSize > 0):
            self.fvInit.Simplify(subsampleTargetSize)
            logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))

        self.x0 = np.copy(self.fvInit.vertices)
        self.x0try = np.copy(self.fvInit.vertices)
        self.fvDef = surfaces.Surface(surf=self.fvInit)
        self.npt = self.fvInit.vertices.shape[0]


        logging.info('orientation: {0:d}'.format(self.fv0ori))

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.randomInit:
            self.at = np.random.normal(0, 1, self.at.shape)
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.x0.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        if self.fv1:
            self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        self.coeffAff1 = 1.
        self.coeffAff2 = 100.
        self.coeffAff = self.coeffAff1
        self.coeffInitx = .1
        self.affBurnIn = 25
        self.pplot = pplot
        if self.pplot:
            fig=plt.figure(3)
            #fig.clf()
            ax = Axes3D(fig)
            lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
            if self.fv1:
                lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
            else:
                lim0 = lim1
    #            ax.plot_trisurf(self.fv1.vertices[self.fv1.faces[:,0],:], self.fv1.vertices[self.fv1.faces[:,1],:],
#                           self.fv1.vertices[self.fv1.faces[:,2],:])# color=[0,0,1])
            #plt.axis('equal')
            ax.set_xlim(min(lim0[0][0],lim1[0][0]), max(lim0[0][1],lim1[0][1]))
            ax.set_ylim(min(lim0[1][0],lim1[1][0]), max(lim0[1][1],lim1[1][1]))
            ax.set_zlim(min(lim0[2][0],lim1[2][0]), max(lim0[2][1],lim1[2][1]))
            plt.pause(0.1)


    def set_fun(self, errorType):
        self.param.errorType = errorType
        if errorType == 'current':
            print('Running Current Matching')
            self.fun_obj0 = partial(surfaces.currentNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(surfaces.currentNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(surfaces.currentNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType == 'currentMagnitude':
            print('Running Current Matching')
            self.fun_obj0 = lambda fv1 : 0
            self.fun_obj = partial(surfaces.currentMagnitude, KparDist=self.param.KparDist)
            self.fun_objGrad = partial(surfaces.currentMagnitudeGradient, KparDist=self.param.KparDist)
            # self.fun_obj0 = curves.currentNorm0
            # self.fun_obj = curves.currentNormDef
            # self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            print('Running Measure Matching')
            self.fun_obj0 = partial(surfaces.measureNorm0, KparDist=self.param.KparDist)
            self.fun_obj = partial(surfaces.measureNormDef,KparDist=self.param.KparDist)
            self.fun_objGrad = partial(surfaces.measureNormGradient,KparDist=self.param.KparDist)
        elif errorType=='varifold':
            self.fun_obj0 = partial(surfaces.varifoldNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(surfaces.varifoldNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(surfaces.varifoldNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType == 'L2Norm':
            self.fun_obj0 = None
            self.fun_obj = None
            self.fun_objGrad = None
        else:
            print('Unknown error Type: ', self.param.errorType)

    def addSurfaceToPlot(self, fv1, ax, ec = 'b', fc = 'r', al=.5, lw=1):
        x = fv1.vertices[fv1.faces[:,0],:]
        y = fv1.vertices[fv1.faces[:,1],:]
        z = fv1.vertices[fv1.faces[:,2],:]
        a = np.concatenate([x,y,z], axis=1)
        poly = [ [a[i,j*3:j*3+3] for j in range(3)] for i in range(a.shape[0])]
        tri = Poly3DCollection(poly, alpha=al, linewidths=lw)
        tri.set_edgecolor(ec)
        tri.set_facecolor(fc)
        ax.add_collection3d(tri)
        xlim = [fv1.vertices[:,0].min(),fv1.vertices[:,0].max()]
        ylim = [fv1.vertices[:,1].min(),fv1.vertices[:,1].max()]
        zlim = [fv1.vertices[:,2].min(),fv1.vertices[:,2].max()]
        return [xlim, ylim, zlim]

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef, _fvInit = None):
        if self.param.errorType == 'L2Norm':
            obj = surfaces.L2Norm(_fvDef, self.fv1.vfld) / (self.param.sigmaError ** 2)
        else:
            obj = self.fun_obj(_fvDef, self.fv1) / (self.param.sigmaError**2)
            if _fvInit is not None:
                obj += self.fun_obj(_fvInit, self.fv0) / (self.param.sigmaError**2)
        #print 'dataterm = ', obj + self.obj0
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
        foo = surfaces.Surface(surf=self.fv0)
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        if withJacobian:
            (xt,Jt)  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A, withJacobian=True)
        else:
            xt  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A)
        #print xt[-1, :, :]
        #print obj
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
            if self.internalCost:
                foo.updateVertices(z)
                obj += self.internalWeight*regWeight*self.internalCost(foo, ra)*timeStep

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
            if self.param.errorType == 'L2Norm':
                self.obj0 = surfaces.L2Norm0(self.fv1) / (self.param.sigmaError ** 2)
            else:
                self.obj0 = self.fun_obj0(self.fv1) / (self.param.sigmaError**2)
            if self.symmetric:
                self.obj0 += self.fun_obj0(self.fv0) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            #foo = surfaces.Surface(surf=self.fvDef)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            if self.symmetric:
                self.fvInit.updateVertices(np.squeeze(self.x0))
            #foo.computeCentersAreas()
            if self.symmetric:
                self.obj += self.obj0 + self.dataTerm(self.fvDef, self.fvInit)
            else:
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
        if self.symmetric:
            x0Try = self.x0 - eps * dir.initx
        else:
            x0Try = self.x0

        foo = self.objectiveFunDef(atTry, AfftTry, x0 = x0Try, withTrajectory=True)
        objTry += foo[0]

        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
        if self.symmetric:
            ffI = surfaces.Surface(surf=self.fvInit)
            ffI.updateVertices(x0Try)
            objTry += self.dataTerm(ff, ffI)
        else:
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
        ff = surfaces.Surface(surf=self.fvDef)
        dff = np.random.normal(size=ff.vertices.shape)
        eps = 1e-6
        ff.updateVertices(ff.vertices+eps*dff)
        c1 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps, (grd*dff).sum()) )

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
        if self.param.errorType == 'L2Norm':
            px = surfaces.L2NormGradient(endPoint, self.fv1.vfld)
        else:
            if self.fv1:
                px = self.fun_objGrad(endPoint, self.fv1)
            else:
                px = self.fun_objGrad(endPoint)
        return px / self.param.sigmaError**2

    def initPointGradient(self):
        px = self.fun_objGrad(self.fvInit, self.fv0, self.param.KparDist)
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
        foo = surfaces.Surface(surf=self.fv0)
        for t in range(M):
            px = np.squeeze(pxt[M-t, :, :])
            z = np.squeeze(xt[M-t-1, :, :])
            a = np.squeeze(at[M-t-1, :, :])
            foo.updateVertices(z)
            v = KparDiff.applyK(z,a)
            if self.internalCost:
                grd = self.internalCostGrad(foo, v)
                Lv =  grd[0]
                DLv = self.internalWeight*regWeight*grd[1]
#                Lv = -2*foo.laplacian(v) 
#                DLv = self.internalWeight*foo.diffNormGrad(v)
                a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*regWeight*a[np.newaxis,...], 
                                     -self.internalWeight*regWeight*a[np.newaxis,...], Lv[np.newaxis,...]))
                a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...], Lv[np.newaxis,...],
                                     -self.internalWeight*regWeight*a[np.newaxis,...]))
                zpx = KparDiff.applyDiffKT(z, a1, a2) - DLv
            else:
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
        if not self.internalCost:
            return evol.landmarkHamiltonianGradient(x0, at, px1, kernel, regWeight, affine=affine, 
                                                    getCovector=True)
                                                    
        foo = surfaces.Surface(surf=self.fv0)
        (pxt, xt) = self.hamiltonianCovector(x0, at, px1, kernel, regWeight, affine=affine)
        #at = self.at        
        dat = np.zeros(at.shape)
        timeStep = 1.0/at.shape[0]
        if not (affine is None):
            A = affine[0]
            dA = np.zeros(affine[0].shape)
            db = np.zeros(affine[1].shape)
        for k in range(at.shape[0]):
            z = np.squeeze(xt[k,...])
            foo.updateVertices(z)
            a = np.squeeze(at[k, :, :])
            px = np.squeeze(pxt[k+1, :, :])
            #print 'testgr', (2*a-px).sum()
            v = kernel.applyK(z,a)
            if self.internalCost:
                Lv = self.internalCostGrad(foo, v, variables='phi') 
                #Lv = -foo.laplacian(v) 
                dat[k, :, :] = 2*regWeight*a-px + self.internalWeight * regWeight * Lv
            else:
                dat[k, :, :] = 2*regWeight*a-px

            if not (affine is None):
                dA[k] = gradExponential(A[k]*timeStep, px, xt[k]) #.reshape([self.dim**2, 1])/timeStep
                db[k] = pxt[k+1].sum(axis=0) #.reshape([self.dim,1]) 
#        if not self.internalCost:
#            foo = evol.landmarkHamiltonianGradient(x0, at, px1, kernel, regWeight, affine=affine, 
#                                                    getCovector=True)
#            tk = -1
#            print 'dat', np.fabs(dat[tk,...]-foo[0][tk,...]).sum()
#            print 'xt', np.fabs(xt[tk,...]-foo[-2][tk,...]).sum()
#            print 'pxt', np.fabs(pxt[tk,...]-foo[-1][tk,...]).sum()

        if affine is None:
            return dat, xt, pxt
        else:
            return dat, dA, db, xt, pxt

    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            at = self.at
            endPoint = self.fvDef
            A = self.affB.getTransforms(self.Afft)
        else:
            A = self.affB.getTransforms(self.Afft - update[1]*update[0].aff)
            at = self.at - update[1] *update[0].diff
            xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine=A)
            endPoint = surfaces.Surface(surf=self.fv0)
            endPoint.updateVertices(xt[-1, :, :])


        px1 = -self.endPointGradient(endPoint=endPoint)
        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, self.Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
        foo = self.hamiltonianGradient(px1, at=at, affine=A)
        grd = Direction()
        if self.euclideanGradient:
            grd.diff = np.zeros(foo[0].shape)
            for t in range(self.Tsize):
                z = self.xt[t, :, :]
                grd.diff[t,:,:] = self.param.KparDiff.applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
        else:
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
        if self.symmetric:
            grd.initx = (self.initPointGradient() - foo[-1][0,...])/(self.coeffInitx * coeff)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.aff = dir1.aff + beta * dir2.aff
        dir.initx = dir1.initx + beta * dir2.initx
        return dir

    def prod(self, dir1, beta):
        dir = Direction()
        dir.diff = beta * dir1.diff
        dir.aff = beta * dir1.aff
        dir.initx = beta * dir1.initx
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)
        dir.aff = np.copy(dir0.aff)
        dir.initx = np.copy(dir0.initx)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
        dirfoo.initx = np.random.randn(self.npt, self.dim)
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

        if self.symmetric:
            for ll,gr in enumerate(g2):
                res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx

        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[t, :, :])
            u = np.squeeze(g1.diff[t, :, :])
            uu = (g1.aff[t]*self.affineWeight.reshape(g1.aff[t].shape))
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  += (ggOld*u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr.aff[t]).sum()
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1
        if self.symmetric:
            for ll,gr in enumerate(g2):
                res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx
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
        if self.internalCost and self.testGradient:
            Phi = np.random.normal(size=self.x0.shape)
            dPhi1 = np.random.normal(size=self.x0.shape)
            dPhi2 = np.random.normal(size=self.x0.shape)
            eps = 1e-6
            fv22 = surfaces.Surface(surf=self.fvDef)
            fv22.updateVertices(self.fvDef.vertices+eps*dPhi2)
            e0 = self.internalCost(self.fvDef, Phi)
            e1 = self.internalCost(self.fvDef, Phi+eps*dPhi1)
            e2 = self.internalCost(fv22, Phi)
            grad = self.internalCostGrad(self.fvDef, Phi)
            print('Laplacian:', (e1-e0)/eps, (grad[0]*dPhi1).sum())
            print('Gradient:', (e2-e0)/eps, (grad[1]*dPhi2).sum())
            
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0) :
            logging.info('Saving surfaces...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            if not self.internalCost and self.affineDim <=0:
                xtEPDiff, atEPdiff = evol.landmarkEPDiff(self.at.shape[0], self.x0,
                                                         np.squeeze(self.at[0, :, :]), self.param.KparDiff)
                self.fvDef.updateVertices(np.squeeze(xtEPDiff[-1, :, :]))
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+'EPDiff.vtk')
                logging.info('EPDiff difference %f' %(np.fabs(self.xt[-1,:,:] - xtEPDiff[-1,:,:]).sum()) )

            if self.saveTrajectories:
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curves.vtk', self.xt)

            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)
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
                f = surfaces.Surface(surf=self.fvInit)
                X = self.affB.integrateFlow(self.Afft)
                displ = np.zeros(self.x0.shape[0])
                dt = 1.0 /self.Tsize
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yyt = np.dot(self.xt[t,...] - X[1][t, ...], U.T)
                    zt = np.dot(xt[t,...] - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        at = np.dot(self.at[t,...], U.T)
                        vt = self.param.KparDiff.applyK(yyt, at, firstVar=zt)
                    f.updateVertices(yyt)
                    vf = surfaces.vtkFields()
                    vf.scalars.append('Jacobian')
                    vf.scalars.append(np.exp(Jt[t, :]))
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity')
                    vf.vectors.append(vt)
                    nu = self.fv0ori*f.computeVertexNormals()
                    f.saveVTK2(self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.vtk', vf)
                    displ += dt * (vt*nu).sum(axis=1)
                f = surfaces.Surface(surf=self.fv1)
                yyt = np.dot(f.vertices - X[1][-1, ...], U.T)
                f.updateVertices(yyt)
                f.saveVTK(self.outputDir +'/TargetCorrected.vtk')
            fvDef = surfaces.Surface(surf=self.fvInit)
            AV0 = fvDef.computeVertexArea()
            nu = self.fv0ori*self.fvInit.computeVertexNormals()
            v = self.v[0,...]
            displ = np.zeros(self.npt)
            dt = 1.0 /self.Tsize
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(xt[kk, :, :]))
                AV = fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0])
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jt[kk, :]))
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(np.exp(Jt[kk, :])/AV)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if kk < self.Tsize:
                    nu = self.fv0ori*self.fvDef.computeVertexNormals()
                    v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                vf.vectors.append('velocity')
                vf.vectors.append(self.v[kkm,:])
                fvDef.saveVTK2(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', vf)
                displ += dt * (v*nu).sum(axis=1)
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')
        else:
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)
        if self.pplot:
            fig=plt.figure(4)
            #fig.clf()
            ax = Axes3D(fig)
            lim0 = self.addSurfaceToPlot(self.fv1, ax, ec = 'k', fc = 'b')
            lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
            ax.set_xlim(min(lim0[0][0],lim1[0][0]), max(lim0[0][1],lim1[0][1]))
            ax.set_ylim(min(lim0[1][0],lim1[1][0]), max(lim0[1][1],lim1[1][1]))
            ax.set_zlim(min(lim0[2][0],lim1[2][0]), max(lim0[2][1],lim1[2][1]))
            #plt.axis('equal')
            plt.pause(0.1)



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
        if self.param.algorithm == 'cg':
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1)
        elif self.param.algorithm == 'bfgs':
            bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
                      Wolfe=self.param.wolfe, memory=25)
        #return self.at, self.xt

