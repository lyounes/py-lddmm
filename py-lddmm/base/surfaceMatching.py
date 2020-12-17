import os
import numpy as np
import numpy.linalg as la
import logging
import h5py
import glob
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, bfgs
from . import surfaces
from . import pointSets
from .affineBasis import AffineBasis, getExponential, gradExponential
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParam:
    def __init__(self, timeStep = .1, algorithm='cg', Wolfe=True, KparDiff = None, KparDist = None, sigmaKernel = 6.5,
                 sigmaDist = 2.5,
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
            #self.KparDist = kfun.Kernel(name = 'laplacian', order=1, sigma = self.sigmaDist)
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
    def __init__(self, Template=None, Target=None, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
                 subsampleTargetSize=-1, affineOnly = False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):
        if param is None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        if self.param.algorithm == 'cg':
             self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.fv0 = None
        self.fv1 = None
        self.fvInit = None
        self.dim = 0
        self.fun_obj = None
        self.fun_obj0 = None
        self.fun_objGrad = None
        self.obj0 = 0
        self.coeffAff = 1
        self.obj = 0
        self.xt = None

        self.setOutputDir(outputDir)
        self.set_fun(self.param.errorType)

        self.set_template_and_target(Template, Target, subsampleTargetSize)



        self.set_parameters(maxIter=maxIter, regWeight = regWeight, affineWeight = affineWeight,
                            internalWeight=internalWeight, verb=verb, affineOnly = affineOnly,
                            rotWeight = rotWeight, scaleWeight = scaleWeight, transWeight = transWeight,
                            symmetric = symmetric, testGradient=testGradient, saveFile = saveFile,
                            saveTrajectories = saveTrajectories, affine = affine)
        self.initialize_variables()
        self.gradCoeff = self.x0.shape[0]

        self.pplot = pplot
        if self.pplot:
            self.initial_plot()


    def set_parameters(self, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
                 affineOnly = False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none'):
        self.saveRate = 10
        self.gradEps = -1
        self.randomInit = False
        self.iter = 0
        self.maxIter = maxIter
        self.verb = verb
        self.saveTrajectories = saveTrajectories
        self.symmetric = symmetric
        self.testGradient = testGradient
        self.internalWeight = internalWeight
        self.regweight = regWeight

        self.affineOnly = affineOnly
        self.affine = affine
        self.affB = AffineBasis(self.dim, affine)
        self.affineDim = self.affB.affineDim
        self.affineBasis = self.affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(self.affB.rotComp) > 0) and (rotWeight is not None):
            self.affineWeight[self.affB.rotComp] = rotWeight
        if (len(self.affB.simComp) > 0) and (scaleWeight is not None):
            self.affineWeight[self.affB.simComp] = scaleWeight
        if (len(self.affB.transComp) > 0) and (transWeight is not None):
            self.affineWeight[self.affB.transComp] = transWeight

        if self.param.internalCost == 'h1':
            self.internalCost = surfaces.normGrad
            self.internalCostGrad = surfaces.diffNormGrad
        else:
            self.internalCost = None



        self.obj = None
        self.objTry = None
        self.saveFile = saveFile
        self.coeffAff1 = 1.
        if self.param.algorithm == 'cg':
            self.coeffAff2 = 100.
        else:
            self.coeffAff2 = 1.
        self.coeffAff = self.coeffAff1
        self.coeffInitx = .1
        self.affBurnIn = 25
        self.forceLineSearch = False


    def set_template_and_target(self, Template, Target, subsampleTargetSize=-1):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = surfaces.Surface(surf=Template)

        if self.param.errorType != 'currentMagnitude':
            if Target is None:
                logging.error('Please provide a target surface')
                return
            else:
                if self.param.errorType == 'L2Norm':
                    self.fv1 = surfaces.Surface()
                    self.fv1.readFromImage(Target)
                else:
                    self.fv1 = surfaces.Surface(surf=Target)
        else:
            self.fv1 = None
        self.fvInit = surfaces.Surface(surf=self.fv0)
        self.fix_orientation()
        if subsampleTargetSize > 0:
            self.fvInit.Simplify(subsampleTargetSize)
            logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        if self.fv1:
            self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        self.dim = self.fv0.vertices.shape[1]


    def fix_orientation(self):
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
                if v0*v1 < 0:
                    self.fv1.flipFaces()
            if self.closed:
                z= self.fvInit.surfVolume()
                if z < 0:
                    self.fv0ori = -1
                else:
                    self.fv0ori = 1

                z= self.fv1.surfVolume()
                if z < 0:
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
        logging.info('orientation: {0:d}'.format(self.fv0ori))


    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.x0 = np.copy(self.fvInit.vertices)
        self.x0try = np.copy(self.fvInit.vertices)
        self.fvDef = surfaces.Surface(surf=self.fvInit)
        self.npt = self.fvInit.vertices.shape[0]


        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.randomInit:
            self.at = np.random.normal(0, 1, self.at.shape)
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])


    def initial_plot(self):
        fig = plt.figure(3)
        ax = Axes3D(fig)
        lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r')
        if self.fv1:
            lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
        else:
            lim0 = lim1
        ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
        ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
        ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
        fig.canvas.flush_events()

    def set_fun(self, errorType):
        self.param.errorType = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj0 = partial(surfaces.currentNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(surfaces.currentNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(surfaces.currentNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType == 'currentMagnitude':
            #print('Running Current Matching')
            self.fun_obj0 = lambda fv1 : 0
            self.fun_obj = partial(surfaces.currentMagnitude, KparDist=self.param.KparDist)
            self.fun_objGrad = partial(surfaces.currentMagnitudeGradient, KparDist=self.param.KparDist)
            # self.fun_obj0 = curves.currentNorm0
            # self.fun_obj = curves.currentNormDef
            # self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            #print('Running Measure Matching')
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
        return fv1.addToPlot(ax, ec = ec, fc = fc, al=al, lw=lw)

    def setOutputDir(self, outputDir, clean=True):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)

        if clean:
            fileList = glob.glob(outputDir + '/*.vtk')
            for f in fileList:
                os.remove(f)


    def dataTerm(self, _fvDef, _fvInit = None):
        if self.param.errorType == 'L2Norm':
            obj = surfaces.L2Norm(_fvDef, self.fv1.vfld) / (self.param.sigmaError ** 2)
        else:
            obj = self.fun_obj(_fvDef, self.fv1) / (self.param.sigmaError**2)
            if _fvInit is not None:
                obj += self.fun_obj(_fvInit, self.fv0) / (self.param.sigmaError**2)
        #print 'dataterm = ', obj + self.obj0
        return obj

    def  objectiveFunDef(self, at, Afft=None, kernel = None, withTrajectory = False, withJacobian=False,
                         fv0 = None, regWeight = None):
        if fv0 is None:
            fv0 = self.fv0
        x0 = fv0.vertices
        if kernel is None:
            kernel = self.param.KparDiff
        #print 'x0 fun def', x0.sum()
        if regWeight is None:
            regWeight = self.regweight
        timeStep = 1.0/self.Tsize
        if Afft is not None:
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
        else:
            A = None
        if withJacobian:
            (xt,Jt)  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A, withJacobian=True)
        else:
            xt  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A)
        #print xt[-1, :, :]
        #print obj
        obj=0
        obj1 = 0
        obj2 = 0
        foo = surfaces.Surface(surf=fv0)
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :, :])
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            ra = kernel.applyK(z, a)
            if hasattr(self, 'v'):  
                self.v[t, :] = ra
            obj += regWeight*timeStep*np.multiply(a, ra).sum()
            if self.internalCost:
                foo.updateVertices(z)
                obj1 += self.internalWeight*self.internalCost(foo, ra)*timeStep

            if self.affineDim > 0:
                obj2 +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        #print(obj, obj1, obj2)
        obj += obj1 + obj2
        if withJacobian:
            return obj, xt, Jt
        elif withTrajectory:
            return obj, xt
        else:
            return obj


    def objectiveFun(self):
        if self.obj is None:
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

    def updateTry(self, dr, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dr.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dr.aff
        else:
            AfftTry = self.Afft
        if self.symmetric:
            x0Try = self.x0 - eps * dr.initx
        else:
            x0Try = self.x0

        fv0 = surfaces.Surface(surf=self.fv0)
        fv0.updateVertices(x0Try)
        foo = self.objectiveFunDef(atTry, AfftTry, fv0 = fv0, withTrajectory=True)
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

        if (objRef is None) or (objTry < objRef):
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
    
    
    def hamiltonianCovector(self, at, px1, KparDiff, regWeight, affine = None, fv0 = None):
        if fv0 is None:
            fv0 = self.fvInit
        x0 = fv0.vertices
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
        foo = surfaces.Surface(surf=fv0)
        for t in range(M):
            px = np.squeeze(pxt[M-t, :, :])
            z = np.squeeze(xt[M-t-1, :, :])
            a = np.squeeze(at[M-t-1, :, :])
            foo.updateVertices(z)
            v = KparDiff.applyK(z,a)
            if self.internalCost:
                grd = self.internalCostGrad(foo, v)
                Lv =  grd[0]
                DLv = self.internalWeight*grd[1]
                zpx = self.param.KparDiff.applyDiffKT(z, px, a, regweight=self.regweight, lddmm=True,
                                                      extra_term = -self.internalWeight*Lv) - DLv
            else:
                zpx = self.param.KparDiff.applyDiffKT(z, px, a, regweight=self.regweight, lddmm=True)

            if not (affine is None):
                pxt[M-t-1, :, :] = np.dot(px, A[M-t-1]) + timeStep * zpx
            else:
                pxt[M-t-1, :, :] = px + timeStep * zpx
        return pxt, xt
    
    def hamiltonianGradient(self, px1, kernel = None, affine=None, regWeight=None, fv0=None, at=None):
        if regWeight is None:
            regWeight = self.regweight
        if fv0 is None:
            fv0 = self.fvInit
        x0 = fv0.vertices
        if at is None:
            at = self.at
        if kernel is None:
            kernel  = self.param.KparDiff
        if not self.internalCost:
            return evol.landmarkHamiltonianGradient(x0, at, px1, kernel, regWeight, affine=affine, 
                                                    getCovector=True)
                                                    
        foo = surfaces.Surface(surf=fv0)
        foo .updateVertices(x0)
        (pxt, xt) = self.hamiltonianCovector(at, px1, kernel, regWeight, fv0=foo, affine=affine)
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
            if not self.affineOnly:
                v = kernel.applyK(z,a)
                if self.internalCost:
                    Lv = self.internalCostGrad(foo, v, variables='phi')
                    #Lv = -foo.laplacian(v)
                    dat[k, :, :] = 2*regWeight*a-px + self.internalWeight * Lv
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
            at = self.at - update[1] * update[0].diff
            if len(update[0].aff) > 0:
                A = self.affB.getTransforms(self.Afft - update[1]*update[0].aff)
            else:
                A = None
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
        dr = Direction()
        dr.diff = dir1.diff + beta * dir2.diff
        if self.affineDim > 0:
            dr.aff = dir1.aff + beta * dir2.aff
        dr.initx = dir1.initx + beta * dir2.initx
        return dr

    def prod(self, dir1, beta):
        dr = Direction()
        dr.diff = beta * dir1.diff
        if self.affineDim > 0:
            dr.aff = beta * dir1.aff
        dr.initx = beta * dir1.initx
        return dr

    def copyDir(self, dir0):
        dr = Direction()
        dr.diff = np.copy(dir0.diff)
        if self.affineDim > 0:
            dr.aff = np.copy(dir0.aff)
        dr.initx = np.copy(dir0.initx)
        return dr


    def randomDir(self):
        dirfoo = Direction()
        if self.affineOnly:
            dirfoo.diff = np.zeros((self.Tsize, self.npt, self.dim))
        else:
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
                res[ll]  = res[ll] + (ggOld*u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr.aff[t]).sum() * self.coeffAff
                ll = ll + 1

        if self.symmetric:
            for ll,gr in enumerate(g2):
                res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx

        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            u = np.squeeze(g1.diff[t, :, :])
            if self.affineDim > 0:
                uu = g1.aff[t]
                # uu = (g1.aff[t]*self.affineWeight.reshape(g1.aff[t].shape))
            else:
                uu = 0
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

    def saveCorrectedTarget(self, U, b):
        f = surfaces.Surface(surf=self.fv1)
        yyt = np.dot(f.vertices - b, U)
        f.updateVertices(yyt)
        f.saveVTK(self.outputDir + '/TargetCorrected.vtk')

    def saveCorrectedEvolution(self, fv0, xt, at, Afft, fileName='evolution', Jacobian=None):
        f = surfaces.Surface(surf=fv0)
        X = self.affB.integrateFlow(Afft)
        displ = np.zeros(xt.shape[1])
        dt = 1.0 / self.Tsize
        for t in range(self.Tsize + 1):
            U = la.inv(X[0][t])
            yyt = np.dot(self.xt[t, ...] - X[1][t, ...], U.T)
            zt = np.dot(xt[t, ...] - X[1][t, ...], U.T)
            if t < self.Tsize:
                atCorr = np.dot(at[t, ...], U.T)
                vt = self.param.KparDiff.applyK(yyt, atCorr, firstVar=zt)
            f.updateVertices(yyt)
            vf = surfaces.vtkFields()
            if Jacobian is not None:
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jacobian[t, :, 0]))
            vf.scalars.append('displacement')
            vf.scalars.append(displ)
            vf.vectors.append('velocity')
            vf.vectors.append(vt)
            nu = self.fv0ori * f.computeVertexNormals()
            f.saveVTK2(self.outputDir + '/' + fileName + 'Corrected' + str(t) + '.vtk', vf)
            displ += dt * (vt * nu).sum(axis=1)
        self.saveCorrectedTarget(U.T, X[1][-1, ...])

    def saveEvolution(self, fv0, xt, Jacobian=None, fileName='evolution', velocity = None, orientation= None,
                      with_area_displacement=False):
        if velocity is None:
            velocity = self.v
        if orientation is None:
            orientation = self.fv0ori
        fvDef = surfaces.Surface(surf=fv0)
        AV0 = fvDef.computeVertexArea()
        nu = orientation * fv0.computeVertexNormals()
        v = velocity[0, ...]
        npt = fv0.vertices.shape[0]
        displ = np.zeros(npt)
        area_displ = np.zeros((self.Tsize + 1, npt))
        dt = 1.0 / self.Tsize
        for kk in range(self.Tsize + 1):
            fvDef.updateVertices(np.squeeze(xt[kk, :, :]))
            AV = fvDef.computeVertexArea()
            AV = (AV[0] / AV0[0])
            vf = surfaces.vtkFields()
            if Jacobian is not None:
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jacobian[kk, :, 0]))
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(np.exp(Jacobian[kk, :, 0]) / AV)
            vf.scalars.append('displacement')
            vf.scalars.append(displ)
            if kk < self.Tsize:
                nu = orientation * fvDef.computeVertexNormals()
                v = velocity[kk, ...]
                kkm = kk
            else:
                kkm = kk - 1
            vf.vectors.append('velocity')
            vf.vectors.append(velocity[kkm, :])
            if with_area_displacement and kk > 0:
                area_displ[kk, :] = area_displ[kk - 1, :] + dt * ((AV + 1) * (v * nu).sum(axis=1))[np.newaxis, :]
            fvDef.saveVTK2(self.outputDir + '/' + fileName + str(kk) + '.vtk', vf)
            displ += dt * (v * nu).sum(axis=1)

    def saveEPDiff(self, fv0, at, fileName='evolution'):
        xtEPDiff, atEPdiff = evol.landmarkEPDiff(at.shape[0], fv0.vertices,
                                                 np.squeeze(at[0, :, :]), self.param.KparDiff)
        fvDef = surfaces.Surface(surf=fv0)
        fvDef.updateVertices(np.squeeze(xtEPDiff[-1, :, :]))
        fvDef.saveVTK(self.outputDir + '/' + fileName + 'EPDiff.vtk')
        return xtEPDiff, atEPdiff


    def endOfIteration(self):
        self.iter += 1
        if self.testGradient:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        if self.iter % self.saveRate == 0:
            logging.info('Saving surfaces...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            if not self.internalCost and self.affineDim <= 0:
                xtEPDiff, atEPdiff = self.saveEPDiff(self.fvInit, self.at, fileName=self.saveFile)
                logging.info('EPDiff difference %f' % (np.fabs(self.xt[-1, :, :] - xtEPDiff[-1, :, :]).sum()))

            if self.saveTrajectories:
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curves.vtk', self.xt)

            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)
            dim2 = self.dim ** 2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2 + self.dim]
            (xt, Jt) = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                         withJacobian=True)
            if self.affine == 'euclidean' or self.affine == 'translation':
                self.saveCorrectedEvolution(self.fvInit, xt, self.at, self.Afft, fileName=self.saveFile,
                                            Jacobian=Jt)
            self.saveEvolution(self.fvInit, xt, Jacobian=Jt, fileName=self.saveFile)
        else:
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)

        if self.pplot:
            fig = plt.figure(4)
            # fig.clf()
            ax = Axes3D(fig)
            lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
            lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
            ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
            ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
            ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
            fig.canvas.flush_events()



    def saveHD5(self, fileName):
        fout = h5py.File(fileName, 'w')
        LDDMMResult = fout.create_group('LDDMM Results')
        template = LDDMMResult.create_group('template')
        template.create_dataset('vertices', data=self.fv0.vertices)
        template.create_dataset('faces', data=self.fv0.vertices)
        target = LDDMMResult.create_group('target')
        target.create_dataset('vertices', data=self.fv1.vertices)
        target.create_dataset('faces', data=self.fv1.vertices)
        deformedTemplate = LDDMMResult.create_group('deformedTemplate')
        deformedTemplate.create_dataset('vertices', data=self.fv1.vertices)
        deformedTemplate.create_dataset('faces', data=self.fv1.vertices)
        variables = LDDMMResult.create_group('variables')
        variables.create_dataset('alpha', data=self.at)
        variables.create_dataset('affine', data=self.Afft)
        descriptors = LDDMMResult.create_group('descriptors')

        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2 + self.dim]
        (xt, Jt) = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                     withJacobian=True)

        AV0 = self.fv0.computeVertexArea()
        AV = self.fvDef.computeVertexArea()[0]/AV0[0]
        descriptors.create_dataset('Jacobian', data=Jt[-1,:])
        descriptors.create_dataset('Surface Jacobian', data=AV)
        descriptors.create_dataset('Displacement', data=xt[-1,...]-xt[0,...])

        fout.close()


    def endOfProcedure(self):
        self.endOfIteration()
    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        self.epsMax = 5.
        logging.info(f'Gradient lower bound: {self.gradEps:.5f}')
        self.coeffAff = self.coeffAff1
        if self.param.algorithm == 'cg':
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1,
                  forceLineSearch=self.forceLineSearch)
        elif self.param.algorithm == 'bfgs':
            bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
                      Wolfe=self.param.wolfe, memory=50)
        #return self.at, self.xt

