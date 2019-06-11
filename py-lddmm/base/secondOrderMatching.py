import logging
import glob
import numpy.linalg as la
#import scipy as sp
from . import surfaces
from .pointSets import *
from .surfaceMatching import SurfaceMatchingParam
#import kernelFunctions as kfun
from . import conjugateGradient as cg, pointEvolution as evol, bfgs
from .affineBasis import getExponential, gradExponential, AffineBasis

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'

class Direction:
    def __init__(self):
        self.a0 = []
        self.rhot = []
        self.aff = []
        self.a00 = []
        self.rhot0 = []
        self.aff0 = []


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
class SurfaceMatching:
    def __init__(self, Template=None, Targets=None, fileTempl=None, fileTarg=None, param=None, initialMomentum=None, times = None,
                 maxIter=1000, regWeight = 1.0, verb=True, typeRegression='spline2', affine = 'none', controlWeight = 1.0,
                 affineWeight = 1.0, rotWeight = None, scaleWeight = None, transWeight = None,
                 subsampleTargetSize=-1, rescaleTemplate=False, testGradient=False, saveFile = 'evolution', outputDir = '.'):
        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param
            
        self.param.KparDiff0 = self.param.KparDiff

        if Template==None:
            if fileTempl==None:
                print('Please provide a template surface')
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileTempl)
        else:
            self.fv0 = surfaces.Surface(surf=Template)
            
        if Targets==None:
            if fileTarg==None:
                print('Please provide a list of target surfaces')
                return
            else:
                self.fv1 = []
                if self.param.errorType == 'L2Norm':
                    for f in fileTarg:
                        fv1 = surfaces.Surface()
                        fv1.readFromImage(f)
                        self.fv1.append(fv1)
                else:
                    for f in fileTarg:
                        self.fv1.append(surfaces.Surface(filename=f))
        else:
            self.fv1 = []
            if self.param.errorType == 'L2Norm':
                for s in Targets:
                    fv1 = surfaces.Surface()
                    fv1.readFromImage(s)
                    self.fv1.append(fv1)
            else:
                for s in Targets:
                    self.fv1.append(surfaces.Surface(surf=s))

        if rescaleTemplate:
            f0 = np.fabs(self.fv0.surfVolume())
            f1 = np.fabs(self.fv1[0].surfVolume()) 
            self.fv0.updateVertices(self.fv0.vertices * (f1/f0)**(1./3))
            m0 = np.mean(self.fv0.vertices, axis = 0)
            m1 = np.mean(self.fv1[0].vertices, axis = 0)
            self.fv0.updateVertices(self.fv0.vertices + (m1-m0))

        self.nTarg = len(self.fv1)
        self.saveRate = 10
        self.iter = 0
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print('Cannot save in ' + outputDir)
                return
            else:
                os.mkdir(outputDir)
                
        self.dim = self.fv0.vertices.shape[1]
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.affine = affine
        if self.affine=='euclidean' or self.affine=='translation':
            self.saveCorrected = True
        else:
            self.saveCorrected = False

        self.affB = AffineBasis(self.dim, affine)
        self.affineDim = self.affB.affineDim
        self.affineBasis = self.affB.basis
        self.affineWeight = affineWeight * np.ones(self.affineDim)
        if (len(self.affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[self.affB.rotComp] = rotWeight
        if (len(self.affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[self.affB.simComp] = scaleWeight
        if (len(self.affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[self.affB.transComp] = transWeight
        self.affw = affineWeight

        self.controlWeight = controlWeight
        if self.affine == 'none':
            self.typeRegression = typeRegression
        else:
            self.typeRegression = 'affine' 
            
        self.typeRegressionSave = typeRegression 
        self.affBurnIn = 50
        self.coeffAff1 = 1
        self.coeffAff2 = 10.
        self.coeffAff = self.coeffAff1


        self.fv0Fine = surfaces.Surface(surf=self.fv0)
        if (subsampleTargetSize > 0):
            self.fv0.Simplify(subsampleTargetSize)
            print('simplified template', self.fv0.vertices.shape[0])
            
        v0 = self.fv0.surfVolume()
        #print 'v0', v0
        if self.param.errorType == 'L2Norm' and v0 < 0:
            #print 'flip'
            self.fv0.flipFaces()
            v0 = -v0 ;
        for s in self.fv1:
            v1 = s.surfVolume()
            #print 'v1', v1
            if (v0*v1 < 0):
                #print 'flip1'
                s.flipFaces()

        #self.x0 = self.fv0.vertices
        self.x0 = self.fv0.vertices
        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.npt = self.x0.shape[0]
        #self.Tsize1 = int(round(1.0/self.param.timeStep))
        if times is None:
            times = np.array(range(self.nTarg))
        self.Tsize0 = int(round(1./self.param.timeStep))
        self.Tsize = int(round(times[-1]/self.param.timeStep))        
        self.jumpIndex = np.int_(np.round(times/self.param.timeStep))
        #print self.jumpIndex
        self.isjump = np.zeros(self.Tsize+1, dtype=bool)
        for k in self.jumpIndex:
            self.isjump[k] = True

        self.rhot0 = np.zeros([self.Tsize0, self.x0.shape[0], self.x0.shape[1]])
        self.rhot = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if initialMomentum==None:
            self.xt0 = np.tile(self.x0, [self.Tsize0+1, 1, 1])
            self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
            self.a00 = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.a0 = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.at = np.tile(self.a0, [self.Tsize+1, 1, 1])
            self.at0 = np.tile(self.a00, [self.Tsize0+1, 1, 1])
        else:
            self.a00 = initialMomentum[0]
            self.a0 = initialMomentum[1]
            (self.xt0, self.at0)  = evol.secondOrderEvolution(self.x0, self.a00, self.rhot0, self.param.KparDiff0)
            (self.xt, self.at)  = evol.secondOrderEvolution(self.x0[-1,...], self.a0, self.rhot, self.param.KparDiff)

        #self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.rhotTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.rhot0Try = np.zeros([self.Tsize0, self.x0.shape[0], self.x0.shape[1]])
        self.a0Try = np.zeros([self.x0.shape[0], self.x0.shape[1]])
        self.a00Try = np.zeros([self.x0.shape[0], self.x0.shape[1]])
        self.Afft0 = np.zeros([self.Tsize0, self.affineDim])
        self.Afft0Try = np.zeros([self.Tsize0, self.affineDim])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])

        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        for k,s in enumerate(self.fv1):
            s.saveVTK(self.outputDir+'/Target'+str(k)+'.vtk')
        z= self.fv0.surfVolume()
        if (z < 0):
            self.fv0ori = -1
        else:
            self.fv0ori = 1



    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)
        files = glob.glob(outputDir)
        for f in files:
            os.remove(f)


    def dataTerm(self, _fvDef):
        obj = 0
        for k,s in enumerate(_fvDef):
            if self.param.errorType == 'L2Norm':
                obj += surfaces.L2Norm(s, self.fv1[k].vfld) / (self.param.sigmaError ** 2)
            else:
                obj += self.param.fun_obj(s, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
        return obj


    def  objectiveFunDef(self, a00, rhot0, Afft0, a0, rhot, Afft, withTrajectory = False, withJacobian=False, Init = None, display=False):
        if Init == None:
            x0 = self.x0
        else:
            x0 = Init[0]
            
        param = self.param
        timeStep = self.param.timeStep
        dim2 = self.dim**2
        A0 = [np.zeros([self.Tsize0, self.dim, self.dim]), np.zeros([self.Tsize0, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize0):
                AB = np.dot(self.affineBasis, Afft0[t])
                A0[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A0[1][t] = AB[dim2:dim2+self.dim]
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        #print a0.shape
        if withJacobian:
            (xt0, at0, Jt0) = evol.secondOrderEvolution(x0, a00, rhot0, param.KparDiff0, timeStep, withJacobian=True, affine=A0)
            (xt, at, Jt) = evol.secondOrderEvolution(xt0[-1,...], a0, rhot, param.KparDiff, timeStep, withJacobian=True,affine=A)
        else:
            (xt0, at0) = evol.secondOrderEvolution(x0, a00, rhot0, param.KparDiff0, timeStep, affine=A0)
            (xt, at) = evol.secondOrderEvolution(xt0[-1,...], a0, rhot, param.KparDiff, timeStep, affine=A)
        #print xt[-1, :, :]
        #print obj
        obj00 = 0.5 * (a00 * param.KparDiff0.applyK(x0,a00)).sum() 
        obj0 = 0.5 * (a0 * param.KparDiff.applyK(xt[0,...],a0)).sum()
        obj10 = 0
        obj1 = 0
        obj20 = 0
        obj2 = 0
        for t in range(self.Tsize0):
            rho = np.squeeze(rhot0[t, :, :])            
            obj10 += timeStep* self.controlWeight * (rho**2).sum()/2
            if self.affineDim > 0:
                obj20 +=  timeStep * (self.affineWeight.reshape(Afft0[t].shape) * Afft0[t]**2).sum()/2
        for t in range(self.Tsize):
            rho = np.squeeze(rhot[t, :, :])            
            obj1 += timeStep* self.controlWeight * (rho**2).sum()/2
            if self.affineDim > 0:
                obj2 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()/2
            #print xt.sum(), at.sum(), obj
        obj = obj1+obj2+obj0+obj10+obj20+obj00
        if display:
            logging.info('deformation terms: init %f, rho %f, aff %f'%(obj0,obj1,obj2))
        if withJacobian:
            return obj, xt0, at0, Jt0, xt, at, Jt 
        elif withTrajectory:
            return obj, xt0, at0, xt, at
        else:
            return obj


    def objectiveFun(self):
        if self.obj == None:
            (self.obj, self.xt0, self.at0, self.xt, self.at) = self.objectiveFunDef(self.a00, self.rhot0, self.Afft0, self.a0, self.rhot, self.Afft, withTrajectory=True)
            self.obj0 = 0
            for k in range(self.nTarg):
                if self.param.errorType == 'L2Norm':
                    self.obj0 += surfaces.L2Norm0(self.fv1[k]) / (self.param.sigmaError ** 2)
                else:   
                    self.obj0 += self.param.fun_obj0(self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                foo = surfaces.Surface(surf=self.fvDef[k])
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))
                foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return (self.a00, self.rhot0, self.Afft0, self.a0, self.rhot, self.Afft)
    
    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        #print self.typeRegression
        if self.typeRegression == 'spline':
            a0Try = self.a0
            rhotTry = self.rhot - eps * dir.rhot
        elif self.typeRegression == 'geodesic':
            a0Try = self.a0 - eps * dir.a0
            rhotTry = self.rhot
        elif self.typeRegression == "affine":
            a0Try = self.a0
            rhotTry = self.rhot
        else:
            a0Try = self.a0 - eps * dir.a0
            rhotTry = self.rhot - eps * dir.rhot
        if self.typeRegression == "affine":
            a00Try = self.a00
            rhot0Try = self.rhot0
        else:
            a00Try = self.a00 - eps * dir.a00
            rhot0Try = self.rhot0 - eps * dir.rhot0

        if self.affineDim > 0 and self.typeRegression=="affine":
            Afft0Try = self.Afft0 - eps * dir.aff0
            AfftTry = self.Afft - eps * dir.aff
        else:
            Afft0Try = self.Afft0
            AfftTry = self.Afft
        foo = self.objectiveFunDef(a00Try, rhot0Try, Afft0Try, a0Try, rhotTry, AfftTry, withTrajectory=True)
        objTry += foo[0]

        ff = [] 
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(foo[3][self.jumpIndex[k], :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.a00Try = a00Try
            self.rhot0Try = rhot0Try
            self.Afft0Try = Afft0Try
            self.a0Try = a0Try
            self.rhotTry = rhotTry
            self.AfftTry = AfftTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry



    def endPointGradient(self):
        px = []
        for k in range(self.nTarg):
            if self.param.errorType == 'L2Norm':
                targGradient = -surfaces.L2NormGradient(self.fvDef[k], self.fv1[k].vfld) / (self.param.sigmaError ** 2)
            else:
                targGradient = -self.param.fun_objGrad(self.fvDef[k], self.fv1[k], self.param.KparDist)/(self.param.sigmaError**2)
            px.append(targGradient)
        return px 

    def secondOrderCovector(self, x00, a00, rhot0, a0, rhot, px1, pa1, isjump, affine = (None, None)):
        nTarg = len(px1)
        N = x00.shape[0]
        dim = x00.shape[1]
        if not(affine[1] is None):
            aff_ = True
            A = affine[1][0]
        else:
            aff_ = False
            
        T = self.Tsize
        if isjump is None:
            isjump = np.zeros(T, dtype=bool)
            for t in range(T):
                if t%nTarg == 0:
                    isjump[t] = True
    
        timeStep = self.param.timeStep
        [xt0, at0] = evol.secondOrderEvolution(x00, a00, rhot0, self.param.KparDiff0, timeStep, affine=affine[0])
        x0 = xt0[-1,...]
        [xt, at] = evol.secondOrderEvolution(x0, a0, rhot, self.param.KparDiff, timeStep, affine=affine[1])
        pxt = np.zeros([T+1, N, dim])
        pxt[T, :, :] = px1[nTarg-1]
        pat = np.zeros([T+1, N, dim])
        pat[T, :, :] = pa1[nTarg-1]
        jIndex = nTarg - 2
        KparDiff = self.param.KparDiff
        for t in range(T):
            px = np.squeeze(pxt[T-t, :, :])
            pa = np.squeeze(pat[T-t, :, :])
            x = np.squeeze(xt[T-t-1, :, :])
            a = np.squeeze(at[T-t-1, :, :])
            #rho = np.squeeze(rhot[T-t-1, :, :])
            
            if aff_:
                U = getExponential(timeStep * A[T-t-1])
                px_ = np.dot(px, U)
                Ui = la.inv(U)
                pa_ = np.dot(pa,Ui.T)
            else:
                px_ = px
                pa_ = pa
    
            a1 = np.concatenate((px_[np.newaxis,...], a[np.newaxis,...]))
            a2 = np.concatenate((a[np.newaxis,...], px_[np.newaxis,...]))
            zpx = KparDiff.applyDiffKT(x, a1, a2) - KparDiff.applyDDiffK11and12(x, a, a, pa_)
            zpa = KparDiff.applyK(x, px_) - KparDiff.applyDiffK1and2(x, pa_, a)
    
            pxt[T-t-1, :, :] = px_ + timeStep * zpx
            pat[T-t-1, :, :] = pa_ + timeStep * zpa
            if isjump[T-t-1]:
                pxt[T-t-1, :, :] += px1[jIndex]
                pat[T-t-1, :, :] += pa1[jIndex]
                jIndex -= 1

        #####
        T = self.Tsize0
        if not(affine[0] is None):
            aff_ = True
            A0 = affine[0][0]
        else:
            aff_ = False
        pxt0 = np.zeros([T+1, N, dim])
        pxt0[T, :, :] = pxt[0,:,:] - KparDiff.applyDiffKT(x0, a0[np.newaxis,...], a0[np.newaxis,...])
        pat0 = np.zeros([T+1, N, dim])
        
        KparDiff = self.param.KparDiff0
        for t in range(T):
            px = np.squeeze(pxt0[T-t, :, :])
            pa = np.squeeze(pat0[T-t, :, :])
            x = np.squeeze(xt0[T-t-1, :, :])
            a = np.squeeze(at0[T-t-1, :, :])
            #rho = np.squeeze(rhot[T-t-1, :, :])
            
            if aff_:
                U = getExponential(timeStep * A0[T-t-1])
                px_ = np.dot(px, U)
                Ui = la.inv(U)
                pa_ = np.dot(pa,Ui.T)
            else:
                px_ = px
                pa_ = pa
    
            a1 = np.concatenate((px_[np.newaxis,...], a[np.newaxis,...]))
            a2 = np.concatenate((a[np.newaxis,...], px_[np.newaxis,...]))
            zpx = KparDiff.applyDiffKT(x, a1, a2) - KparDiff.applyDDiffK11and12(x, a, a, pa_)
            zpa = KparDiff.applyK(x, px_) - KparDiff.applyDiffK1and2(x, pa_, a)
    
            pxt0[T-t-1, :, :] = px_ + timeStep * zpx
            pat0[T-t-1, :, :] = pa_ + timeStep * zpa

        return [[pxt0, pat0, xt0, at0], [pxt, pat, xt, at]]

    # Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
    def secondOrderGradient(self, x00, a00, rhot0, a0, rhot, px1, pa1, isjump, getCovector = False, affine=(None, None), controlWeight=1.0):
        foo = self.secondOrderCovector(x00, a00, rhot0, a0, rhot, px1, pa1, isjump, affine=affine)
        (pxt0, pat0, xt0, at0) = foo[0]
        if not (affine[0] is None):
            dA0 = np.zeros(affine[0][0].shape)
            db0 = np.zeros(affine[0][1].shape)
        Tsize0 = self.Tsize0
        timeStep = self.param.timeStep
        drhot0 = np.zeros(rhot0.shape)
        KparDiff = self.param.KparDiff0
        if not (affine[0] is None):
            for t in range(Tsize0):
                x = np.squeeze(xt0[t, :, :])
                a = np.squeeze(at0[t, :, :])
                rho = np.squeeze(rhot0[t, :, :])
                px = np.squeeze(pxt0[t+1, :, :])
                pa = np.squeeze(pat0[t+1, :, :])
                zx = x + timeStep * KparDiff.applyK(x, a)
                za = a + timeStep * (-KparDiff.applyDiffKT(x, a[np.newaxis,...], a[np.newaxis,...]) + rho)
                U = getExponential(timeStep * affine[0][0][t])
                #U = np.eye(dim) + timeStep * affine[0][k]
                Ui = la.inv(U)
                pa = np.dot(pa, Ui.T)
                za = np.dot(za, Ui)
                dA0[t,...] =  (gradExponential(timeStep*affine[0][0][t], px, zx) 
                                - gradExponential(timeStep*affine[0][0][t], za, pa))
                drhot0[t,...] = rho*controlWeight - pa
            db0 = pxt0[1:Tsize0+1,...].sum(axis=1)
    
        da00 = KparDiff.applyK(x00, a00) - pat0[0,...]
        
        
        (pxt, pat, xt, at) = foo[1]
        if not (affine is None):
            dA = np.zeros(affine[1][0].shape)
            db = np.zeros(affine[1][1].shape)
        Tsize = self.Tsize
        drhot = np.zeros(rhot.shape)
        KparDiff = self.param.KparDiff
        if not (affine is None):
            for t in range(Tsize):
                x = np.squeeze(xt[t, :, :])
                a = np.squeeze(at[t, :, :])
                rho = np.squeeze(rhot[t, :, :])
                px = np.squeeze(pxt[t+1, :, :])
                pa = np.squeeze(pat[t+1, :, :])
                zx = x + timeStep * KparDiff.applyK(x, a)
                za = a + timeStep * (-KparDiff.applyDiffKT(x, a[np.newaxis,...], a[np.newaxis,...]) + rho)
                U = getExponential(timeStep * affine[1][0][t])
                Ui = la.inv(U)
                pa = np.dot(pa, Ui.T)
                za = np.dot(za, Ui)
                dA[t,...] =  (gradExponential(timeStep*affine[1][0][t], px, zx) 
                                - gradExponential(timeStep*affine[1][0][t], za, pa))
                drhot[t,...] = rho*controlWeight - pa
            db = pxt[1:Tsize+1,...].sum(axis=1)

        da0 = KparDiff.applyK(xt[0,...], a0) - pat[0,...]
    
        if affine is None:
            if getCovector == False:
                return [[da00, drhot0, xt0, at0], [da0, drhot, xt, at]]
            else:
                return [[da00, drhot0, xt0, at0, pxt0, pat0], [da0, drhot, xt, at, pxt, pat]]
        else:
            if getCovector == False:
                return [[da00, drhot0, dA0, db0, xt0, at0], [da0, drhot, dA, db, xt, at]]
            else:
                return [[da00, drhot0, dA0, db0, xt0, at0, pxt0, pat0], [da0, drhot, dA, db, xt, at, pxt, pat]]

    def getGradient(self, coeff=1.0):
        px1 = self.endPointGradient()
        pa1 = []
        for k in range(self.nTarg):
            pa1.append(np.zeros(self.a0.shape))
            
        A0 = [np.zeros([self.Tsize0, self.dim, self.dim]), np.zeros([self.Tsize0, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize0):
                AB = np.dot(self.affineBasis, self.Afft0[t])
                A0[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A0[1][t] = AB[dim2:dim2+self.dim]

        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]

        foo = self.secondOrderGradient(self.x0, self.a00, self.rhot0, self.a0, self.rhot, px1, pa1, self.isjump,
                                       affine=(A0, A), controlWeight=self.controlWeight)
        grd = Direction()
        if self.typeRegression == 'affine':
            grd.a00 = np.zeros(foo[0][0].shape)
            grd.rhot0 = np.zeros(foo[0][1].shape)
        else:
            grd.a00 = foo[0][0] / coeff
            grd.rhot0 = foo[0][1] / coeff
        
        if self.typeRegression == 'spline':
            grd.a0 = np.zeros(foo[1][0].shape)
            grd.rhot = foo[1][1]/(coeff)
            #grd.rhot = foo[1]/(coeff*self.rhot.shape[0])
        elif self.typeRegression == 'geodesic':
            grd.a0 = foo[1][0] / coeff
            grd.rhot = np.zeros(foo[1][1].shape)
        elif self.typeRegression == 'affine':
            grd.a0 = np.zeros(foo[1][0].shape)
            grd.rhot = np.zeros(foo[1][1].shape)
        else:
            grd.a0 = foo[1][0] / coeff
            grd.rhot = foo[1][1]/(coeff)

        grd.aff = np.zeros(self.Afft.shape)
        grd.aff0 = np.zeros(self.Afft0.shape)
        if self.affineDim > 0 and self.iter < self.affBurnIn:
            dA0 = foo[0][2]
            db0 = foo[0][3]
            dA = foo[1][2]
            db = foo[1][3]
            grd.aff0 = np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft0)
            for t in range(self.Tsize0):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA0[t].reshape([dim2,1]), db0[t].reshape([self.dim, 1])]))
               grd.aff0[t] -=  dAff.reshape(grd.aff[t].shape)
            grd.aff0 *= self.param.timeStep/(self.coeffAff*coeff)
            grd.aff = np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft)
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd.aff[t] -=  dAff.reshape(grd.aff[t].shape)
            grd.aff *= self.param.timeStep/(self.coeffAff*coeff)
        
        #print (grd.a00**2).sum(),(grd.rhot0**2).sum(),(grd.aff0**2).sum(),(grd.a0**2).sum(),(grd.rhot**2).sum(),(grd.aff**2).sum() 
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.a0 = dir1.a0 + beta * dir2.a0
        dir.rhot = dir1.rhot + beta * dir2.rhot
        dir.aff = dir1.aff + beta * dir2.aff
        dir.a00 = dir1.a00 + beta * dir2.a00
        dir.rhot0 = dir1.rhot0 + beta * dir2.rhot0
        dir.aff0 = dir1.aff0 + beta * dir2.aff0
        return dir

    def prod(self, dir1, beta):
        dir = Direction()
        dir.a0 = beta * dir1.a0
        dir.rhot = beta * dir1.rhot
        dir.aff = beta * dir1.aff
        dir.a00 = beta * dir1.a00
        dir.rhot0 = beta * dir1.rhot0
        dir.aff0 = beta * dir1.aff0
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.a0 = np.copy(dir0.a0)
        dir.rhot = np.copy(dir0.rhot)
        dir.aff = np.copy(dir0.aff)
        dir.a00 = np.copy(dir0.a00)
        dir.rhot0 = np.copy(dir0.rhot0)
        dir.aff0 = np.copy(dir0.aff0)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        if self.typeRegression == 'spline':
            dirfoo.a0 = np.zeros([self.npt, self.dim])
            dirfoo.rhot = np.random.randn(self.Tsize, self.npt, self.dim)
        elif self.typeRegression == 'geodesic':
            dirfoo.a0 = np.random.randn(self.npt, self.dim)
            dirfoo.rhot = np.zeros([self.Tsize, self.npt, self.dim])
        elif self.typeRegression == 'affine':
            dirfoo.a0 = np.zeros([self.npt, self.dim])
            dirfoo.rhot = np.zeros([self.Tsize, self.npt, self.dim])
        else:
            dirfoo.a0 = np.random.randn(self.npt, self.dim)
            dirfoo.rhot = np.random.randn(self.Tsize, self.npt, self.dim)

        if self.typeRegression == 'affine':
            dirfoo.a00 = np.zeros([self.npt, self.dim])
            dirfoo.rhot0 = np.zeros([self.Tsize0, self.npt, self.dim])
        else:
            dirfoo.a00 = np.random.randn(self.npt, self.dim)
            dirfoo.rhot0 = np.random.randn(self.Tsize0, self.npt, self.dim)
        
        if self.iter < self.affBurnIn:
            dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
            dirfoo.aff0 = np.random.randn(self.Tsize0, self.affineDim)
        else:
            dirfoo.aff = np.zeros((self.Tsize, self.affineDim))
            dirfoo.aff0 = np.zeros((self.Tsize0, self.affineDim))
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        gg = g1.rhot
        gga = g1.a0
        uu = g1.aff
        ll = 0
        for gr in g2:
            ggOld = gr.rhot
            res[ll]  = (ggOld*gg).sum()*self.param.timeStep
            res[ll] += (gr.a0 * gga).sum()
            res[ll] += (uu * gr.aff).sum() * self.coeffAff
            ll = ll+1

        gg = g1.rhot0
        gga = g1.a00
        uu = g1.aff0
        ll = 0
        for gr in g2:
            ggOld = gr.rhot0
            res[ll] += (ggOld*gg).sum()*self.param.timeStep
            res[ll] += (gr.a00 * gga).sum()
            res[ll] += (uu * gr.aff0).sum() * self.coeffAff
            ll = ll+1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.a0 = np.copy(self.a0Try)
        self.rhot = np.copy(self.rhotTry)
        self.Afft = np.copy(self.AfftTry)
        self.a00 = np.copy(self.a00Try)
        self.rhot0 = np.copy(self.rhot0Try)
        self.Afft0 = np.copy(self.Afft0Try)
        #print self.at

    def endOfIteration(self):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.typeRegression = self.typeRegressionSave
            self.affine = 'none'
            #self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0):
            logging.info('Saving surfaces...')
            (obj1, self.xt0, self.at0, self.xt, self.at) = self.objectiveFunDef(self.a00, self.rhot0, self.Afft0, self.a0, self.rhot, self.Afft, withTrajectory=True, display=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))
            dim2 = self.dim**2
            A0 = [np.zeros([self.Tsize0, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize0):
                    AB = np.dot(self.affineBasis, self.Afft0[t])
                    A0[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A0[1][t] = AB[dim2:dim2+self.dim]
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
                    
            (xt0, at0, ft0)  = evol.secondOrderEvolution(self.x0, self.a00,  self.rhot0, self.param.KparDiff0, self.param.timeStep, affine=A0,
                                                           withPointSet = self.fv0Fine.vertices)
            (xt, at, ft, Jt)  = evol.secondOrderEvolution(xt0[-1,...], self.a0,  self.rhot, self.param.KparDiff, self.param.timeStep, affine=A,
                                                           withPointSet = ft0[-1,...], withJacobian=True)

            if self.saveCorrected:
                f = surfaces.Surface(surf=self.fv0Fine)
                X0 = self.affB.integrateFlow(self.Afft0)
                X = self.affB.integrateFlow(self.Afft)
                displ = np.zeros(self.x0.shape[0])
                at0Corr = np.zeros(at0.shape) 
                for t in range(self.Tsize0+1):
                    R0 = X0[0][t,...]
                    U0 = la.inv(R0)
                    b0 = X0[1][t,...]
                    yyt = np.dot(self.xt0[t,...] - b0, U0.T)
                    zt = np.dot(xt0[t,...] - b0, U0.T)
                    if t < self.Tsize0:
                        a = np.dot(self.at0[t,...], R0)
                        at0Corr[t,...] = a
                        vt = self.param.KparDiff0.applyK(yyt, a, firstVar=zt)
                        vt = np.dot(vt, U0.T)
                    f.updateVertices(zt)
                    vf = surfaces.vtkFields()
                    vf.scalars.append('Jacobian')
                    vf.scalars.append(displ)
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity')
                    vf.vectors.append(vt)
                    f.saveVTK2(self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.vtk', vf)

                dt = 1.0 /self.Tsize0
                atCorr = np.zeros(at.shape) 
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t,...])
                    yyt = np.dot(self.xt[t,...] - b0, U0.T)
                    yyt = np.dot(yyt - X[1][t, ...], U.T)
                    zt = np.dot(xt[t,...] - b0, U0.T)
                    zt = np.dot(zt - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        a = np.dot(self.at[t,...], R0)
                        a = np.dot(a, X[0][t,...])
                        atCorr[t,...] = a
                        vt = self.param.KparDiff.applyK(yyt, a, firstVar=zt)
                        vt = np.dot(vt, U.T)
                    f.updateVertices(zt)
                    vf = surfaces.vtkFields()
                    vf.scalars.append('Jacobian')
                    vf.scalars.append(np.exp(Jt[t, :])-1)
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity')
                    vf.vectors.append(vt)
                    nu = self.fv0ori*f.computeVertexNormals()
                    displ += dt * (vt*nu).sum(axis=1)
                    f.saveVTK2(self.outputDir +'/'+self.saveFile+'Corrected'+str(t+self.Tsize0)+'.vtk', vf)
#                (foo,zt) = evol.landmarkDirectEvolutionEuler(self.x0, atCorr, self.param.KparDiff, withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.outputDir +'/'+self.saveFile+'CorrectedCheck'+str(t)+'.vtk')
#                (foo,foo2,zt) = evol.secondOrderEvolution(self.x0, atCorr[0,...], self.rhot, self.param.KparDiff, withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.outputDir +'/'+self.saveFile+'CorrectedCheckBis'+str(t)+'.vtk')
                 

                for k,fv in enumerate(self.fv1):
                    f = surfaces.Surface(surf=fv)
                    U = la.inv(X[0][self.jumpIndex[k]])
                    yyt = np.dot(f.vertices - b0, U0.T)
                    yyt = np.dot(yyt - X[1][self.jumpIndex[k], ...], U.T)
                    f.updateVertices(yyt)
                    f.saveVTK(self.outputDir +'/Target'+str(k)+'Corrected.vtk')
            
            fvDef = surfaces.Surface(surf=self.fv0Fine)
            AV0 = fvDef.computeVertexArea()
            nu = self.fv0ori*self.fv0Fine.computeVertexNormals()
            #v = self.v[0,...]
            displ = np.zeros(self.npt)
            dt = 1.0 /self.Tsize
            v = self.param.KparDiff0.applyK(ft0[0,...], self.at0[0,...], firstVar=self.xt0[0,...])
            for kk in range(self.Tsize0+1):
                fvDef.updateVertices(np.squeeze(ft0[kk, :, :]))
                AV = fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0])-1
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(displ)
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(displ)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(displ)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if kk < self.Tsize:
                    v = self.param.KparDiff0.applyK(ft0[kk,...], self.at0[kk,...], firstVar=self.xt0[kk,...])
                    kkm = kk
                else:
                    kkm = kk-1
                vf.vectors.append('velocity')
                vf.vectors.append(np.copy(v))
                fvDef.saveVTK2(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', vf)

            dt = 1.0 /self.Tsize
            v = self.param.KparDiff.applyK(ft[0,...], self.at[0,...], firstVar=self.xt[0,...])
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(ft[kk, :, :]))
                AV = fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0])-1
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jt[kk, :])-1)
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(np.exp(Jt[kk, :])/(AV+1)-1)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                displ += dt * (v*nu).sum(axis=1)
                if kk < self.Tsize:
                    nu = self.fv0ori*fvDef.computeVertexNormals()
                    v = self.param.KparDiff.applyK(ft[kk,...], self.at[kk,...], firstVar=self.xt[kk,...])
                    #v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                vf.vectors.append('velocity')
                vf.vectors.append(np.copy(v))
                fvDef.saveVTK2(self.outputDir +'/'+ self.saveFile+str(kk+self.Tsize0)+'.vtk', vf)
        else:
            (obj1, self.xt0, self.at0, self.xt, self.at) = self.objectiveFunDef(self.a00, self.rhot0, self.Afft0, self.a0, self.rhot, self.Afft, withTrajectory=True, display=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))


    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        logging.info('Gradient lower bound: {0:f}'.format(self.gradEps))
        #print 'x0:', self.x0
        #print 'y0:', self.y0
        self.cgBurnIn = self.affBurnIn
        
        if self.param.algorithm == 'cg':
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.01)
        elif self.param.algorithm == 'bfgs':
            bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
                      Wolfe=self.param.wolfe, memory=25)
        #return self.at, self.xt
