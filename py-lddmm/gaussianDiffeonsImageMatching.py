import os
import numpy as np
import numpy.linalg as LA
import scipy as sp
import scipy.ndimage as Img
import diffeo
import kernelFunctions as kfun
import gaussianDiffeons as gd
import pointEvolution as evol
import conjugateGradient as cg
from affineBasis import *

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normalization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'


def ImageMatchingDist(gr, J, im0, im1):
    imdef = Img.interpolation.map_coordinates(im1.data, gr.transpose(range(-1, gr.ndim-1)), order=1)
    res = (((im0.data - imdef)**2)*np.exp(J)).sum()
    return res

def ImageMatchingGradient(gr, J, im0, im1):
    gradIm1 = diffeo.gradient(im1.data, im1.resol)
    imdef = Img.interpolation.map_coordinates(im1.data, gr.transpose(range(-1, gr.ndim-1)), order=1)
    gradDef = np.zeros(gradIm1.shape)
    for k in range(gradIm1.shape[0]):
        gradDef[k,...] = Img.interpolation.map_coordinates(gradIm1[k, ...], gr.transpose(range(-1, gr.ndim-1)), order=1)

    expJ = np.exp(J)
    pgr = ((-2*(im0.data-imdef)*expJ)*gradDef).transpose(np.append(range(1, gr.ndim), 0))
    pJ =  ((im0.data - imdef)**2)*expJ
    return pgr, pJ



class ImageMatchingParam:
    def __init__(self, timeStep = .1, sigmaKernel = 6.5, sigmaError=1.0, dimension=2, errorType='L2', KparDiff = None, typeKernel='gauss'):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.sigmaError = sigmaError
        self.typeKernel = typeKernel
        self.errorType = errorType
        self.dimension = dimension
        if errorType == 'L2':
            self.fun_obj = ImageMatchingDist
            self.fun_objGrad = ImageMatchingGradient
        else:
            print 'Unknown error Type: ', self.errorType
        if KparDiff == None:
            self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        else:
            self.KparDiff = KparDiff



class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []


## Main class for image matching
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
class ImageMatching:
    def __init__(self, Template=None, Target=None, Diffeons=None, EpsilonNet=None, DecimationTarget=1,
                 subsampleTemplate = 1, 
                 DiffeonEpsForNet=None, DiffeonSegmentationRatio=None, zeroVar=False, fileTempl=None,
                 fileTarg=None, param=None, maxIter=1000, regWeight = 1.0, affineWeight = 1.0, verb=True,
                 rotWeight = None, scaleWeight = None, transWeight = None, testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        if Template==None:
            if fileTempl==None:
                print 'Please provide a template image'
                return
            else:
                self.im0 = diffeo.gridScalars(filename=fileTempl)
        else:
            self.im0 = diffeo.gridScalars(grid=Template)
        print self.im0.data.shape, Template.data.shape
        if Target==None:
            if fileTarg==None:
                print 'Please provide a target image'
                return
            else:
                self.im1 = diffeo.gridScalars(filename=fileTarg)
        else:
            self.im1 = diffeo.gridScalars(grid=Target)

        self.im0Fine = diffeo.gridScalars(grid=self.im0)
        self.saveRate = 10
        self.iter = 0
        self.gradEps = -1
        self.dim = self.im0.data.ndim
        self.setOutputDir(outputDir)
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.affine = affine
        affB = AffineBasis(self.dim, affine)
        self.affineDim = affB.affineDim
        self.affineBasis = affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[affB.rotComp] = rotWeight
        if (len(affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[affB.simComp] = scaleWeight
        if (len(affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[affB.transComp] = transWeight

        if param==None:
            self.param = ImageMatchingParam()
        else:
            self.param = param
        self.dim = self.param.dimension
        #self.x0 = self.fv0.vertices
        if Diffeons==None:
            if DecimationTarget==None:
                DecimationTarget = 1
            if self.dim == 1:
                self.c0 = range(0, self.im0.data.shape[0], DecimationTarget)
            elif self.dim == 2:
                u = np.mgrid[0:self.im0.data.shape[0]:DecimationTarget, 0:self.im0.data.shape[1]:DecimationTarget]
                self.c0 = np.zeros([u[0].size, self.dim])
                self.c0[:,0] = u[0].flatten()
                self.c0[:,1] = u[1].flatten()
            elif self.dim == 3:
                u = np.mgrid[0:self.im0.data.shape[0]:DecimationTarget, 0:self.im0.data.shape[1]:DecimationTarget, 0:self.im0.data.shape[2]:DecimationTarget]
                self.c0 = np.zeros([u[0].size, self.dim])
                self.c0[:,0] = u[0].flatten()
                self.c0[:,1] = u[1].flatten()
                self.c0[:,2] = u[2].flatten()
            print self.im0.resol
            self.c0 = self.im0.origin + self.c0 * self.im0.resol
            self.S0 = np.tile(DecimationTarget*np.diag(self.im0.resol), [self.c0.shape[0], 1, 1])
        else:
            (self.c0, self.S0, self.idx) = Diffeons

        if zeroVar:
	    self.S0 = np.zeros(self.S0.shape)

            #print self.S0
        if subsampleTemplate == None:
            subsampleTemplate = 1
        self.im0.resol *= subsampleTemplate
        self.im0.data = Img.filters.median_filter(self.im0.data, size=subsampleTemplate)
        if self.dim == 1:
            self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate]
            self.gr0 = range(self.im0.data.shape[0])
        elif self.dim == 2:
            self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate, 0:self.im0.data.shape[1]:subsampleTemplate]
            self.gr0 = np.mgrid[0:self.im0.data.shape[0], 0:self.im0.data.shape[1]].transpose((1, 2,0))
        elif self.dim == 3:
            self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate, 0:self.im0.data.shape[1]:subsampleTemplate, 0:self.im0.data.shape[2]:subsampleTemplate]
            self.gr0 = np.mgrid[0:self.im0.data.shape[0], 0:self.im0.data.shape[1], 0:self.im0.data.shape[2]].transpose((1,2, 3, 0))
        self.gr0 = self.im0.origin + self.gr0 * self.im0.resol 
        self.J0 = np.log(self.im0.resol.prod()) * np.ones(self.im0.data.shape) 
	self.ndf = self.c0.shape[0]
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.c0.shape[0], self.dim])
        self.atTry = np.zeros([self.Tsize, self.c0.shape[0], self.dim])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.imt = np.tile(self.im0, np.insert(np.ones(self.dim), 0, self.Tsize+1))
        self.Jt = np.tile(self.J0, np.insert(np.ones(self.dim), 0, self.Tsize+1))
        self.grt = np.tile(self.gr0, np.insert(np.ones(self.dim+1), 0, self.Tsize+1))
        self.ct = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.St = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        print 'error type:', self.param.errorType
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        self.saveFile = saveFile
        self.im0.save(self.outputDir+'/Template.png')
        self.im1.save(self.outputDir+'/Target.png')

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.makedirs(outputDir)


    def  objectiveFunDef(self, at, Afft, withTrajectory = False, initial = None):
        if initial == None:
            c0 = self.c0
            S0 = self.S0
            gr0 = self.gr0
            J0 = self.J0
        else:
            gr0 = self.gr0
            J0 = self.J0
            (c0, S0) = initial
        param = self.param
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
	(ct, St, grt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(c0, S0, at, param.sigmaKernel, affine=A, withPointSet= gr0, withJacobian=J0)

        #print xt[-1, :, :]
        #print obj
        obj=0
        #print St.shape
        for t in range(self.Tsize):
            c = np.squeeze(ct[t, :, :])
            S = np.squeeze(St[t, :, :, :])
            a = np.squeeze(at[t, :, :])
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            rcc = gd.computeProducts(c, S, param.sigmaKernel)
            obj = obj + self.regweight*timeStep*np.multiply(a, np.dot(rcc,a)).sum()
            if self.affineDim > 0:
                obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withTrajectory:
            return obj, ct, St, grt, Jt
        else:
            return obj

    def dataTerm(self, _data):
        obj = self.param.fun_obj(_data[0], _data[1], self.im0, self.im1) / (self.param.sigmaError**2)
        return obj

    def objectiveFun(self):
        if self.obj == None:
            (self.obj, self.ct, self.St, self.xt, self.Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            data = (self.xt[-1,:,:], self.Jt[-1,:])
            self.obj += self.dataTerm(data)

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        atTry = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft
        objTry, ct, St, grt, Jt = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        data = (grt[-1,:,:], Jt[-1,:])
        objTry += self.dataTerm(data)

        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry

            #print 'objTry=',objTry, dir.diff.sum()
        return objTry



    def endPointGradient(self):
        (pg, pJ) = self.param.fun_objGrad(self.grt[-1, :, :], self.Jt[-1, :], self.im0, self.im1)
        pc = np.zeros(self.c0.shape)
        pS = np.zeros(self.S0.shape)
        #gd.testDiffeonCurrentNormGradient(self.ct[-1, :, :], self.St[-1, :, :, :], self.bt[-1, :, :],
        #                               self.fv1, self.param.KparDist.sigma)
        pg = pg / self.param.sigmaError**2
        pJ = pJ / self.param.sigmaError**2
        return (pc, pS, pg, pJ)


    def getGradient(self, coeff=1.0):
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]

        (pc1, pS1, pg1, pJ1) = self.endPointGradient()
        foo = evol.gaussianDiffeonsGradientPset(self.c0, self.S0, self.gr0, self.at, -pc1, -pS1, -pg1, self.param.sigmaKernel, self.regweight,
                                                affine=A, withJacobian = (self.J0, -pJ1))

        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
            grd.aff /= (coeff*self.Tsize)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.aff = dir1.aff + beta * dir2.aff
        return dir

    def copyDir(self, dir0):
        ddir = Direction()
        ddir.diff = np.copy(dir0.diff)
        ddir.aff = np.copy(dir0.aff)
        return ddir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.ndf, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def getBGFSDir(Var, oldVar, grd, grdOld):
        s = (Var[0] - oldVar[0]).unravel()
        y = (grd.diff - grdOld.diff).unravel()
        if skipBGFS==0:
            rho = max(0, (s*y).sum())
        else:
            rho = 0 
        Imsy = np.eye((s.shape[0], s.shape[0])) - rho*np.dot(s, y.T)
        H0 = np.dot(Imsy, np.dot(H0, Imsy)) + rho * np.dot(s, s.T)
        dir0.diff = (np.dot(H0, grd.diff.unravel()))

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
        for t in range(self.Tsize):
            c = np.squeeze(self.ct[t, :, :])
            S = np.squeeze(self.St[t, :, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            rcc = gd.computeProducts(c, S, self.param.sigmaKernel)
            (L, W) = LA.eigh(rcc)
            rcc += (L.max()/1000)*np.eye(rcc.shape[0])
            u = np.dot(rcc, gg)
            uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    res[ll] += np.multiply(uu, gr.aff[t]).sum()
                ll = ll + 1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        #print self.at

    def endOfIteration(self):
        #print self.obj0
        self.iter += 1
        if (self.iter % self.saveRate) == 0:
            print 'saving...'
            (obj1, self.ct, self.St, self.xt, self.Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
	    (ct, St, xt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.at, self.param.sigmaKernel, affine=A,
                                                                    withPointSet = self.fv0Fine.vertices, withJacobian=True)
            imDef = self.im1.copy()
            for kk in range(self.Tsize+1):
                imDef = Img.interpolation.map_coordinates(self.im1, self.gr0)
                imDef.save(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
                gd.saveDiffeons(self.outputDir +'/'+ self.saveFile+'Diffeons'+str(kk)+'.vtk', self.ct[kk,:,:], self.St[kk,:,:,:])
        else:
            (obj1, self.ct, self.St, self.xt, self.Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)



    def optimizeMatching(self):
        # obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) # / (self.param.sigmaError**2)
        # if self.dcurr:
        #     (obj, self.ct, self.St, self.bt, self.xt, self.xSt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
        #     data = (self.xt[-1,:,:], self.xSt[-1,:,:,:], self.bt[-1,:,:])
        #     print 'objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(data)* (self.param.sigmaError**2)
        #     print obj0 + surfaces.currentNormDef(self.fv0, self.fv1, self.param.KparDist)
        # else:
        #     (obj, self.ct, self.St, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
        #     self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
        #     print 'objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(self.fvDef)

        if self.gradEps < 0:
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])

            self.gradEps = max(0.001, np.sqrt(grd2) / 10000)

        print 'Gradient lower bound: ', self.gradEps
        cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt

