import numpy as np
from base import surfaces
import logging
import surfaceMatching as smatch
import multiprocessing as mp
from multiprocessing import Lock
from base import conjugateGradient as cg, kernelFunctions as kfun
from base.affineBasis import AffineBasis
from base import loggingUtils
from base.surfaces import Surface
import glob




class SurfaceTemplateParam(smatch.SurfaceMatchingParam):
    def __init__(self, timeStep = .1, KparDiff = None, KparDiff0 = None, KparDist = None,
                 sigmaError = 1.0, errorType = 'measure',  typeKernel='gauss', internalCost = None):
        smatch.SurfaceMatchingParam.__init__(self, timeStep = timeStep, KparDiff = KparDiff, KparDist = KparDist,
                     sigmaError = sigmaError, errorType = errorType,  typeKernel=typeKernel, internalCost = internalCost)
        if KparDiff0 == None:
            self.KparDiff0 = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        else:
            self.KparDiff0 = KparDiff0


class Direction:
    def __init__(self):
        self.prior = []
        self.all = []

## Main class for surface template estimation
#        HyperTmpl: surface class (from surface.py); if not specified, opens fileHTempl
#        Targets: list of surface class (from surface.py); if not specified, open them from fileTarg
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
class SurfaceTemplate(smatch.SurfaceMatching):

    def __init__(self, HyperTmpl=None, Targets=None, fileHTempl=None, fileTarg=None, param=None, maxIter=1000,
                 internalWeight=1.0, lambdaPrior = 1.0, regWeight = 1.0, affineWeight = 1.0, verb=True, sgd = None,
                 rotWeight = None, scaleWeight = None, transWeight = None, testGradient=False, saveFile = 'evolution',
                 affine = 'none', outputDir = '.'):
        if HyperTmpl is None:
            if fileHTempl is None:
                print 'Please provide A hyper-template surface'
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileHTempl)
        else:
            self.fv0 = surfaces.Surface(surf=HyperTmpl)
        if Targets is None:
            if fileTarg is None:
                print 'Please provide Target surfaces'
                return
            else:
                for ff in fileTarg:
                    self.fv1.append(surfaces.Surface(filename=ff))
        else:
            self.fv1 = []
            for ff in Targets:
                self.fv1.append(surfaces.Surface(surf=ff))

        self.Ntarg = len(self.fv1)
        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        self.saveRate = 1
        self.iter = 0
        self.setOutputDir(outputDir)
        self.saveFile = saveFile
#        self.outputDir = outputDir
#        if not os.access(outputDir, os.W_OK):
#            if os.access(outputDir, os.F_OK):
#                print 'Cannot save in ' + outputDir
#                return
#            else:
#                os.mkdir(outputDir)

        self.fv0.saveVTK(self.outputDir +'/'+ 'HyperTemplate.vtk')
        for kk in range(self.Ntarg):
            self.fv1[kk].saveVTK(self.outputDir +'/'+ 'Target'+str(kk)+'.vtk')

        #self.compGrd = np.zeros(self.Ntarg, dtype=bool)
        self.fvTmpl = surfaces.Surface(surf=self.fv0)
        self.fvTmplTry = surfaces.Surface(surf=self.fv0)
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
        self.coeffAff1 = 1.
        self.coeffAff2 = 100.
        self.coeffAff = self.coeffAff1
        self.affBurnIn = 25

        self.lambdaPrior = lambdaPrior
        if param==None:
            self.param = SurfaceTemplateParam()
        else:
            self.param = param

        self.Tsize = int(round(1.0/self.param.timeStep))
        
        self.updateTemplate = True
        self.updateAllTraj = True
        self.templateBurnIn = 10
        if self.affineDim > 0:
            self.updateAffine = True
        else:
            self.updateAffine = False

        # Htempl to Templ
        self.x0 = self.fv0.vertices
        self.at = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])

        #Template to all
        self.atAll = []
        self.atAllTry = []
        self.xtAll = []
        self.fvDef = []
        self.fvDefTry = []
        self.AfftAll = []
        self.AfftAllTry = []
        for ff in self.fv1:
            self.atAll.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            #smatch.SurfaceMatching(Template=self.fvTmpl, Target=ff,par=self.param))
            self.atAllTry.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            self.AfftAll.append(np.zeros([self.Tsize, self.affineDim]))
            self.AfftAllTry.append(np.zeros([self.Tsize, self.affineDim]))
            self.xtAll.append(np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1]))
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
            self.fvDefTry.append(surfaces.Surface(surf=self.fv0))

        self.tmplCoeff = self.lambdaPrior/float(self.Ntarg)
        self.obj = None
        self.objTry = None
        self.gradCoeff = 1.*self.Ntarg#self.fv0.vertices.shape[0]

        if sgd is None:
            self.sgd = self.Ntarg
        else:
            self.sgd = sgd

        if self.param.internalCost == 'h1':
            self.internalCost = surfaces.normGrad
            self.internalCostGrad = surfaces.diffNormGrad
        else:
            self.internalCost = None
        self.internalWeight = internalWeight
        self.select = np.ones(self.Ntarg, dtype=bool)

    def init(self, ff):
        self.fv0 = ff.fvTmpl
        self.fv1 = ff.fv1

        self.fvTmpl = surfaces.Surface(surf=self.fv0)
        self.maxIter = ff.maxIter
        self.verb = ff.verb
        self.testGradient = ff.testGradient
        self.regweight = 1.
        self.lambdaPrior = ff.lambdaPrior
        self.param = ff.param

        self.Tsize = int(round(1.0/self.param.timeStep))

        self.at = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
        self.atAll = []
        self.atAllTry = []
        self.AfftAll = []
        self.AfftAllTry = []
        self.xtAll = []
        for f0 in self.fv1:
            self.atAll.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            #smatch.SurfaceMatching(Template=self.fvTmpl, Target=ff,par=self.param))
            self.atAllTry.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            self.AfftAll.append(np.zeros([self.Tsize, self.affineDim]))
            self.AfftAllTry.append(np.zeros([self.Tsize, self.affineDim]))
            self.xtAll.append(np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1]))
            self.fvDef.append(surfaces.Surface(surf=self.fv0))

        self.Ntarg = len(self.fv1)
        self.tmplCoeff = 1.0 #float(self.Ntarg)
        self.obj = None #ff.obj
        self.obj0 = ff.obj0
        self.objTry = ff.objTry


        for kk in range(self.Ntarg):
            self.atAll[kk] = np.copy(ff.atAll[kk])
            self.AfftAll[kk] = np.copy(ff.AfftAll[kk])
            self.xtAll[kk] = np.copy(ff.xt[kk])


    def dataTerm(self, _fvDef):
        c = float(self.Ntarg)/self.select.sum()
        obj = 0
        if self.param.errorType == 'L2Norm':
            for k,f in enumerate(_fvDef):
                if self.select[k]:
                    obj += c*surfaces.L2Norm(f, self.fv1[k].vfld) / (self.param.sigmaError ** 2)
        else:
            for k,f in enumerate(_fvDef):
                if self.select[k]:
                    obj += c*self.param.fun_obj(f, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
        #print 'dataterm = ', obj + self.obj0
        return obj
        
    def objectiveFun(self, force=False):
        if self.obj is None or force:
            self.obj0 = 0
            c = float(self.Ntarg) / self.select.sum()
            for k,fv in enumerate(self.fv1):
                if self.select[k]:
                    self.obj0 += c*self.param.fun_obj0(fv, self.param.KparDist) / (self.param.sigmaError**2)

            self.obj = self.obj0

            # Regularization part
            foo = self.objectiveFunDef(self.at, self.Afft, kernel=self.param.KparDiff0, withTrajectory=True,
                                       regWeight=self.lambdaPrior)
            #foo = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.xt = np.copy(foo[1])
            self.fvTmpl.updateVertices(np.squeeze(self.xt[-1]))
            self.obj += foo[0]
            ff = surfaces.Surface(surf=self.fv0)
            for (kk, a) in enumerate(self.atAll):
                if self.select[kk]:
                    foo = self.objectiveFunDef(a, self.AfftAll[kk], withTrajectory=True, x0 = self.fvTmpl.vertices)
                    ff.updateVertices(np.squeeze(foo[1][-1]))
                    self.obj += c*(foo[0] + self.param.fun_obj(ff, self.fv1[kk], self.param.KparDist) / (self.param.sigmaError**2))
                    self.xtAll[kk] = np.copy(foo[1])
                    self.fvDef[kk] = surfaces.Surface(surf=ff)

        return self.obj

    def getVariable(self):
        return self.fvTmpl


    def copyDir(self, dir):
        dfoo = Direction()
        dfoo.prior = np.copy(dir.prior)
        for d in dir.all:
            dfoo.all.append(smatch.Direction())
            dfoo.all[-1].diff = np.copy(d.diff)
            dfoo.all[-1].aff = np.copy(d.aff)
        return(dfoo)

    def randomDir(self):
        dfoo = Direction()
        if self.updateTemplate:
            dfoo.prior = np.random.randn(self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1])
        else:
            dfoo.prior = np.zeros((self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]))
            
        dfoo.all = []
        if self.updateAllTraj:
            for k in range(self.Ntarg):
                dfoo.all.append(smatch.Direction())
                dfoo.all[k].diff = np.random.randn(self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1])
                if self.updateAffine:
                    dfoo.all[k].aff = np.random.randn(self.Tsize, self.affineDim)
                else:
                    dfoo.all[k].aff = np.zeros((self.Tsize, self.affineDim))
        else:
            for k in range(self.Ntarg):
                dfoo.all.append(smatch.Direction())
                dfoo.all[k].diff = np.zeros((self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]))
                dfoo.all[k].aff = np.zeros((self.Tsize, self.affineDim))
        return(dfoo)

    def updateTry(self, direction, eps, objRef=None):
        objTry = self.obj0
        if self.updateTemplate:
            atTry = self.at - eps * direction.prior
        else:
            atTry = np.copy(self.at)
        #print 'x0 hypertemplate', self.x0.sum()
        foo = self.objectiveFunDef(atTry, self.Afft, kernel = self.param.KparDiff0, withTrajectory=True, regWeight=self.lambdaPrior)
        objTry += foo[0]
        x0 = np.squeeze(foo[1][-1,...])
        fvTmplTry = surfaces.Surface(surf=self.fv0)
        fvTmplTry.updateVertices(x0)
        #print 'x0 template', x0.sum()
        #print -1, objTry - self.obj0
        atAllTry = []
        AfftAllTry = []
        _ff = []
        c = float(self.Ntarg)/self.select.sum()
        for (kk, d) in enumerate(direction.all):
            if self.select[kk]:
                #if self.compGrd[kk]:
                atAllTry.append(self.atAll[kk] - eps * d.diff)
                if self.updateAffine:
                    AfftAllTry.append(self.AfftAll[kk] - eps * d.aff)
                else:
                    AfftAllTry.append(np.copy(self.AfftAll[kk]))
                foo = self.objectiveFunDef(atAllTry[-1], AfftAllTry[-1], kernel = self.param.KparDiff, withTrajectory=True, x0 = x0)
                objTry += c*foo[0]
                ff = surfaces.Surface(surf=self.fv0)
                ff.updateVertices(np.squeeze(foo[1][-1,...]))
                #print 'x0 deformed template', ff.vertices.sum()
                _ff.append(ff)
            else:
                _ff.append([])
                atAllTry.append(np.copy(self.atAll[kk]))
                AfftAllTry.append(np.copy(self.AfftAll[kk]))

        objTry += self.dataTerm(_ff)

        if (objRef is None) | (objTry < objRef):
            self.atTry = atTry
            self.atAllTry = atAllTry
            self.AfftAllTry = AfftAllTry
            self.objTry = objTry
            self.fvTmplTry = surfaces.Surface(fvTmplTry)
            for (kk,f) in enumerate(_ff):
                if self.select[kk]:
                    self.fvDefTry[kk] = surfaces.Surface(surf=f)
            #logging.info('updateTry ' + str(self.dataTerm(_ff)))

        return objTry


    def gradientComponent(self, q, kk):
        #print kk, 'th gradient'
        if self.param.errorType == 'L2Norm':
            px1 = -surfaces.L2NormGradient(self.fvDef[kk], self.fv1[kk].vfld) / self.param.sigmaError ** 2
        else:
            px1 = -self.param.fun_objGrad(self.fvDef[kk], self.fv1[kk], self.param.KparDist) / self.param.sigmaError**2
            #print 'in fun' ,kk
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.AfftAll[kk][t])
                #print self.dim, dim2, AB.shape, self.affineBasis.shape
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        foo = self.hamiltonianGradient(px1, kernel=self.param.KparDiff, x0=self.fvTmpl.vertices, at=self.atAll[kk], affine=A)
        grd = foo[0:3]
        pxTmpl = foo[4][0, ...]

        #print kk, (px1-pxTmpl).sum()
        #print 'before put', kk
        q.put([kk, grd, pxTmpl])
        #print 'end fun', kk
        return True

    def testEndpointGradient(self):
        c0 = self.dataTerm(self.fvDef)
        _ff = []
        _dff = []
        incr = 0
        eps = 1e-6
        for k in range(self.Ntarg):
            ff = surfaces.Surface(surf=self.fvDef[k])
            dff = np.random.normal(size=ff.vertices.shape)
            ff.updateVertices(ff.vertices+eps*dff)
            _ff.append(ff)
            _dff.append(dff)
            if self.param.errorType == 'L2Norm':
                grd = surfaces.L2NormGradient(self.fvDef[k], self.fv1[k].vfld) / self.param.sigmaError ** 2
            else:   
                grd = self.param.fun_objGrad(self.fvDef[k], self.fv1[k], self.param.KparDist) / self.param.sigmaError**2
            incr += (grd*dff).sum()
        c1 = self.dataTerm(_ff)
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps, incr ))

    def getGradient(self, coeff=1.0):
        #grd0 = np.zeros(self.atAll.shape)
        #pxIncr = np.zeros([self.Ntarg, self.atAll.shape[1], self.atAll.shape[2]])
        pxTmpl = np.zeros(self.at.shape[1:3])
        q = mp.Queue()

        useMP = False
        if useMP:
            procGrd = []
            for kk in range(self.Ntarg):
                procGrd.append(mp.Process(target = self.gradientComponent, args=(q,kk)))
            for kk in range(self.Ntarg):
                print kk, self.Ntarg
                procGrd[kk].start()
            print "end start"
            for kk in range(self.Ntarg):
                print "join", kk
                procGrd[kk].join()
                self.compGrd[kk] = True
            # print 'all joined'
        else:
            #select = np.random.permutation(self.Ntarg)
            for kk in range(self.Ntarg):
                if self.select[kk]:
                    self.gradientComponent(q, kk)

        grd = Direction()
        for kk in range(self.Ntarg):
            #self.gradientComponent(q, kk)
            #if self.compGrd[kk]:
            dir = smatch.Direction()
            dir.diff = np.zeros(self.at.shape)
            dir.aff = np.zeros(self.Afft.shape)
            grd.all.append(dir)

        dim2 = self.dim**2
        c = float(self.Ntarg)/self.select.sum()
        while not q.empty():
            foo = q.get()
            #print 'got', kk
            dat = foo[1][0]/(coeff*self.Tsize)
            dAfft = np.zeros(self.AfftAll[foo[0]].shape)
            if self.affineDim > 0:
                dA = foo[1][1]
                db = foo[1][2]
                dAfft = 2 *self.affineWeight.reshape([1, self.affineDim]) * self.AfftAll[foo[0]]
                for t in range(self.Tsize):
                    dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
                    dAfft[t] -=  dAff.reshape(dAfft[t].shape)
                dAfft /= (self.coeffAff*coeff*self.Tsize)
            #c = (float(self.Ntarg)/self.sgd)
            grd.all[foo[0]].diff = c*dat
            grd.all[foo[0]].aff = c*dAfft
            #print kk, foo[2].sum()
            pxTmpl += c*foo[2]

        #print q.get()
        #print pxTmpl.sum()
        #print 'Template gradient'
        foo2 = self.hamiltonianGradient(pxTmpl, kernel = self.param.KparDiff0, regWeight=self.lambdaPrior, x0=self.x0, at=self.at)
        #print self.at.shape, self.atAll[0].shape
        #print((foo2[1][-1,...]-self.fvTmpl.vertices)**2).bit_length
        if self.updateTemplate:
            grd.prior = foo2[0] / (self.tmplCoeff*coeff*self.Tsize)
        else:
            grd.prior = np.zeros(foo2[0].shape)
        #print 'grds', grd.prior[5,...].sum(), grd.all[0].diff[5,...].sum()
        #print 'grds', grd.prior.shape, grd.all[0].diff.shape
        return grd

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for (kk, gg1) in enumerate(g1.all):
            if self.select[kk]:
                for t in range(self.Tsize):
                    #print gg1[0].shape, gg1[1].shape
                    z = np.squeeze(self.xtAll[kk][t, :, :])
                    gg = np.squeeze(gg1.diff[t, :, :])
                    u = self.param.KparDiff.applyK(z, gg)
                    #uu = np.multiply(gg1.aff[t], self.affineWeight.reshape(gg1.aff[t].shape))
                    uu = gg1.aff[t]
                    #u = rzz*gg
                    ll = 0
                    for gr in g2:
                        ggOld = np.squeeze(gr.all[kk].diff[t, :, :])
                        res[ll]  += (ggOld*u).sum()
                        if self.affineDim > 0:
                            res[ll] += (uu*gr.all[kk].aff[t]).sum() * self.coeffAff
                        ll = ll + 1
        for t in range(g1.prior.shape[0]):
            z = np.squeeze(self.xt[t, :, :])
            gg = np.squeeze(g1.prior[t, :, :])
            u = self.param.KparDiff0.applyK(z, gg)
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.prior[t, :, :])
                res[ll]  += (ggOld*u).sum()*(self.tmplCoeff)
                ll = ll + 1
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.fvTmpl = surfaces.Surface(self.fvTmplTry)
        for kk,a in enumerate(self.atAllTry):
            if self.select[kk]:
                self.atAll[kk] = np.copy(a)
                self.AfftAll[kk] = np.copy(self.AfftAllTry[kk])
                self.fvDef[kk] = surfaces.Surface(surf=self.fvDefTry[kk])
        #logging.info('Obj Fun: ' + str(self.objectiveFun(force=True)))



    def addProd(self, dir1, dir2, coeff):
        res = Direction()
        res.prior = dir1.prior + coeff * dir2.prior
        for kk, dd in enumerate(dir1.all):
            res.all.append(smatch.Direction())
            if self.select[kk]:
                res.all[kk].diff = dd.diff + coeff*dir2.all[kk].diff
                res.all[kk].aff = dd.aff + coeff*dir2.all[kk].aff
        return res


    def endOfIteration(self):
        self.iter += 1
        #if self.testGradient:
        #    self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.updateAffine = False
            self.coeffAff = self.coeffAff2
        if self.iter >= self.templateBurnIn:
            self.updateAllTraj = True
        (obj1, self.xt, Jt) = self.objectiveFunDef(self.at, self.Afft, kernel = self.param.KparDiff0, withTrajectory=True, withJacobian=True)
        self.fvTmpl.updateVertices(np.squeeze(self.xt[-1, :, :]))
        self.fvTmpl.saveVTK(self.outputDir +'/'+ 'Template.vtk', scalars = Jt[-1], scal_name='Jacobian')
        for kk in range(self.Ntarg):
            if self.select[kk]:
                (obj1, self.xtAll[kk], Jt) = self.objectiveFunDef(self.atAll[kk], self.AfftAll[kk], kernel = self.param.KparDiff,
                                                                  withTrajectory=True, x0 = self.fvTmpl.vertices, withJacobian=True)
                self.fvDef[kk].updateVertices(self.xtAll[kk][self.xtAll[kk].shape[0]-1, ...])
                self.fvDef[kk].saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[-1, :], scal_name='Jacobian')
        #logging.info('Obj Fun: ' + str(self.objectiveFun(force=True)))


    #def endOfProcedure(self):
    #    logging.info('Obj Fun: ' + str(self.objectiveFun(force=True)))


    def computeTemplate(self):
        # self.coeffAff = self.coeffAff2
        # grd = self.getGradient(self.gradCoeff)
        # [grd2] = self.dotProduct(grd, [grd])
        # self.coeffAff = self.coeffAff1
        # self.gradEps = max(0.1, np.sqrt(grd2) / 10000)
        if self.sgd < self.Ntarg:
            for k in range(self.maxIter):
                self.select = np.zeros(self.Ntarg, dtype=bool)
                sel = np.random.permutation(self.Ntarg)
                #sel[0] = 8
                s = 'Selected: '
                for kk in range(self.sgd):
                    s += str(sel[kk]) + ' '
                    self.select[sel[kk]] = True
                logging.info('\nRandom step ' + str(k) + ' ' + s)
                self.epsMax = .01/(k+1)
                self.reset = True
                cg.cg(self, verb=self.verb, maxIter=10, TestGradient=False, epsInit=0.01)

            #sgd = (10, 0.00001)
        else:
            cg.cg(self, verb=self.verb, maxIter=self.maxIter, TestGradient=True, epsInit=0.01)
        return self

if __name__ == "__main__":
    fv = []
    Test = False
    if Test:
        for k in range(1,11):
            fv.append(Surface(filename ='/Users/younes/Development/Data/sculptris/Dataset/surface'+str(k)+'.vtk'))
        htmpl = fv[0]

    else:
        fls = glob.glob('/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/2_qc_flipped_registered/*_1_*_hippo_*_reg.byu')
        if (len(fls) > 0):
            for name in fls:
                fv.append(surfaces.Surface(filename=name))
        hf = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template_smooth/newTemplate.byu'
        htmpl = surfaces.Surface(filename=hf)

    loggingUtils.setup_default_logging('/Users/younes/Results/surfaceTemplate2', fileName='info.txt', stdOutput = True)
    logging.info('Read {0:d} surfaces'.format(len(fv)))
    K1 = kfun.Kernel(name='laplacian', sigma = 5)
    K2 = kfun.Kernel(name='gauss', sigma = 2.5)

    sm = SurfaceTemplateParam(timeStep=0.1, KparDiff=K1, KparDist=K2, sigmaError=1., errorType='current')
    f = SurfaceTemplate(HyperTmpl=htmpl, Targets=fv, outputDir='/Users/younes/Results/surfaceTemplateBiocard',param=sm, testGradient=False,
                        lambdaPrior = 1, maxIter=1000, affine='none', rotWeight=10., sgd=10,
                        transWeight = 1., scaleWeight=10., affineWeight=100.)
    f.computeTemplate()




