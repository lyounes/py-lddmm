import numpy.linalg as LA
import numpy as np
from .bfgs import bfgs
from.conjugateGradient import cg
from . import gaussianDiffeons as gd
from .surfaces import Surface
from . import diffeonEvolution as evol
from .surfaceMatching import Control, SurfaceMatching

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normalization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class SurfaceMatchingParam(surfaceMatching.SurfaceMatchingParam):
#     def __init__(self, timeStep = .1, sigmaKernel = 6.5, algorithm='bfgs',
#                  sigmaDist=2.5, sigmaError=1.0, errorType='measure'):
#         super().__init__(timeStep = timeStep, sigmaKernel =  sigmaKernel, errorType=errorType,
#                        algorithm=algorithm, sigmaDist = sigmaDist, sigmaError = sigmaError, typeKernel ='gauss')
#         self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
#         self.KparDist = kfun.Kernel(name = 'gauss', sigma = self.sigmaDist)
#         self.errorType = errorType
#         if errorType == 'current':
#             self.fun_obj0 = surfaces.currentNorm0
#             self.fun_obj = surfaces.currentNormDef
#             self.fun_objGrad = surfaces.currentNormGradient
#         elif errorType == 'measure':
#             self.fun_obj0 = surfaces.measureNorm0
#             self.fun_obj = surfaces.measureNormDef
#             self.fun_objGrad = surfaces.measureNormGradient
#         elif errorType == 'varifold':
#             self.fun_obj0 = surfaces.varifoldNorm0
#             self.fun_obj = surfaces.varifoldNormDef
#             self.fun_objGrad = surfaces.varifoldNormGradient
#         elif errorType=='diffeonCurrent':
#             self.fun_obj0 = gd.diffeonCurrentNorm0
#             self.fun_obj = gd.diffeonCurrentNormDef
#             self.fun_objGrad = gd.diffeonCurrentNormGradient
#         else:
#             print('Unknown error Type: ', self.errorType)
	
 

# class Direction:
#     def __init__(self):
#         self.diff = []
#         self.aff = []


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

class State(dict):
    def __init__(self):
        super().__init__()
        self['Jt'] = None
        self['ct'] = None
        self['St'] = None
        self['bt'] = None
        self['xSt'] = None
 

class SurfaceMatchingDiffeons(SurfaceMatching):
    def __init__(self, Template=None, Target=None, options = None):
        # Diffeons=None, EpsilonNet=None, DecimationTarget=None,
        #          subsampleTargetSize = -1, 
        #          DiffeonEpsForNet=None, DiffeonSegmentationRatio=None, zeroVar=False, fileTempl=None,
        #          fileTarg=None, param=None, maxIter=1000, regWeight = 1.0, affineWeight = 1.0, verb=True,
        #          rotWeight = None, scaleWeight = None, transWeight =
        super().__init__(Template, Target, options)


        
    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['dimension'] = 2
        options['errorType'] = 'L2'
        options['Diffeons'] = None
        options['DiffeonEpsForNet']=None 
        options['DiffeonSegmentationRatio']=None
        options['EpsilonNet'] = None
        options['DecimationTarget'] = 1
        options['zeroVar'] = False
        options['gradEps'] = -1
        return options


    def set_template_and_target(self, Template, Target, misc=None):
        super().set_template_and_target(Template, Target, misc)
        self.fv0Fine = Surface(surf=self.fv0)

        if self.options['Diffeons'] is None:
            if self.options['EpsilonNet'] is None:
                if self.options['DecimationTarget'] is None:
                    if self.options['DiffeonEpsForNet'] is None:
                        if self.options['DiffeonSegmentationRatio'] is None:
                            self.c0 = np.copy(self.x0)
                            self.S0 = np.zeros([self.x0.shape[0], self.x0.shape[1], self.x0.shape[1]])
                            self.idx = None
                        else:
                            (self.c0, self.S0, self.idx) = \
                                gd.generateDiffeonsFromSegmentation(self.fv0, self.options['DiffeonSegmentationRatio'])
                            #self.S0 *= self.options['sigmaKernel']**2;
                    else:
                        (self.c0, self.S0, self.idx) = \
                            gd.generateDiffeonsFromNet(self.fv0, self.options['DiffeonEpsForNet'])
                else:
                    (self.c0, self.S0, self.idx) = \
                        gd.generateDiffeonsFromDecimation(self.fv0, self.options['DecimationTarget'])
            else:
                (self.c0, self.S0, self.idx) = \
                    gd.generateDiffeons(self.fv0, self.options['EpsilonNet'][0], self.options['EpsilonNet'][1])
        else:
            (self.c0, self.S0, self.idx) = self.options['Diffeons']
            
        if self.options['zeroVar']:
            self.S0 = np.zeros(self.S0.shape)

            #print self.S0
        if (self.options['subsampleTargetSize'] > 0):
            self.fv0.Simplify(self.options['subsampleTargetSize'])
            v0 = self.fv0.surfVolume()
            v1 = self.fv0Fine.surfVolume()
            if (v0*v1 < 0):
                self.fv0.flipFaces()
            if self.options['errorType'] == 'diffeonCurrent':
                n = self.fv0Fine.vertices.shape[0]
                m = self.fv0.vertices.shape[0]
                dist2 = ((self.fv0Fine.vertices.reshape([n, 1, 3]) -
                          self.fv0.vertices.reshape([1,m,3]))**2).sum(axis=2)
                idx = - np.ones(n, dtype=np.int)
                for p in range(n):
                    closest = np.unravel_index(np.argmin(dist2[p, :].ravel()), [m, 1])
                    idx[p] = closest[0]
                (x0, xS0, idx) = gd.generateDiffeons(self.fv0Fine, self.fv0.vertices, idx)
                b0 = gd.approximateSurfaceCurrent(x0, xS0, self.fv0Fine, self.options['KparDist'].sigma)
                gdOpt = gd.gdOptimizer(surf=self.fv0Fine, sigmaDist = self.options['KparDist'].sigma, 
                                       Diffeons = (x0, xS0, b0) , testGradient=False, maxIter=50)
                gdOpt.optimize()
                self.x0 = gdOpt.c0
                self.xS0 = gdOpt.S0
                self.b0 = gdOpt.b0
            else:
                self.x0 = self.fv0.vertices
            print('simplified template', self.fv0.vertices.shape[0])
            
        self.fvDef = Surface(surf=self.fv0)
        self.ndf = self.c0.shape[0]
        
    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.control = Control()
        self.controlTry = Control()
        self.control['at'] = np.zeros([self.Tsize, self.c0.shape[0], self.x0.shape[1]])
        self.controlTry['at'] = np.zeros([self.Tsize, self.c0.shape[0], self.x0.shape[1]])
        if self.affineDim > 0:
            self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
            # self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.state = State()
        self.state['ct'] = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.state['St'] = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        print('error type:', self.options['errorType'])
        if self.options['errorType'] =='diffeonCurrent':
            self.xSt = np.tile(self.xS0, [self.Tsize+1, 1, 1, 1])
            self.bt = np.tile(self.b0, [self.Tsize+1, 1, 1])
            self.dcurr = True
        else:
            self.dcurr=False

        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        # self.options['saveFile'] = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')


    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False):
        if var is not None and 'initial' in var:
            x0 = self.x0
            if self.dcurr:
                (c0, S0, b0, xS0) = var['initial']
            else:
                (c0, S0) = var['initial']
                b0 = None
                xS0 = None
        else:
            x0 = self.x0
            c0 = self.c0
            S0 = self.S0
            if self.dcurr:
                b0 = self.b0
                xS0 = self.xS0
            else:
                b0 = None
                xS0 = None

        timeStep = 1.0/self.Tsize
        if self.affineDim > 0:
            dim2 = self.dim ** 2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, control['Afft'][t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        else:
            A = None

        bt = None
        xSt = None
        Jt = None
        if withJacobian:
            if self.dcurr:
                (ct, St, bt, xt, xSt, Jt)  = \
                    evol.gaussianDiffeonsEvolutionEuler(c0, S0, control['at'], self.options['sigmaKernel'], affine=A, 
                                                        withJacobian=True, withNormals=b0, withDiffeonSet=(x0, xS0))
            else:
                (ct, St, xt, Jt)  = \
                    evol.gaussianDiffeonsEvolutionEuler(c0, S0, control['at'], self.options['sigmaKernel'], affine=A, 
                                                        withPointSet = x0, withJacobian=True)
        else:
            if self.dcurr:
                (ct, St, bt, xt, xSt)  \
                    = evol.gaussianDiffeonsEvolutionEuler(c0, S0, control['at'], self.options['sigmaKernel'], affine=A, 
                                                          withNormals=b0, withDiffeonSet=(x0, xS0))
            else:
                (ct, St, xt)  = evol.gaussianDiffeonsEvolutionEuler(c0, S0, control['at'], self.options['sigmaKernel'],
                                                                    affine=A, withPointSet = x0)


        state = State()
        state['ct'] = ct
        state['xt'] = xt
        state['St'] = St
        state['Jt'] = Jt
        state['xSt'] = xSt
        state['bt'] = bt
        #print xt[-1, :, :]
        #print obj
        obj=0
        #print St.shape
        for t in range(self.Tsize):
            c = ct[t, :, :]
            S = St[t, :, :, :]
            a = control['at'][t, :, :]
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            rcc = gd.computeProducts(c, S, self.options['sigmaKernel'])
            obj = obj + self.options['regWeight']*timeStep* (a * (rcc@a)).sum()
            if self.affineDim > 0:
                obj +=  timeStep * (self.affineWeight.reshape(control['Afft'][t].shape) * control['Afft'][t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withJacobian or withTrajectory:
            return obj, state
        else:
            return obj

    def dataTerm(self, _data, var = None):
        if self.dcurr:
            obj = self.fun_obj(_data[0], _data[1], _data[2], self.fv1, self.options['KparDist'].sigma) \
                  / (self.options['sigmaError']**2)
        else:
            obj = self.fun_obj(_data, self.fv1) / (self.options['sigmaError']**2)
        return obj
    
    def objectiveFun(self):
        if self.obj is None:
            self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError']**2)
            (self.obj, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
            if self.dcurr:
                data = (self.state['xt'][-1,:,:], self.state['xSt'][-1,:,:,:], self.state['bt'][-1,:,:])
                self.obj += self.obj0 + self.dataTerm(data)
            else:
                self.fvDef.updateVertices(np.squeeze(self.state['xt'][-1, :, :]))
                self.obj += self.obj0 + self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return self.control

    def updateTry(self, dr, eps, objRef=None):
        controlTry = Control()
        for k in dr.keys():
            if dr[k] is not None:
                controlTry[k] = self.control[k] - eps * dr[k]

        objTry, state = self.objectiveFunDef(controlTry, withTrajectory=True)
        if self.dcurr:
            data = (state['xt'][-1,:,:], state['xSt'][-1,:,:,:], state['bt'][-1,:,:])
            objTry += self.obj0 + self.dataTerm(data)
        else:
            ff = Surface(surf=self.fvDef)
            ff.updateVertices(state['xt'][-1, :, :])
            objTry += self.obj0 + self.dataTerm(ff)

        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = controlTry
            self.stateTry = state
            self.objTry = objTry
        return objTry



    def endPointGradient(self, endpoint=None):
        if endpoint is None:
            if self.dcurr:
                endpoint = [self.control['xt'][-1, :, :], self.control['xSt'][-1, :, :, :], 
                            self.control['bt'][-1, :, :]]
            else:
                endpoint = self.fvDef
        if self.dcurr:
            #print self.bt.shape
            (px, pxS, pb) = self.fun_objGrad(endpoint[0], endpoint[1], endpoint[2], self.fv1, 
                                             self.options['KparDist'].sigma)
            pc = np.zeros(self.c0.shape)
            pS = np.zeros(self.S0.shape)
            #gd.testDiffeonCurrentNormGradient(self.ct[-1, :, :], self.St[-1, :, :, :], self.bt[-1, :, :],
            #                               self.fv1, self.options['KparDist'].sigma)
            px = px / self.options['sigmaError']**2
            pxS = pxS / self.options['sigmaError']**2
            pb = pb / self.options['sigmaError']**2
            return (pc, pS, pb, px, pxS)
        else:
            px = self.fun_objGrad(endpoint, self.fv1)/ self.options['sigmaError']**2
            pc = np.zeros(self.c0.shape)
            pS = np.zeros(self.S0.shape)
            return (pc, pS, px) 


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            endPoint = self.fvDef
            A = self.affB.getTransforms(self.control['Afft'])
        else:
            control = Control()
            for k in update[0].keys():
                if update[0][k] is not None:
                    control[k] = self.control[k] - update[1]*update[0][k]

            A = self.affB.getTransforms(control['Afft'])
            if self.dcurr:
                ct ,St ,bt ,xt ,xSt = \
                    evol.gaussianDiffeonsEvolutionEuler(self.c0 ,self.S0 , control['at'] ,self.options['sigmaKernel'] ,
                                                        affine=A, withNormals=self.b0 ,withDiffeonSet=(self.x0 ,self.xS0))
                endPoint = [xt[-1, :, :], xSt[-1, :, :, :], bt[-1, :, :]]
            else:
                ct ,St ,xt = evol.gaussianDiffeonsEvolutionEuler(self.c0 ,self.S0 ,control['at'] ,self.options['sigmaKernel'] ,
                                                                 affine=A ,withPointSet=self.x0)
                endPoint = Surface(surf=self.fv0)
                endPoint.updateVertices(xt[-1, :, :])

        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        # dim2 = self.dim**2
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, self.Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]

        if self.dcurr:
            (pc1, pS1, pb1, px1, pxS1) = self.endPointGradient(endpoint=endPoint)
            foo = evol.gaussianDiffeonsGradientNormals(self.c0, self.S0, self.b0, self.x0, self.xS0,
                                                       control['at'], -pc1, -pS1, -pb1, -px1, -pxS1, 
                                                       self.options['sigmaKernel'], self.options['regWeight'],
                                                       affine=A, euclidean=self.euclideanGradient)
        else:
            (pc1, pS1, px1) = self.endPointGradient(endpoint=endPoint)
            foo = evol.gaussianDiffeonsGradientPset(self.c0, self.S0, self.x0,
                                                    control['at'], -pc1, -pS1, -px1, self.options['sigmaKernel'],
                                                    self.options['regWeight'], affine=A, euclidean=self.euclideanGradient)

        grd = Control()
        grd['at'] = foo['dat']/(coeff*self.Tsize)
            
        if self.affineDim > 0:
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            dim2 = self.dim**2
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2 * self.affineWeight[:,0][None, :] * control['Afft']
            for t in range(self.Tsize):
                dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
                grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            if not self.euclideanGradient:
                grd['Afft'] /= self.affineWeight[:,0][None, :]
            grd['Afft'] /= (coeff*self.Tsize)
        return grd



    # def addProd(self, dir1, dir2, beta):
    #     dir = Direction()
    #     dir.diff = dir1.diff + beta * dir2.diff
    #     dir.aff = dir1.aff + beta * dir2.aff
    #     return dir
    #
    # def copyDir(self, dir0):
    #     dir = Direction()
    #     dir.diff = np.copy(dir0.diff)
    #     dir.aff = np.copy(dir0.aff)
    #     return dir
    #
    #
    def randomDir(self):
        dirfoo = Control()
        dirfoo['at'] = np.random.randn(self.Tsize, self.ndf, self.dim)
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            c = self.state['ct'][t, :, :]
            S = self.state['St'][t, :, :, :]
            gg = g1['at'][t, :, :]
            rcc = gd.computeProducts(c, S, self.options['sigmaKernel'])
            (L, W) = LA.eigh(rcc)
            rcc += (L.max()/1000)*np.eye(rcc.shape[0])
            u = rcc @ gg
            if self.affineDim > 0:
                uu = g1['Afft'][t] * self.affineWeight.reshape(g1['Afft'][t].shape)
            else:
                uu=0
            ll = 0
            for gr in g2:
                ggOld = gr['at'][t, :, :]
                res[ll]  = res[ll] + (ggOld*u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['Afft'][t]).sum()
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1

        return res

    # def dotProduct_euclidean(self, g1, g2):
    #     res = np.zeros(len(g2))
    #     for t in range(self.Tsize):
    #         gg = g1['at'][t, :, :]
    #         uu = g1['Afft'][t]
    #         ll = 0
    #         for gr in g2:
    #             ggOld = gr.diff[t, :, :]
    #             res[ll]  = res[ll] + (ggOld*gg).sum()
    #             if self.affineDim > 0:
    #                 res[ll] += (uu*gr['Afft'][t]).sum()
    #             ll = ll + 1
    # 
    #     return res

    # def acceptVarTry(self):
    #     self.obj = self.objTry
    #     self.control = deepcopy(self.controlTry)
    #     #print self.at

    def saveB(self, fileName, c, b):
        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(c.shape[0]))
            for ll in range(c.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(c[ll,0], c[ll,1], c[ll,2]))
            fvtkout.write(('\nPOINT_DATA {0: d}').format(c.shape[0]))

            fvtkout.write('\nVECTORS bt float')
            for ll in range(c.shape[0]):
                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(b[ll, 0], b[ll, 1], b[ll, 2]))
            fvtkout.write('\n')


    def endOfIteration(self, forceSave = False):
        #print self.obj0
        self.iter += 1
        if (self.iter % self.options['saveRate'] == 0) :
            print('saving...')
            (obj1, self.state) = self.objectiveFunDef(self.control, withTrajectory=True, withJacobian=True)
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])
            dim2 = self.dim**2
            if self.affineDim > 0:
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.control['Afft'][t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            else:
                A = None
            (ct, St, xt, Jt) = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.control['at'],
                                                                   self.options['sigmaKernel'], affine=A,
                                                                   withPointSet = self.fv0Fine.vertices,
                                                                   withJacobian=True)
            fvDef = Surface(surf=self.fv0Fine)
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(xt[kk, :, :])
                # foo = (gd.diffeonCurrentNormDef(self.xt[kk], self.xSt[kk], self.bt[kk], self.fvDef, self.options['KparDist'].sigma)
                #        + gd.diffeonCurrentNorm0(self.fvDef, self.options['KparDist']))/ (self.options['sigmaError']**2)
                # print foo
                fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk',
                              scalars = Jt[kk, :], scal_name='Jacobian')
                gd.saveDiffeons(self.outputDir +'/'+ self.options['saveFile']+'Diffeons'+str(kk)+'.vtk',
                                self.state['ct'][kk,:,:], self.state['St'][kk,:,:,:])
                if self.dcurr:
                    self.saveB(self.outputDir +'/'+ self.options['saveFile']+'Bt'+str(kk)+'.vtk',
                               self.state['xt'][kk,:,:], self.state['bt'][kk,:,:])
            # else:
            #     (obj1, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
            #     if self.affineDim > 0:
            #         dim2 = self.dim ** 2
            #         A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            #         for t in range(self.Tsize):
            #             AB = np.dot(self.affineBasis, self.control['Afft'][t])
            #             A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
            #             A[1][t] = AB[dim2:dim2+self.dim]
            #     else:
            #         A = None
            #     (ct, St, xt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.control['at'],
            #                                                             self.options['sigmaKernel'], affine=A,
            #                                                             withPointSet = self.fv0Fine.vertices,
            #                                                             withJacobian=True)
            #     fvDef = surfaces.Surface(surf=self.fv0Fine)
            #     for kk in range(self.Tsize+1):
            #         fvDef.updateVertices(xt[kk, :, :])
            #         fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk', scalars = Jt[kk, :],
            #                       scal_name='Jacobian')
            #             #self.fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')
            #         gd.saveDiffeons(self.outputDir +'/'+ self.options['saveFile']+'Diffeons'+str(kk)+'.vtk',
            #                         self.state['ct'][kk,:,:], self.state['St'][kk,:,:,:])
            #
            # #print self.bt

        else:
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True)
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])

    def restart(self, EpsilonNet=None, DiffeonEpsForNet=None, DiffeonSegmentationRatio=None, DecimationTarget=None):
        if EpsilonNet is None:
            if DecimationTarget is None:
                if DiffeonEpsForNet is None:
                    if DiffeonSegmentationRatio is None:
                        c0 = np.copy(self.x0)
                        S0 = np.zeros([self.x0.shape[0], self.x0.shape[1], self.x0.shape[1]])
                        #net = range(c0.shape[0])
                        idx = range(c0.shape[0])
                    else:
                        (c0, S0, idx) = gd.generateDiffeonsFromSegmentation(self.fv0, DiffeonSegmentationRatio)
                        #self.S0 *= self.options['sigmaKernel']**2;
                else:
                    (c0, S0, idx) = gd.generateDiffeonsFromNet(self.fv0, DiffeonEpsForNet)
            else:
                (c0, S0, idx) = gd.generateDiffeonsFromDecimation(self.fv0, DecimationTarget)
        else:
            net = EpsilonNet[2]
            (c0, S0, idx) = gd.generateDiffeons(self.fv0, EpsilonNet[0], EpsilonNet[1])

        (ctPrev, StPrev, ct, St) = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.control['at'],
                                                                       self.options['sigmaKernel'],
                                                                       withDiffeonSet=(c0, S0))
        at = np.zeros([self.Tsize, c0.shape[0], self.x0.shape[1]])
        #fvDef = surfaces.Surface(surf=self.fvDef)
        for t in range(self.Tsize):
            g1 = gd.computeProducts(ct[t,:,:],St[t,:,:], self.options['sigmaKernel'])
            g2 = gd.computeProductsAsym(ct[t,:,:],St[t,:,:], ctPrev[t,:,:], StPrev[t,:,:], self.options['sigmaKernel'])
            g2a = g2 @ self.control['at'][t, :, :]
            at[t, :, :] = LA.solve(g1, g2a)
            g0 = gd.computeProducts(ctPrev[t,:,:], StPrev[t,:,:], self.options['sigmaKernel'])
            n0 = (self.control['at'][t, :, :] * (g0 @ self.control['at'][t, :, :])).sum()
            n1 = (at[t, :, :] * (g1 @ at[t, :, :])).sum()
            print('norms: ', n0, n1)
            # fvDef.updateVertices(np.squeeze(self.xt[t, :, :]))
            # (AV, AF) = fvDef.computeVertexArea()
            # weights = np.zeros([c0.shape[0], self.c0.shape[0]])
            # diffArea = np.zeros(self.c0.shape[0])
            # diffArea2 = np.zeros(c0.shape[0])
            # for k in range(self.npt):
            #     diffArea[self.idx[k]] += AV[k] 
            #     diffArea2[idx[k]] += AV[k]
            #     weights[idx[k], self.idx[k]] += AV[k]
            # weights /= diffArea.reshape([1, self.c0.shape[0]])
            # at[t] = np.dot(weights, self.at[t, :, :])
        self.c0 = c0
        self.idx = idx
        self.S0 = S0
        self.control['at'] = at
        self.ndf = self.c0.shape[0]
        self.state['ct'] = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.state['St'] = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        if self.dcurr:
            self.b0 = gd.approximateSurfaceCurrent(self.c0, self.S0, self.fv0, self.options['KparDist'].sigma)
            #print self.b0.shape
            self.state['bt'] = np.tile(self.b0, [self.Tsize+1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        self.optimizeMatching()


    def optimizeMatching(self):
        obj0 = self.fun_obj0(self.fv1) # / (self.options['sigmaError']**2)
        (obj, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
        if self.dcurr:
            data = (self.state['xt'][-1,:,:], self.state['xSt'][-1,:,:,:], self.state['bt'][-1,:,:])
            print('objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(data)* (self.options['sigmaError']**2))
            # print(obj0 + surfaces.currentNormDef(self.fv0, self.fv1, self.options['KparDist']))
        else:
            self.fvDef.updateVertices(np.squeeze(self.state['xt'][-1, :, :]))
            print('objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(self.fvDef))

        if self.gradEps < 0:
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])

            self.gradEps = max(0.001, np.sqrt(grd2) / 10000)

        print('Gradient lower bound: ', self.gradEps)
        if self.options['algorithm'] == 'cg':
            cg(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=.01,
                  Wolfe=self.options['Wolfe'])
        elif self.options['algorithm'] == 'bfgs':
            bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=1.,
                      Wolfe=self.options['Wolfe'], lineSearch=self.options['lineSearch'], memory=50)
        # if self.param.algorithm == 'cg':
        #     cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        # else:
        #     bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt

