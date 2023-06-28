from copy import deepcopy
import numpy as np
import scipy.linalg as la
import logging
from functools import partial
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, bfgs
from .pointSets import PointSet
from . import pointSets, pointset_distances as psd
from .affineBasis import AffineBasis
from .basicMatching import BasicMatching


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class PointSetMatchingParam(matchingParam.MatchingParam):
#     def __init__(self, timeStep = .1, algorithm='cg', Wolfe=True, KparDiff = None, KparDist = None,
#                  sigmaError = 1.0, errorType = 'measure'):
#         super().__init__(timeStep=timeStep, algorithm = algorithm, Wolfe=Wolfe,
#                          KparDiff = KparDiff, KparDist = KparDist, sigmaError=sigmaError,
#                          errorType = errorType)
#         self.sigmaError = sigmaError
#

class Control(dict):
    def __init__(self):
        super().__init__()
        self['at'] = None
        self['Afft'] = None

class State(dict):
    def __init__(self):
        super().__init__()
        self['xt'] = None
        self['yt'] = None
        self['Jt'] = None


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
class PointSetMatching(BasicMatching):
    def __init__(self, Template=None, Target=None, options = None):
        super().__init__(Template, Target, options)

        if self.options['algorithm'] == 'cg':
             self.euclideanGradient = False
        else:
            self.euclideanGradient = True


        self.Kdiff_dtype = self.options['pk_dtype']
        self.Kdist_dtype = self.options['pk_dtype']
        self.gradCoeff = 1 #self.x0.shape[0] ** 2

    def createObject(self, data, other=None):
        if other is None:
            return PointSet(data=input)
        else:
            return PointSet(data=input, weights=other)

    def updateObject(self, object, data, other=None):
        return object.updateVertices(data)

    def solveStateEquation(self, control= None, init_state = None, kernel = None, options=None):
        if control is None:
            control = self.control
        if init_state is None:
            init_state = self.x0
        if kernel is None:
            kernel = self.options['KparDiff']

        A = self.affB.getTransforms(control['Afft'])

        return evol.landmarkDirectEvolutionEuler(init_state, control['at'], kernel,
                                                 affine=A, options=options)


    def initialize_variables(self):
        self.x0 = np.copy(self.fv0.vertices)
        self.fvDef = deepcopy(self.fv0)
        self.npt = self.x0.shape[0]
        # self.u = np.zeros((self.dim, 1))
        # self.u[0:2] = 1/np.sqrt(2.)

        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.control['at'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.controlTry['at'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.affineDim > 0:
            self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.state['xt'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])


    def setDotProduct(self, unreduced=False):
        if self.options['algorithm'] == 'cg' and not unreduced:
             self.euclideanGradient = False
             self.dotProduct = self.dotProduct_Riemannian
        else:
            self.euclideanGradient = True
            self.dotProduct = self.dotProduct_euclidean

    def set_template_and_target(self, Template, Target, misc=None):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            if isinstance(Template, PointSet):
                self.fv0 = deepcopy(Template)
            else:
                self.fv0 = pointSets.loadlmk(Template)[0]

        if Target is None:
            logging.error('Please provide a target surface')
            return
        else:
            if isinstance(Target, PointSet):
                self.fv1 = deepcopy(Target)
            else:
                self.fv1 = pointSets.loadlmk(Template)[0]

        self.fv0.save(self.outputDir + '/Template.vtk')
        self.fv1.save(self.outputDir + '/Target.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def set_fun(self, errorType, vfun=None):
        self.options['errorType'] = errorType
        if errorType == 'L2':
            self.fun_obj0 = psd.L2Norm0
            self.fun_obj = psd.L2NormDef
            self.fun_objGrad = psd.L2NormGradient
        elif errorType == 'measure':
            self.fun_obj0 = partial(psd.measureNorm0, KparDist=self.options['KparDist'])
            self.fun_obj = partial(psd.measureNormDef, KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(psd.measureNormGradient, KparDist=self.options['KparDist'])
        else:
            logging.error('Unknown error Type: ' + self.options['errorType'])

        self.extraTerm = None

    def set_parameters(self):
        super().set_parameters()
        sigmaKernel = 6.5
        orderKernel = 3
        sigmaDist = 2.5
        orderKDist = 3
        typeKDist = 'gauss'
        typeKernel = 'gauss'

        if type(self.options['KparDiff']) in (list,tuple):
            typeKernel = self.options['KparDiff'][0]
            sigmaKernel = self.options['KparDiff'][1]
            if typeKernel == 'laplacian' and len(self.options['KparDiff']) > 2:
                orderKernel = self.options['KparDiff'][2]
            self.options['KparDiff'] = None

        if self.options['KparDiff'] is None:
            self.options['KparDiff'] = kfun.Kernel(name = typeKernel, sigma = sigmaKernel, order=orderKernel)

        if type(self.options['KparDist']) in (list,tuple):
            typeKDist = self.options['KparDist'][0]
            sigmaDist = self.options['KparDist'][1]
            if typeKDist == 'laplacian' and len(self.options['KparDist']) > 2:
                orderKdist = self.options['KparDist'][2]
            self.options['KparDist'] = None

        if self.options['KparDist'] is None:
            self.options['KparDist'] = kfun.Kernel(name = typeKDist, sigma = sigmaDist, order= orderKDist)

        self.options['KparDiff'].pk_dtype = self.options['pk_dtype']
        self.options['KparDist'].pk_dtype = self.options['pk_dtype']

        self.gradEps = self.options['gradTol']
        self.affineOnly = self.options['affineOnly']

        if self.options['affineKernel']:
            if self.options['affine'] in ('euclidean', 'affine'):
                if self.options['affine'] == 'euclidean' and self.options['rotWeight'] is not None:
                    w1 = self.options['rotWeight']
                else:
                    w1 = self.options['affineWeight']
                if self.options['transWeight'] is not None:
                    w2 = self.options['transWeight']
                else:
                    w2 = self.options['affineWeight']
                self.options['KparDiff'].setAffine(self.options['affine'], w1=1/w1, w2=1/w2,
                                                   center=self.fv0.vertices.mean(axis=0))
                self.options['affine'] = 'none'
                self.affB = AffineBasis(self.dim, 'none')
                self.affineDim = self.affB.affineDim
            else:
                logging.info('Affine kernels only Euclidean or full affine')
                self.options['affineKernel'] = False

        if not self.options['affineKernel']:
            self.affB = AffineBasis(self.dim, self.options['affine'])
            self.affineDim = self.affB.affineDim
            self.affineBasis = self.affB.basis
            self.affineWeight = self.options['affineWeight'] * np.ones([self.affineDim, 1])
            if (len(self.affB.rotComp) > 0) and (self.options['rotWeight'] is not None):
                self.affineWeight[self.affB.rotComp] = self.options['rotWeight']
            if (len(self.affB.simComp) > 0) and (self.options['scaleWeight'] is not None):
                self.affineWeight[self.affB.simComp] = self.options['scaleWeight']
            if (len(self.affB.transComp) > 0) and (self.options['transWeight'] is not None):
                self.affineWeight[self.affB.transComp] = self.options['transWeight']

        self.coeffInitx = .1
        self.forceLineSearch = False
        self.saveEPDiffTrajectories = False
        self.varCounter = 0
        self.trajCounter = 0
        self.pkBuffer = 0




    def dataTerm(self, _fvDef, var = None):
        if self.options['errorType'] == 'measure':
            obj = self.fun_obj(_fvDef, self.fv1) / (self.options['sigmaError'] ** 2)
        else:
            obj = self.fun_obj(_fvDef, self.fv1) / (self.options['sigmaError']**2)
        return obj

    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False):
        if var is None or 'x0' not in var:
            x0 = self.x0
        else:
            x0= var['x0']
        if var is None or 'kernel' not in var:
            kernel = self.options['KparDiff']
        else:
            kernel = var['kernel']
        #print 'x0 fun def', x0.sum()
        if var is None or 'regWeight' not in var:
            regWeight = self.options['regWeight']
        else:
            regWeight = var['regWeight']

        if 'Afft' not in control:
            Afft = None
        else:
            Afft = control['Afft']

        at = control['at']
        st = State()
        timeStep = 1.0/self.Tsize
        # dim2 = self.dim**2
        A = self.affB.getTransforms(Afft)
        # if self.affineDim > 0:
        #     A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
        # else:
        #     A = None
        if withJacobian:
            st  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A, options={'withJacobian':True})
            # st['xt'] = xt
            # st['Jt'] = Jt
        else:
            st  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A)
            # st['xt'] = xt

        obj=0
        obj1 = 0 
        for t in range(self.Tsize):
            z = st['xt'][t, :, :]
            a = at[t, :, :]
            ra = kernel.applyK(z, a)
            if hasattr(self, 'v'):  
                self.v[t, :] = ra
            obj = obj + regWeight*timeStep*(a*ra).sum()

            if self.extraTerm is not None:
                obj += self.extraTerm['coeff'] * self.extraTerm['fun'](z, ra) * timeStep
            if self.affineDim > 0:
                obj1 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        #print obj, obj+obj1
        obj += obj1
        if withTrajectory or withJacobian:
            return obj, st
        else:
            return obj

    def makeTryInstance(self, state):
        ff = self.createObject(state['xt'][-1,:,:], other=self.fv0.weights)
        return ff

    def objectiveFun(self):
        if self.obj == None:
            if self.options['errorType'] == 'measure':
                self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError'] ** 2)
            else:
                self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError']**2)
            self.objDef, self.state = self.objectiveFunDef(self.control, withTrajectory=True)
            self.fvDef.vertices = np.copy(np.squeeze(self.state['xt'][-1, :, :]))
            self.objData = self.dataTerm(self.fvDef)
            self.obj = self.obj0 + self.objData + self.objDef
        return self.obj

    def getVariable(self):
        return self.control
    def initVariable(self):
        return Control()

    def updateTry(self, dr, eps, objRef=None):
        controlTry = self.initVariable()
        for k in dr.keys():
            if dr[k] is not None:
                controlTry[k] = self.control[k] - eps * dr[k]

        objTryDef, st = self.objectiveFunDef(controlTry, withTrajectory=True)
        ff = self.makeTryInstance(st)
        objTryData = self.dataTerm(ff)
        objTry = self.obj0 + objTryData + objTryDef

        if np.isnan(objTry):
            # logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = deepcopy(controlTry)
            self.objTry = objTry
            self.objTryData = objTryData
            self.objTryDef = objTryDef

        return objTry


    def testEndpointGradient(self):
        c0 = self.dataTerm(self.fvDef)
        ff = deepcopy(self.fvDef)
        dff = np.random.normal(size=ff.vertices.shape)
        eps = 1e-6
        ff.vertices += eps*dff
        c1 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps, (grd*dff).sum()) )

    def endPointGradient(self, endPoint= None):
        if endPoint is None:
            endPoint = self.fvDef
        if self.options['errorType'] == 'measure':
            px = self.fun_objGrad(endPoint, self.fv1)
        else:
            px = self.fun_objGrad(endPoint, self.fv1)
        return px / self.options['sigmaError']**2

    

    def hamiltonianGradient(self, px1, kernel = None, regWeight=None, x0=None, control=None):
        if regWeight is None:
            regWeight = self.options['regWeight']
        if x0 is None:
            x0 = self.x0
        if control is None:
            control = self.control
        affine = self.affB.getTransforms(control['Afft'])
        if kernel is None:
            kernel  = self.options['KparDiff']
        return evol.landmarkHamiltonianGradient(x0, control['at'], px1, kernel, regWeight, affine=affine,
                                                getCovector=True, extraTerm=self.extraTerm)
                                                    
    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]
        A = self.affB.getTransforms(control['Afft'])
        st = evol.landmarkDirectEvolutionEuler(self.x0, control['at'], self.options['KparDiff'], affine=A)
        xt = st['xt']
        endPoint = self.createObject(self.fv0)
        endPoint.updateVertices(xt[-1, :, :])
        st = State()
        st['xt'] = xt

        return control, st, endPoint

    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            endPoint = self.fvDef
            # A = self.affB.getTransforms(self.control['Afft'])
            state = self.state
        else:
            control, state, endPoint = self.setUpdate(update)
            # A = self.affB.getTransforms(control['Afft'])

        dim2 = self.dim**2
        px1 = -self.endPointGradient(endPoint=endPoint)
        foo = self.hamiltonianGradient(px1, control=control)
        grd = Control()
        if self.euclideanGradient:
            grd['at'] = np.zeros(foo[0].shape)
            for t in range(self.Tsize):
                z = state['xt'][t, :, :]
                grd['at'][t,:,:] = self.options['KparDiff'].applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
        else:
            grd['at'] = foo[0]/(coeff*self.Tsize)
        grd['Afft'] = np.zeros(self.control['Afft'].shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*control['Afft']
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] /= (self.coeffAff*coeff*self.Tsize)
        return grd



    def resetPK(self, newType = None):
        if newType is None:
            self.options['KparDiff'].pk_dtype = self.Kdiff_dtype
            self.options['KparDist'].pk_dtype = self.Kdist_dtype
        else:
            self.options['KparDiff'].pk_dtype = newType
            self.options['KparDist'].pk_dtype = newType

    def startOfIteration(self):
        if self.reset:
            logging.info('Switching to 64 bits')
            self.resetPK('float64')
            self.pkBuffer = 0




    def randomDir(self):
        dirfoo = Control()
        dirfoo['at'] = np.random.randn(self.Tsize, self.npt, self.dim)
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        else:
            dirfoo['Afft'] = None
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = self.state['xt'][t, :, :]
            gg = g1['at'][t, :, :]
            u = self.options['KparDiff'].applyK(z, gg)
            if self.affineDim > 0:
                uu = g1['Afft'][t]
            else:
                uu = 0
            ll = 0
            for gr in g2:
                ggOld = gr['at'][t, :, :]
                res[ll]  = res[ll] + (ggOld * u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['Afft'][t]).sum() * self.coeffAff
                ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            u = g1['at'][t, :, :]
            if self.affineDim > 0:
                uu = g1['Afft'][t]
            else:
                uu = 0
            ll = 0
            for gr in g2:
                ggOld = gr['at'][t, :, :]
                res[ll]  += (ggOld*u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['Afft'][t]).sum()
                ll = ll + 1
        return res


    def acceptVarTry(self):
        self.obj = self.objTry
        self.objDef = self.objTryDef
        self.objData = self.objTryData
        self.control = deepcopy(self.controlTry)

    def endOfIteration(self, endP=False):
        self.iter += 1
        if self.options['testGradient']:
            self.testEndpointGradient()

        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0 or endP) :
            logging.info('Saving Points...')
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True)

            self.fvDef.vertices = np.copy(np.squeeze(self.state['xt'][-1, :, :]))
            dim2 = self.dim**2
            A = self.affB.getTransforms(self.control['Afft'])
            st  = evol.landmarkDirectEvolutionEuler(self.x0, self.control['at'], self.options['KparDiff'], affine=A,
                                                    options={'withJacobian':True})
            xt = st['xt']
            Jt = st['Jt']
            if self.options['affine']=='euclidean' or self.options['affine']=='translation':
                X = self.affB.integrateFlow(self.control['Afft'])
                displ = np.zeros(self.x0.shape[0])
                dt = 1.0 /self.Tsize
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yyt = np.dot(self.state['xt'][t,...] - X[1][t, ...], U.T)
                    f = np.copy(yyt)
                    pointSets.savelmk(f, self.outputDir + '/' + self.options['saveFile'] + 'Corrected' + str(t) + '.lmk')
                f = deepcopy(self.fv1)
                yyt = np.dot(f.vertices - X[1][-1, ...], U.T)
                f = np.copy(yyt)
                pointSets.savePoints(self.outputDir + '/TargetCorrected.vtk', f)
            for kk in range(self.Tsize+1):
                fvDef = self.createObject(np.squeeze(xt[kk, :, :]), other=self.fv0.weights)
                fvDef.save(self.outputDir + '/' + self.options['saveFile'] + str(kk) + '.vtk')
        obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True)
        self.fvDef.vertices = np.copy(np.squeeze(self.state['xt'][-1, :, :]))

        if self.pkBuffer > 10:
            self.resetPK()


    def optimizeMatching(self):
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        if self.gradEps < 0:
            self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        logging.info('Gradient lower bound: %f' %(self.gradEps))
        self.coeffAff = self.coeffAff1

        if self.options['algorithm'] == 'cg':
            cg.cg(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=0.1)
        elif self.options['algorithm'] == 'bfgs':
            bfgs.bfgs(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=1.,
                      Wolfe=self.options['Wolfe'], memory=50)




