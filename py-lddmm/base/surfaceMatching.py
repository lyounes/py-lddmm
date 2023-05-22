import time
from copy import deepcopy
import numpy as np
import numpy.linalg as la
import logging
import h5py
from . import conjugateGradient as cg, bfgs, sgd
from . surfaces import Surface, vtkFields
from .surface_distances import currentNorm0, currentNormGradient, currentMagnitude, currentMagnitudeGradient, currentNormDef
from .surface_distances import measureNorm0, measureNormGradient, measureNormDef, varifoldNormGradient, varifoldNormDef, varifoldNorm0
from .surface_distances import measureNormPS0, measureNormPSDef, measureNormPSGradient, L2NormGradient, L2Norm0, L2Norm
from .surface_distances import normGrad, elasticNorm, diffNormGrad, diffElasticNorm
from . pointSets import PointSet, saveTrajectories, savePoints
import pointset_distances as psd
from .pointSetMatching import PointSetMatching
from .affineBasis import getExponential, gradExponential
from . import pointEvolution as evol
from . import pointEvolutionSemiReduced as evolSR
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import default_rng
rng = default_rng()


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class SurfaceMatchingParam(matchingParam.MatchingParam):
#     def __init__(self, timeStep = .1, algorithm='cg', Wolfe=True, KparDiff = None, KparDist = None,
#                  sigmaError = 1.0, errorType = 'measure', vfun = None, internalCost = None):
#         super().__init__(timeStep=timeStep, algorithm = algorithm, Wolfe=Wolfe,
#                          KparDiff = KparDiff, KparDist = KparDist, sigmaError=sigmaError,
#                          errorType = errorType, vfun=vfun)
#         self.sigmaError = sigmaError
#         self.internalCost = internalCost

class Control(dict):
    def __init__(self):
        super().__init__()
        self['at'] =None
        self['Afft'] = None
        self['x0'] = None
        self['ct'] = None

class State(dict):
    def __init__(self):
        super().__init__()
        self['xt'] = None
        self['yt'] = None
        self['Jt'] = None
        self['x0'] = None



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
class SurfaceMatching(PointSetMatching):
    def __init__(self, Template=None, Target=None, options=None):
        if Target is not None and (issubclass(type(Target), PointSet) or
                                   (type(Target) in (tuple, list) and issubclass(type(Target[0]), PointSet))):
            if options is None:
                options = {'errorType':'PointSet'}
            else:
                options['errorType'] = 'PointSet'
        super().__init__(Template, Target, options)
        self.setDotProduct(self.options['unreduced'])
        if self.options['algorithm'] == 'sgd':
            self.sgd = True
            self.unreduced = True
        else:
            self.sgd = False
            self.unreduced = self.options['unreduced']


        self.options['unreducedWeight'] *=  1000.0 / self.fv0.vertices.shape[0]
        #self.unreducedWeight = self.options['unreducedWeight']

        #if self.unreduced:
        if self.options['algorithm'] == 'sgd' or not self.options['reweightCells']:
            self.ds = 1.
        else:
            self.ds = self.fv0.surfArea() /  self.fv0.vertices.shape[0]

        if self.options['algorithm'] == 'sgd':
            self.set_sgd()


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['Landmarks'] = None
        options['reweightCells'] = False
        options['unreducedResetRate'] = -1
        options['fidelityWeight'] = 0.
        return options


    def set_passenger(self, passenger):
        self.passenger = passenger
        if isinstance(self.passenger, Surface):
            self.passenger_points = self.passenger.vertices
        elif self.passenger is not None:
            self.passenger_points = self.passenger
        else:
            self.passenger_points = None
        self.passengerDef = deepcopy(self.passenger)



    def set_parameters(self):
        super().set_parameters()
        self.lineSearch = "Weak_Wolfe"
        self.randomInit = False
        self.iter = 0
        self.reset = True
        self.options['epsInit'] /= self.fv0.vertices.shape[0]

        if self.options['internalCost'] == 'h1':
            self.internalCost = normGrad
            self.internalCostGrad = diffNormGrad
        elif self.options['internalCost'] == 'elastic':
            self.internalCost = elasticNorm
            self.internalCostGrad = diffElasticNorm
        else:
            if self.options['internalCost'] is not None:
                logging.info(f"unknown {self.options['internalCost']:.04f}")
            self.internalCost = None

        self.unreducedResetRate = self.options['unreducedResetRate']
        self.fidelityWeight = self.options['fidelityWeight']
        self.obj_unreduced_save = np.inf


    def set_sgd(self, control=100, template=100, target=100):
        self.weightSubset = 0.
        self.sgdEpsInit = 1e-4

        self.sgdNormalization = 'sdev'
        self.sgdBurnIn = 10000
        self.sgdMeanSelectControl = control
        self.sgdMeanSelectTemplate = template
        self.sgdMeanSelectTarget = target
        self.probSelectControl = min(1.0, self.sgdMeanSelectControl / self.fv0.vertices.shape[0])
        self.probSelectFaceTemplate = min(1.0, self.sgdMeanSelectTemplate / self.fv0.faces.shape[0])
        self.probSelectFaceTarget = min(1.0, self.sgdMeanSelectTarget / self.fv1.faces.shape[0])
        self.probSelectVertexTemplate = np.ones(self.fv0.vertices.shape[0])
        nf = np.zeros(self.fv0.vertices.shape[0])
        for k in range(self.fv0.faces.shape[0]):
            for j in range(3):
                self.probSelectVertexTemplate[self.fv0.faces[k,j]] *= \
                    1 - self.sgdMeanSelectTemplate/(self.fv0.faces.shape[0] - nf[self.fv0.faces[k,j]])
                nf[self.fv0.faces[k,j]] += 1

        self.probSelectVertexTemplate = 1 - self.probSelectVertexTemplate
        self.stateSubset = None


    def set_template_and_target(self, Template, Target, misc=None):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = Surface(surf=Template)

        if self.options['errorType'] != 'currentMagnitude':
            if Target is None:
                logging.error('Please provide a target surface')
                return
            else:
                if self.options['errorType'] == 'L2Norm':
                    self.fv1 = Surface()
                    self.fv1.readFromImage(Target)
                elif self.options['errorType'] == 'PointSet':
                    self.fv1 = PointSet(data=Target)
                else:
                    self.fv1 = Surface(surf=Target)
        else:
            self.fv1 = None
        self.fvInit = Surface(surf=self.fv0)
        self.fix_orientation()
        if misc is not None and 'subsampleTargetSize' in misc and misc['subsampleTargetSize'] > 0:
            self.fvInit.Simplify(misc['subsampleTargetSize'])
            logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        if self.fv1:
            self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def set_landmarks(self, landmarks):
        if landmarks is None:
            self.match_landmarks = False
            self.tmpl_lmk = None
            self.targ_lmk = None
            self.def_lmk = None
            self.wlmk = 0
            return

        self.match_landmarks = True
        tmpl_lmk, targ_lmk, self.wlmk = landmarks
        self.tmpl_lmk = PointSet(data=tmpl_lmk)
        self.targ_lmk = PointSet(data=targ_lmk)

    def fix_orientation(self, fv1=None):
        if fv1 is None:
            fv1 = self.fv1
        if issubclass(type(fv1), Surface):
            self.fv0.getEdges()
            fv1.getEdges()
            self.closed = self.fv0.bdry.max() == 0 and fv1.bdry.max() == 0
            if self.closed:
                v0 = self.fv0.surfVolume()
                if self.options['errorType'] == 'L2Norm' and v0 < 0:
                    self.fv0.flipFaces()
                    v0 = -v0
                v1 = fv1.surfVolume()
                if v0*v1 < 0:
                    fv1.flipFaces()
            if self.closed:
                z= self.fvInit.surfVolume()
                if z < 0:
                    self.fv0ori = -1
                else:
                    self.fv0ori = 1

                z= fv1.surfVolume()
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
        #self.fv0Fine = Surface(surf=self.fv0)
        logging.info('orientation: {0:d}'.format(self.fv0ori))


    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.nvert = self.fvInit.vertices.shape[0]
        if self.match_landmarks:
            self.control['x0'] = np.concatenate((self.fvInit.vertices, self.tmpl_lmk.points), axis=0)
            self.nlmk = self.tmpl_lmk.points.shape[0]
        else:
            self.control['x0'] = np.copy(self.fvInit.vertices)
            self.nlmk = 0
        if self.options['symmetric']:
            self.control['x0'] = np.copy(self.control['x0'])
            self.controlTry['x0'] = np.copy(self.control['x0'])
        self.fvDef = Surface(surf=self.fvInit)
        if self.match_landmarks:
            self.def_lmk = PointSet(data=self.tmpl_lmk)
        self.npt = self.control['x0'].shape[0]

        self.control['at'] = np.zeros([self.Tsize, self.control['x0'].shape[0], self.control['x0'].shape[1]])
        if self.randomInit:
            self.control['at'] = np.random.normal(0, 1, self.control['at'].shape)
        self.controlTry['at'] = np.zeros([self.Tsize, self.control['x0'].shape[0], self.control['x0'].shape[1]])


        if self.options['algorithm'] == 'sgd':
            self.SGDSelectionPts = [None, None]
            # self.SGDSelectionCost = [None, None]

        if self.affineDim > 0:
            self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        else:
            self.control['Afft'] = None
            self.controlTry['Afft'] = None
        self.state['xt'] = np.tile(self.control['x0'], [self.Tsize+1, 1, 1])
        self.stateTry = State()
        self.stateTry['xt'] = np.copy(self.state['xt'])
        if self.options['unreduced']:
            self.control['ct'] = np.tile(self.control['x0'], [self.Tsize, 1, 1])
            if self.randomInit:
                self.control['ct'] += np.random.normal(0, 1, self.control['ct'].shape)
            self.controlTry['ct'] = np.copy(self.control['ct'])
        else:
            self.control['ct'] = None
            self.controlTry['ct'] = None



        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.saveFileList = []
        for kk in range(self.Tsize+1):
            self.saveFileList.append(self.options['saveFile'] + f'{kk:03d}')


    def initial_plot(self):
        fig = plt.figure(3)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r', setLim=False)
        if self.fv1:
            lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b', setLim=False)
        else:
            lim0 = lim1
        ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
        ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
        ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
        if self.match_landmarks:
            ax.scatter3D(self.tmpl_lmk.points[:,0], self.tmpl_lmk.points[:,1], self.tmpl_lmk.points[:,2], color='r')
            ax.scatter3D(self.targ_lmk.points[:, 0], self.targ_lmk.points[:, 1], self.targ_lmk.points[:, 2], color='b')
        fig.canvas.flush_events()

    def set_fun(self, errorType, vfun = None):
        self.options['errorType'] = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj0 = partial(currentNorm0, KparDist=self.options['KparDist'], weight=1.)
            self.fun_obj = partial(currentNormDef, KparDist=self.options['KparDist'], weight=1.)
            self.fun_objGrad = partial(currentNormGradient, KparDist=self.options['KparDist'], weight=1.)
        elif errorType == 'currentMagnitude':
            #print('Running Current Matching')
            self.fun_obj0 = lambda fv1 : 0
            self.fun_obj = partial(currentMagnitude, KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(currentMagnitudeGradient, KparDist=self.options['KparDist'])
            # self.fun_obj0 = curves.currentNorm0
            # self.fun_obj = curves.currentNormDef
            # self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            #print('Running Measure Matching')
            self.fun_obj0 = partial(measureNorm0, KparDist=self.options['KparDist'])
            self.fun_obj = partial(measureNormDef,KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(measureNormGradient,KparDist=self.options['KparDist'])
        elif errorType=='varifold':
            self.fun_obj0 = partial(varifoldNorm0, KparDist=self.options['KparDist'], fun=vfun,
                                    dtype=self.options['KparDist'].pk_dtype)
            self.fun_obj = partial(varifoldNormDef, KparDist=self.options['KparDist'], fun=vfun,
                                   dtype=self.options['KparDist'].pk_dtype)
            self.fun_objGrad = partial(varifoldNormGradient, KparDist=self.options['KparDist'], fun=vfun,
                                       dtype=self.options['KparDist'].pk_dtype)
        elif errorType == 'L2Norm':
            self.fun_obj0 = None
            self.fun_obj = None
            self.fun_objGrad = None
        elif errorType == 'PointSet':
            self.fun_obj0 = partial(measureNormPS0, KparDist=self.options['KparDist'])
            self.fun_obj = partial(measureNormPSDef, KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(measureNormPSGradient, KparDist=self.options['KparDist'])
        else:
            logging.info(f"Unknown error Type:  {self.options['errorType']}")

        if self.match_landmarks:
            self.lmk_obj0 = psd.L2Norm0
            self.lmk_obj = psd.L2NormDef
            self.lmk_objGrad = psd.L2NormGradient
        else:
            self.lmk_obj0 = None
            self.lmk_obj = None
            self.lmk_objGrad = None


    def addSurfaceToPlot(self, fv1, ax, ec = 'b', fc = 'r', al=.5, lw=1, setLim=False):
        return fv1.addToPlot(ax, ec = ec, fc = fc, al=al, lw=lw)


    def dataTerm(self, _fvDef, var = None):
        #fv1 = None, _fvInit = None, _lmk_def = None, lmk1 = None
        if var is not None and 'fv1' in var:
            fv1 = var['fv1']
        else:
            fv1 = self.fv1

        if self.options['errorType'] == 'L2Norm':
            obj = L2Norm(_fvDef, fv1.vfld) / (self.options['sigmaError'] ** 2)
        else:
            obj = self.fun_obj(_fvDef, fv1) / (self.options['sigmaError']**2)
            if 'fvInit' in var:
                obj += self.fun_obj(var['fvInit'], self.fv0) / (self.options['sigmaError']**2)

        if self.match_landmarks:
            if var is None or not 'lmk_def' in var:
                logging.error('Data term: Missing deformed landmarks')
            if var is not None and 'lmk1' in var:
                lmk1 = var['lmk1']
            else:
                lmk1 = self.targ_lmk.points
            obj += self.wlmk * self.lmk_obj(var['lmk_def'].points, lmk1)
        #print 'dataterm = ', obj + self.obj0
        return obj

    def objectiveFunDef(self, control, var = None, withTrajectory = True, withJacobian=False):
        if var is None or 'fv0' not in var:
            fv0 = self.fv0
        else:
            fv0 = var['fv0']
        if self.match_landmarks:
            x0 = np.concatenate((fv0.vertices, self.tmpl_lmk.points), axis=0)
        else:
            x0 = fv0.vertices
        if var is None or 'kernel' not in var:
            kernel = self.options['KparDiff']
        else:
            kernel = var['kernel']
        #print 'x0 fun def', x0.sum()
        if var is None or 'regWeight' not in var:
            regWeight = self.options['regWeight']
        else:
            regWeight = var['regWeight']

        if np.isscalar(regWeight):
            regWeight_ = np.zeros(self.Tsize)
            regWeight_[:] = regWeight
        else:
            regWeight_ = regWeight

        st = State()
        timeStep = 1.0/self.Tsize
        if 'Afft' in control:
            Afft = control['Afft']
            A = self.affB.getTransforms(Afft)
        else:
            A = None

        if self.unreduced:
            ct = control['ct']
            at = control['at']
        else:
            ct = None
            at = control['at']

        if withJacobian:
            if self.unreduced:
                xt,Jt  = evolSR.landmarkSemiReducedEvolutionEuler(x0, ct, at*self.ds, kernel,
                                                                  fidelityWeight=self.fidelityWeight,
                                                                  affine=A, withJacobian=True)
            else:
                xt,Jt  = evol.landmarkDirectEvolutionEuler(x0, at*self.ds, kernel, affine=A, withJacobian=True)
            st['xt'] = xt
            st['Jt'] = Jt
        else:
            if self.unreduced:
                xt = evolSR.landmarkSemiReducedEvolutionEuler(x0, ct, at*self.ds, kernel, affine=A,
                                                              fidelityWeight=self.fidelityWeight)
            else:
                xt  = evol.landmarkDirectEvolutionEuler(x0, at*self.ds, kernel, affine=A)
            st['xt'] = xt
        #print xt[-1, :, :]
        #print obj
        obj=0
        obj1 = 0
        obj2 = 0
        obj3 = 0
        foo = Surface(surf=fv0)
        for t in range(self.Tsize):
            z = xt[t, :, :]
            a = at[t, :, :]
            if self.unreduced:
                c = ct[t,:,:]
            else:
                c = None

            if self.unreduced:
                ca = kernel.applyK(c,a)
                ra = kernel.applyK(c, a, firstVar=z)
                obj += regWeight_[t] * timeStep * (a * ca).sum() * self.ds**2
                obj3 += self.options['unreducedWeight'] * timeStep * ((c - z)**2).sum()
            else:
                ra = kernel.applyK(z, a)
                obj += regWeight_[t]*timeStep*(a*ra).sum() * self.ds**2
            if hasattr(self, 'v'):
                self.v[t, :] = ra * self.ds
            if self.internalCost:
                foo.updateVertices(z[:self.nvert, :])
                obj1 += self.options['internalWeight']*self.internalCost(foo, ra*self.ds)*timeStep

            if self.affineDim > 0:
                obj2 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if self.options['mode'] == 'debug':
            logging.info(f'LDDMM: {obj:.4f}, unreduced penalty: {obj3:.4f}, internal cost: {obj1:.4f}, Affine cost: {obj2:.4f}')
        obj += obj1 + obj2 + obj3

        if withTrajectory or withJacobian:
            return obj, st
        else:
            return obj


    def objectiveFun(self):
        if self.obj is None:
            if self.options['errorType'] == 'L2Norm':
                self.obj0 = L2Norm0(self.fv1) / (self.options['sigmaError'] ** 2)
            else:
                self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError']**2)
            if self.options['symmetric']:
                self.obj0 += self.fun_obj0(self.fv0) / (self.options['sigmaError']**2)
            if self.match_landmarks:
                self.obj0 += self.wlmk * self.lmk_obj0(self.targ_lmk) / (self.options['sigmaError']**2)
            # if self.unreduced:
            #     (self.obj, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
            # else:
            self.obj, self.state = self.objectiveFunDef(self.control, withTrajectory=True)
            #foo = Surface(surf=self.fvDef)
            self.fvDef.updateVertices(self.state['xt'][-1, :self.nvert, :])
            if self.match_landmarks:
                self.def_lmk.points = self.state['xt'][-1, self.nvert:, :]
            if self.options['symmetric']:
                self.fvInit.updateVertices(np.squeeze(self.control['x0'][:self.nvert, :]))
                self.obj += self.obj0 + self.dataTerm(self.fvDef, {'fvInit':self.fvInit, 'lmk_def':self.def_lmk})
            else:
                self.obj += self.obj0 + self.dataTerm(self.fvDef, {'lmk_def':self.def_lmk})
            #print self.obj0,  self.dataTerm(self.fvDef)

        return self.obj


    def objectiveFun_(self, control):
        if self.obj is None:
            if self.options['errorType'] == 'L2Norm':
                obj0 = L2Norm0(self.fv1) / (self.options['sigmaError'] ** 2)
            else:
                obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError']**2)
            if self.options['symmetric']:
                obj0 += self.fun_obj0(self.fv0) / (self.options['sigmaError']**2)
            if self.match_landmarks:
                obj0 += self.wlmk * self.lmk_obj0(self.targ_lmk) / (self.options['sigmaError']**2)
        else:
            obj0 = self.obj0
            # if self.unreduced:
            #     (self.obj, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
            # else:
        obj, state = self.objectiveFunDef(control, withTrajectory=True)
        #foo = Surface(surf=self.fvDef)
        fvDef = Surface(surf=self.fv0)
        fvDef.updateVertices(state['xt'][-1, :self.nvert, :])
        if self.match_landmarks:
            def_lmk = PointSet(self.def_lmk)
            def_lmk.points = state['xt'][-1, self.nvert:, :]
        else:
            def_lmk = None
        if self.options['symmetric']:
            fvInit = Surface(surf=self.fv0)
            fvInit.updateVertices(np.squeeze(control['x0'][:self.nvert, :]))
            obj += obj0 + self.dataTerm(fvDef, {'fvInit':fvInit, 'lmk_def':def_lmk})
        else:
            obj += obj0 + self.dataTerm(fvDef, {'lmk_def':def_lmk})
        #print self.obj0,  self.dataTerm(self.fvDef)

        return obj, state

    # def Direction(self):
    #     return Direction()

    def update(self, dr, eps):
        for k in dr.keys():
            if dr[k] is not None:
                self.control[k] -= dr[k]

    def updateTry(self, dr, eps, objRef=None):
        objTry = self.obj0
        controlTry = Control()
        for k in dr.keys():
            if dr[k] is not None:
                controlTry[k] = self.control[k] - eps * dr[k]


        fv0 = Surface(surf=self.fv0)
        if self.options['symmetric']:
            fv0.updateVertices(controlTry['x0'])

        obj_, stateTry = self.objectiveFunDef(controlTry, var = {'fv0': fv0}, withTrajectory=True)
        objTry += obj_

        ff = Surface(surf=self.fvDef)
        ff.updateVertices(stateTry['xt'][-1, :self.nvert, :])
        if self.match_landmarks:
            pp = PointSet(data=self.def_lmk)
            pp.updatePoints(np.squeeze(stateTry['xt'][-1, self.nvert:, :]))
        else:
            pp = None
        if self.options['symmetric']:
            ffI = Surface(surf=self.fvInit)
            ffI.updateVertices(controlTry['x0'])
            objTry += self.dataTerm(ff, {'fvInit': ffI, 'lmk_def':pp})
        else:
            objTry += self.dataTerm(ff, {'lmk_def':pp})
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = controlTry
            self.stateTry = stateTry
            self.objTry = objTry

        return objTry


    def testEndpointGradient(self):
        dff = np.random.normal(size=self.fvDef.vertices.shape)
        if self.match_landmarks:
            dpp = np.random.normal(size=self.def_lmk.points.shape)
            dall = np.concatenate((dff, dpp), axis=0)
        else:
            dall = dff
            dpp = None
        c = []
        eps0 = 1e-6
        for eps in [-eps0, eps0]:
            ff = Surface(surf=self.fvDef)
            ff.updateVertices(ff.vertices+eps*dff)
            if self.match_landmarks:
                pp = PointSet(data=self.def_lmk)
                pp.updatePoints(pp.points + eps * dpp)
            else:
                pp = None
            c.append(self.dataTerm(ff, {'lmk_def':pp}))
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c[1]-c[0])/(2*eps), (grd*dall).sum()) )



    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
            endPoint_lmk = self.def_lmk
        elif self.match_landmarks:
            endPoint_lmk = endPoint[1]
            endPoint = endPoint[0]
        else:
            endPoint_lmk = None
        if self.options['errorType'] == 'L2Norm':
            px = L2NormGradient(endPoint, self.fv1.vfld)
        else:
            if self.fv1:
                px = self.fun_objGrad(endPoint, self.fv1)
            else:
                px = self.fun_objGrad(endPoint)
        if self.match_landmarks:
            pxl = self.wlmk*self.lmk_objGrad(endPoint_lmk.points, self.targ_lmk.points)
            px = np.concatenate((px, pxl), axis=0)
        return px / self.options['sigmaError']**2

    def initPointGradient(self):
        px = self.fun_objGrad(self.fvInit, self.fv0, self.options['KparDist'])
        return px / self.options['sigmaError']**2
    
    
    def hamiltonianCovector(self, px1, KparDiff, regWeight, fv0 = None, control = None):
        if fv0 is None:
            fv0 = self.fvInit
        if control is None:
            control = self.control
            current_at = True
            if self.varCounter == self.trajCounter:
                computeTraj = False
            else:
                computeTraj = True
        else:
            current_at = False
            computeTraj = True
        if self.match_landmarks:
            x0 = np.concatenate((fv0.vertices, self.tmpl_lmk.points), axis=0)
        else:
            x0 = fv0.vertices

        if self.unreduced and self.control['ct'].shape[1] == x0.shape[0]:
            fidelityTerm = True
        else:
            fidelityTerm = False

        N = x0.shape[0]
        dim = x0.shape[1]
        T = control['at'].shape[0]
        timeStep = 1.0/T
        affine = self.affB.getTransforms(control['Afft'])
        if computeTraj:
            if self.unreduced:
                xt = evolSR.landmarkSemiReducedEvolutionEuler(x0, control['ct'], control['at']*self.ds,
                                                            KparDiff, fidelityWeight=self.fidelityWeight, affine=affine)
            else:
                xt = evol.landmarkDirectEvolutionEuler(x0, control['at']*self.ds, KparDiff, affine=affine)
            if current_at:
                self.trajCounter = self.varCounter
                self.state['xt'] = xt
        else:
            xt = self.state['xt']

        if affine is not None:
            A0 = affine[0]
            A = np.zeros([T,dim,dim])
            for t in range(A0.shape[0]):
                A[t,:,:] = getExponential(timeStep*A0[t])
        else:
            A = None

        pxt = np.zeros([T+1, N, dim])
        pxt[T, :, :] = px1
        # if self.unreduced:
        #     pxt[T-1, :, :] -= self.unreducedWeight * ((xt[T, :, :] + xt[T-1, :, :])/2 - ct[T-1, :, :])*timeStep
        foo = Surface(surf=fv0)
        for t in range(T):
            px = pxt[T-t, :, :]
            z = xt[T-t-1, :, :]
            a = control['at'][T-t-1, :, :]
            if self.unreduced:
                c = np.squeeze(control['ct'][T - t-1, :, :])
                v = KparDiff.applyK(c,a, firstVar=z)*self.ds
            else:
                c = None
                v = KparDiff.applyK(z,a)*self.ds

            foo.updateVertices(z)
            if self.internalCost:
                grd = self.internalCostGrad(foo, v)
                Lv = grd[0]
                DLv = self.options['internalWeight']*grd[1]
                if self.unreduced:
                    zpx = KparDiff.applyDiffKT(c, px - self.options['internalWeight']*Lv, a*self.ds,
                                               lddmm=False, firstVar=z) - DLv - 2*self.options['unreducedWeight'] * (z-c)
                else:
                    zpx = KparDiff.applyDiffKT(z, px, a*self.ds, regweight=self.options['regWeight'], lddmm=True,
                                               extra_term=-self.options['internalWeight'] * Lv) - DLv
            else:
                if self.unreduced:
                    zpx = KparDiff.applyDiffKT(c, px, a*self.ds, lddmm=False, firstVar=z) \
                        - 2*self.options['unreducedWeight'] * (z-c)
                else:
                    zpx = KparDiff.applyDiffKT(z, px, a*self.ds, regweight=self.options['regWeight'], lddmm=True)

            if self.unreduced and fidelityTerm:
                zpx -= self.fidelityWeight * px

            if affine is not None:
                pxt[T-t-1, :, :] = px @ A[T-t-1, :, :] + timeStep * zpx
            else:
                pxt[T-t-1, :, :] = px + timeStep * zpx
        return pxt, xt



    def hamiltonianGradient(self, px1, kernel = None, regWeight=None, fv0=None, control=None):
        if regWeight is None:
            regWeight = self.options['regWeight']
        if fv0 is None:
            fv0 = self.fvInit
        x0 = fv0.vertices
        if control is None:
            control = self.control

        if kernel is None:
            kernel  = self.options['KparDiff']

        foo = Surface(surf=fv0)
        foo.updateVertices(x0)

        if self.unreduced and self.control['ct'].shape[1] == x0.shape[0]:
            fidelityTerm = True
        else:
            fidelityTerm = False


        (pxt, xt) = self.hamiltonianCovector(px1, kernel, regWeight, fv0=foo, control = control)

        dat = np.zeros(control['at'].shape)
        if self.unreduced:
            dct = np.zeros(control['ct'].shape)
        else:
            dct = None
        timeStep = 1.0/control['at'].shape[0]
        foo = Surface(surf=fv0)
        nvert = foo.vertices.shape[0]
        affine = self.affB.getTransforms(control['Afft'])
        if affine is not None:
            A = affine[0]
            dA = np.zeros(affine[0].shape)
            db = np.zeros(affine[1].shape)
        for t in range(control['at'].shape[0]):
            z = xt[t,:,:]
            a = control['at'][t, :, :]
            if self.unreduced:
                c = control['ct'][t,:,:]
            else:
                c = None
            px = pxt[t+1, :, :]

            if not self.affineOnly:
                if self.unreduced:
                    dat[t, :, :] = 2 * regWeight * kernel.applyK(c, a) * self.ds**2 - kernel.applyK(z, px, firstVar=c) * self.ds
                    #if k > 0:
                    dct[t, :, :] = 2 * regWeight * kernel.applyDiffKT(c, a, a) * self.ds**2 \
                                   - kernel.applyDiffKT(z, a, px, firstVar=c) * self.ds \
                                    + 2 * self.options['unreducedWeight'] * (c-z)
                    if self.unreduced and fidelityTerm:
                        dct[t, :, :] -= self.fidelityWeight * px
                    v = kernel.applyK(c, a, firstVar=z)*self.ds
                else:
                    dat[t, :, :] = 2 * regWeight * a * self.ds**2 - px * self.ds
                    v = kernel.applyK(z,a)*self.ds
                if self.internalCost:
                    foo.updateVertices(z[:nvert, :])
                    Lv = self.internalCostGrad(foo, v, variables='phi')
                    if self.unreduced:
                        dat[t, :, :] += self.options['internalWeight'] * kernel.applyK(z, Lv, firstVar=c) * self.ds
                        dct[t, :, :] += self.options['internalWeight'] * kernel.applyDiffKT(z, a, Lv, firstVar=c)*self.ds
                    else:
                        dat[t, :, :] += self.options['internalWeight'] * Lv * self.ds

                if not self.unreduced and self.euclideanGradient:
                    dat[t, :, :] = kernel.applyK(z, dat[t, :, :])

            if affine is not None:
                dA[t] = gradExponential(A[t]*timeStep, px, z)
                db[t] = px.sum(axis=0)

        res = {}
        if self.unreduced:
            if self.options['mode'] == 'debug':
                logging.info(f'gradient {np.fabs(dct).max()} {np.fabs(dat).max()}')
            res['dct'] = dct
        res['dat'] = dat
        res['xt'] = xt
        res['pxt'] = pxt
        if affine is not None:
            res['dA'] = dA
            res['db'] = db
        return res


    def endPointGradientSGD(self):
        if self.sgdMeanSelectTemplate >= self.fv0.faces.shape[0]:
            I0_ = np.arange(self.fv0.faces.shape[0])
            p0 = 1.
            sqp0 = 1.
        else:
            I0_ = rng.choice(self.fv0.faces.shape[0], self.sgdMeanSelectTemplate, replace=False)
            p0 = self.sgdMeanSelectTemplate / self.fv0.faces.shape[0]
            sqp0 = np.sqrt(self.sgdMeanSelectTemplate * (self.sgdMeanSelectTemplate - 1)
                           / (self.fv0.faces.shape[0] * (self.fv0.faces.shape[0] - 1)))

        if self.sgdMeanSelectTarget > self.fv1.faces.shape[0]:
            I1_ = np.arange(self.fv1.faces.shape[0])
            p1 = p0 / sqp0
        else:
            I1_ = rng.choice(self.fv1.faces.shape[0], self.sgdMeanSelectTarget, replace=False)
            p1 = (self.sgdMeanSelectTarget / self.fv1.faces.shape[0]) * p0 / sqp0

        select0 = np.zeros(self.fv0.faces.shape[0], dtype=bool)
        select0[I0_] = True
        fv0, I0 = self.fv0.select_faces(select0)
        self.stateSubset = I0
        xt = evolSR.landmarkSemiReducedEvolutionEuler(fv0.vertices, self.control['ct'], self.control['at'],
                                                      self.options['KparDiff'],
                                                      fidelityWeight=self.fidelityWeight, affine=self.control['Afft'])
        endPoint = Surface(surf=fv0)
        endPoint.updateVertices(xt[-1, :, :])
        endPoint.face_weights /= sqp0
        # endPoint.updateWeights(endPoint.weights / sqp0)

        select1 = np.zeros(self.fv1.faces.shape[0], dtype=bool)
        select1[I1_] = True
        fv1, I1 = self.fv1.select_faces(select1)
        # endPoint.saveVTK('foo.vtk')
        fv1.face_weights /= p1
        #        fv1.updateWeights(fv1.weights / p1)
        # self.SGDSelectionCost = [I0, I1]

        if self.options['errorType'] == 'L2Norm':
            px_ = L2NormGradient(endPoint, self.fv1.vfld)
        else:
            px_ = self.fun_objGrad(endPoint, fv1)
            ## Correction for diagonal term
            if self.sgdMeanSelectTemplate < self.fv0.faces.shape[0]:
                s0 = (1/(sqp0**2) - 1/p0) #(1 / sqp0 - sqp0 / p0)  #  # (sqp0 **2 /p0-1) * p0/sqp0
                if self.options['errorType'] == 'varifold':
                    s1 = 2.
                else:
                    s1 = 1.

                pc = np.zeros(fv0.vertices.shape)
                xDef0 = endPoint.vertices[fv0.faces[:, 0], :]
                xDef1 = endPoint.vertices[fv0.faces[:, 1], :]
                xDef2 = endPoint.vertices[fv0.faces[:, 2], :]
                nu = np.cross(xDef1 - xDef0, xDef2 - xDef0)
                dz0 = np.cross(xDef1 - xDef2, nu)
                dz1 = np.cross(xDef2 - xDef0, nu)
                dz2 = np.cross(xDef0 - xDef1, nu)
                for k in range(fv0.faces.shape[0]):
                    pc[fv0.faces[k, 0], :] += dz0[k, :]
                    pc[fv0.faces[k, 1], :] += dz1[k, :]
                    pc[fv0.faces[k, 2], :] += dz2[k, :]
                px_ -= s1 * s0 * pc / 2

        self.state['xt'][:, I0, :] = xt
        return px_ / self.options['sigmaError'] ** 2, xt

    def checkSGDEndpointGradient(self):
        endPoint = Surface(surf=self.fv0)
        xt = evolSR.landmarkSemiReducedEvolutionEuler(self.fv0.vertices, self.control['ct'], self.control['at'],
                                                      self.options['KparDiff'],
                                                      fidelityWeight=self.fidelityWeight, affine=self.control['Afft'])
        endPoint.updateVertices(xt[-1, :, :])

        pxTrue = self.endPointGradient(endPoint=endPoint)
        px = np.zeros(pxTrue.shape)
        nsim = 25
        for k in range(nsim):
            px += self.endPointGradientSGD()[0]

        px /= nsim
        diff = ((px - pxTrue) ** 2).mean()
        logging.info(f'check SGD gradient: {diff:.4f}')

    def getGradientSGD(self, coeff=1.0):
        #self.checkSGDEndpointGradient()
        A = self.affB.getTransforms(self.control['Afft'])
        px1, xt = self.endPointGradientSGD()
        # I0 = self.SGDSelectionCost[0]
        #x0[self.stateSubset, :] = xt[-1, :, :]
        if self.sgdMeanSelectControl <= self.control['ct'].shape[1]:
            J0 = rng.choice(self.control['ct'].shape[1], self.sgdMeanSelectControl, replace=False)
            #J1 = rng.choice(self.ct.shape[1], self.sgdMeanSelectControl, replace=False)
        else:
            J0 = np.arange(self.control['ct'].shape[1])
            #J1 = np.arange(self.ct.shape[1])
        foo = evolSR.landmarkSemiReducedHamiltonianGradient(self.control['x0'], self.control['ct'], self.control['at'],
                                                            -px1, self.options['KparDiff'],
                                                            self.options['regWeight'], getCovector = True, affine = A,
                                                            fidelityWeight=self.fidelityWeight,
                                                            weightSubset=self.options['unreducedWeight'],
                                                            controlSubset = J0, stateSubset=self.stateSubset,
                                                            controlProb=self.probSelectControl,
                                                            stateProb=self.probSelectVertexTemplate,
                                                            forwardTraj=xt)
        dim2 = self.dim**2
        grd = Control()
        grd['ct'] = foo[0] / (coeff*self.Tsize)
        grd['at'] = foo[1] / (coeff*self.Tsize)
        grd['x0'] = np.zeros((self.npt, self.dim))
        if self.affineDim > 0:
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            dA = foo[2]
            db = foo[3]
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*self.control['Afft']
            for t in range(self.Tsize):
               dAff = self.affineBasis.T @ np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])])
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] /= (self.coeffAff*coeff*self.Tsize)
        else:
            grd['Afft'] = None
        return grd



    def getGradient(self, coeff=1.0, update=None):
        if self.options['algorithm'] == 'sgd':
            return self.getGradientSGD(coeff=coeff)

        if update is None:
            control = self.control
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
        else:
            control = Control()
            for k in update[0].keys():
                if update[0][k] is not None:
                    control[k] = self.control[k] - update[1]*update[0][k]
            A = self.affB.getTransforms(control['Afft'])
            if self.unreduced:
                xt = evolSR.landmarkSemiReducedEvolutionEuler(self.control['x0'], control['ct'], control['at']*self.ds,
                                                            self.options['KparDiff'], affine=A,
                                                              fidelityWeight=self.fidelityWeight)
            else:
                xt = evol.landmarkDirectEvolutionEuler(self.control['x0'], control['at']*self.ds,
                                                       self.options['KparDiff'], affine=A)


            if self.match_landmarks:
                endPoint0 = Surface(surf=self.fv0)
                endPoint0.updateVertices(xt[-1, :self.nvert, :])
                endPoint1 = PointSet(data=xt[-1, self.nvert:,:])
                endPoint = (endPoint0, endPoint1)
            else:
                endPoint = Surface(surf=self.fv0)
                endPoint.updateVertices(xt[-1, :, :])


        px1 = -self.endPointGradient(endPoint=endPoint)
        dim2 = self.dim**2
        foo = self.hamiltonianGradient(px1, control=control)
        grd = Control()

        if self.unreduced:
            grd['ct'] = foo['dct']/(coeff*self.Tsize)
            grd['at'] = foo['dat'] / (coeff * self.Tsize)
        else:
            grd['at'] = foo['dat']/(coeff*self.Tsize)

        if self.affineDim > 0:
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*control['Afft']
            for t in range(self.Tsize):
                dAff = self.affineBasis.T @ np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])])
                grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            if not self.euclideanGradient:
                grd['Afft'] /= (self.coeffAff*coeff*self.Tsize)
            else:
                grd['Afft'] /= coeff * self.Tsize

        if self.options['symmetric']:
            grd['x0'] = (self.initPointGradient() - foo['pxt'][0,...])/(self.coeffInitx * coeff)
        else:
            grd['x0'] = np.zeros((self.npt, self.dim))
        return grd




    def randomDir(self):
        dirfoo = Control()
        if self.affineOnly:
            dirfoo['at'] = np.zeros((self.Tsize, self.npt, self.dim))
        else:
            dirfoo['at'] = np.random.randn(self.Tsize, self.npt, self.dim)
        if self.unreduced:
            dirfoo['ct'] = np.random.normal(0, 1, size=self.control['ct'].shape)
            dirfoo['ct'][0, :, :] = 0
        if self.options['symmetric']:
            dirfoo['x0'] = np.random.randn(self.npt, self.dim)
        else:
            dirfoo['x0'] = np.zeros((self.npt, self.dim))
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = np.squeeze(self.state['xt'][t, :, :])
            gg = np.squeeze(g1['at'][t, :, :])
            u = self.options['KparDiff'].applyK(z, gg)
            if self.affineDim > 0:
                uu = g1['Afft'][t]
            else:
                uu = 0
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr['at'][t, :, :])
                res[ll]  = res[ll] + (ggOld*u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['Afft'][t]).sum() * self.coeffAff
                ll = ll + 1

        if self.options['symmetric']:
            for ll,gr in enumerate(g2):
                res[ll] += (g1['x0'] * gr['x0']).sum() * self.coeffInitx

        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for k in g1.keys():
            if g1[k] is not None:
                for ll,gr in enumerate(g2):
                    res[ll] += (g1[k]*gr[k]).sum()
        return res


    def acceptVarTry(self):
        self.obj = self.objTry
        self.control = deepcopy(self.controlTry)
        self.state = deepcopy(self.stateTry)

    def saveCorrectedTarget(self, X0, X1):
        U = la.inv(X0[-1])
        f = Surface(surf=self.fv1)
        yyt = np.dot(f.vertices - X1[-1,...], U)
        f.updateVertices(yyt)
        f.saveVTK(self.outputDir + '/TargetCorrected.vtk')
        if self.match_landmarks:
            p = PointSet(data=self.targ_lmk)
            yyt = np.dot(p.points - X1[-1,...], U)
            p.updatePoints(yyt)
            p.saveVTK(self.outputDir + '/TargetLandmarkCorrected.vtk')


    def saveCorrectedEvolution(self, fv0, state, control, fileName='evolution'):
        Jacobian = state['Jt']
        f = Surface(surf=fv0)
        if self.match_landmarks:
            p = PointSet(data=self.tmpl_lmk)
        else:
            p = None
        X = self.affB.integrateFlow(control['Afft'])
        displ = np.zeros(state['xt'].shape[1])
        dt = 1.0 / self.Tsize
        fn = []
        if type(fileName) is str:
            for kk in range(self.Tsize + 1):
                fn.append(fileName + f'_corrected{kk:03d}')
        else:
            fn = fileName
        vt = None
        for t in range(self.Tsize + 1):
            U = la.inv(X[0][t])
            yyt = (state['xt'][t, ...] - X[1][t, ...]) @ U.T
            zt = (state['xt'][t, ...] - X[1][t, ...]) @ U.T
            if t < self.Tsize:
                atCorr = control['at'][t, ...] @ U.T
                vt = self.options['KparDiff'].applyK(yyt, atCorr, firstVar=zt)
            f.updateVertices(yyt[:self.nvert, :])
            if self.match_landmarks:
                p.updatePoints(yyt[self.nvert:, :])
                p.saveVTK(self.outputDir + '/' + fn[t] + '_lmk.vtk')
            vf = vtkFields()
            if Jacobian is not None:
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jacobian[t, :self.nvert, 0]))
            vf.scalars.append('displacement')
            vf.scalars.append(displ[:self.nvert])
            vf.vectors.append('velocity')
            vf.vectors.append(vt[:self.nvert, :])
            nu = self.fv0ori * f.computeVertexNormals()
            f.saveVTK2(self.outputDir + '/' + fn[t] + '.vtk', vf)
            displ += dt * (vt * nu).sum(axis=1)
        self.saveCorrectedTarget(X[0], X[1])

    def saveEvolution(self, fv0, state, passenger = None, fileName='evolution', velocity = None,
                      orientation= None, with_area_displacement=False):
        xt = state['xt']
        if 'Jt' in state:
            Jacobian = state['Jt']
        else:
            Jacobian = None
        if velocity is None:
            velocity = self.v
        if orientation is None:
            orientation = self.fv0ori
        fn = []
        if type(fileName) is str:
            for kk in range(self.Tsize + 1):
                fn.append(fileName + f'{kk:03d}')
        else:
            fn = fileName


        fvDef = Surface(surf=fv0)
        AV0 = fvDef.computeVertexArea()
        nu = orientation * fv0.computeVertexNormals()
        nvert = fv0.vertices.shape[0]
        npt = xt.shape[1]
        v = velocity[0, :nvert, :]
        displ = np.zeros(nvert)
        area_displ = np.zeros((self.Tsize + 1, npt))
        dt = 1.0 / self.Tsize
        for kk in range(self.Tsize + 1):
            fvDef.updateVertices(np.squeeze(xt[kk, :nvert, :]))
            AV = fvDef.computeVertexArea()
            AV = (AV[0] / AV0[0])
            vf = vtkFields()
            if Jacobian is not None:
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jacobian[kk, :nvert, 0]))
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(np.exp(Jacobian[kk, :nvert, 0]) / AV)
            vf.scalars.append('displacement')
            vf.scalars.append(displ)
            if kk < self.Tsize:
                nu = orientation * fvDef.computeVertexNormals()
                v = velocity[kk, :nvert, :]
                kkm = kk
            else:
                kkm = kk - 1
            vf.vectors.append('velocity')
            vf.vectors.append(velocity[kkm, :nvert])
            if with_area_displacement and kk > 0:
                area_displ[kk, :] = area_displ[kk - 1, :] + dt * ((AV + 1) * (v * nu).sum(axis=1))[np.newaxis, :]
            fvDef.saveVTK2(self.outputDir + '/' + fn[kk] + '.vtk', vf)
            displ += dt * (v * nu).sum(axis=1)
            if passenger is not None and passenger[0] is not None:
                if isinstance(passenger[0], Surface):
                    fvp = Surface(surf=passenger[0])
                    fvp.updateVertices(passenger[1][kk,...])
                    fvp.saveVTK(self.outputDir+'/'+fn[kk]+'_passenger.vtk')
                else:
                    savePoints(self.outputDir+'/'+fn[kk]+'_passenger.vtk', passenger[1][kk,...])
            if self.match_landmarks:
                pp = PointSet(data=xt[kk,nvert:,:])
                pp.saveVTK(self.outputDir+'/'+fn[kk]+'_lmk.vtk')

    def saveEPDiff(self, fv0, at, fileName='evolution'):
        if self.match_landmarks:
            x0 = np.concatenate((fv0.vertices, self.tmpl_lmk), axis=0)
        else:
            x0 = fv0.vertices
        xtEPDiff, atEPdiff = evol.landmarkEPDiff(at.shape[0], x0,
                                                 np.squeeze(at[0, :, :]), self.options['KparDiff'])
        fvDef = Surface(surf=fv0)
        nvert = fv0.vertices.shape[0]
        fvDef.updateVertices(np.squeeze(xtEPDiff[-1, :nvert, :]))
        fvDef.saveVTK(self.outputDir + '/' + fileName + 'EPDiff.vtk')
        return xtEPDiff, atEPdiff

    def updateEndPoint(self, xt):
        self.fvDef.updateVertices(np.squeeze(xt[-1, :self.nvert, :]))
        if self.match_landmarks:
            self.def_lmk.updatePoints(xt[-1, self.nvert:, :])

    def plotAtIteration(self):
        fig = plt.figure(4)
        # fig.clf()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
        lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
        ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
        ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
        ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
        if self.match_landmarks:
            ax.scatter3D(self.def_lmk.points[:,0], self.def_lmk.points[:,1], self.def_lmk.points[:,2], color='r')
            ax.scatter3D(self.targ_lmk.points[:, 0], self.targ_lmk.points[:, 1], self.targ_lmk.points[:, 2], color='b')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.5)

    def endOfIterationSGD(self, forceSave=False):
        if forceSave or self.iter % self.saveRate == 0:
            # self.xt = evol.landmarkSemiReducedEvolutionEuler(self.x0, self.ct, self.at, self.param.KparDiff,
            #                                                  affine=self.Afft)
            saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesSGDState.vtk',
                                       self.state['xt'])
            saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesSGDControl.vtk',
                                       self.control['ct'])
            #self.updateEndPoint(self.xt)
            #self.ct = np.copy(self.xt[:-1, :, :])
            self.saveEvolution(self.fv0, self.state)

        if self.unreducedResetRate > 0 and self.iter % self.unreducedResetRate == 0:
            dist = ((self.control['ct'] - self.state['xt'][-1, :, :])**2).sum(axis=-1).max()
            logging.info(f'Resetting trajectories: max distance = {dist:.4f}')
            self.control['ct'] = np.copy(self.state['xt'][:-1, :, :])
            # f.at = np.zeros(f.at.shape)
            self.controlTry['ct'] = np.copy(self.control['ct'])


    def startOfIteration(self):
        if self.options['algorithm'] != 'sgd':
            if self.reset:
                self.options['KparDiff'].pk_dtype = 'float64'
                self.options['KparDist'].pk_dtype = 'float64'

    def endOfIteration(self, forceSave=False):
        self.iter += 1
        if self.options['algorithm'] == 'sgd':
            self.endOfIterationSGD(forceSave=forceSave)
            return


        if self.options['testGradient']:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        dim2 = self.dim ** 2
        if self.affineDim > 0:
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.control['Afft'][t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2 + self.dim]
        else:
            A = None
        if forceSave or self.iter % self.saveRate == 0:
            logging.info('Saving surfaces...')
            if self.passenger_points is None:
                if self.unreduced:
                    xt, Jt = evolSR.landmarkSemiReducedEvolutionEuler(self.control['x0'], self.control['ct'],
                                                                      self.control['at']*self.ds,
                                                                      self.options['KparDiff'], affine=A,
                                                                      fidelityWeight=self.fidelityWeight,
                                                                      withJacobian=True)
                else:
                    xt, Jt = evol.landmarkDirectEvolutionEuler(self.control['x0'], self.control['at']*self.ds,
                                                                    self.options['KparDiff'],
                                                                    affine=A, withJacobian=True)
                yt = None
            else:
                if self.unreduced:
                    xt, yt, Jt = evolSR.landmarkSemiReducedEvolutionEuler(self.control['x0'], self.control['ct'],
                                                                         self.control['at']*self.ds,
                                                                         self.options['KparDiff'], affine=A,
                                                                          fidelityWeight=self.fidelityWeight,
                                                                         withPointSet=self.passenger_points,
                                                                         withJacobian=True)
                else:
                    xt, yt, Jt = evol.landmarkDirectEvolutionEuler(self.control['x0'], self.control['at']*self.ds,
                                                                        self.options['KparDiff'], affine=A,
                                                                        withPointSet=self.passenger_points,
                                                                        withJacobian=True)
                if isinstance(self.passenger, Surface):
                    self.passengerDef.updateVertices(yt[-1,...])
                else:
                    self.passengerDef = deepcopy(yt[-1,...])

            self.trajCounter = self.varCounter
            self.state['xt'] = xt
            self.state['Jt'] = Jt
            self.state['yt'] = yt

            if self.saveEPDiffTrajectories and not self.internalCost and self.affineDim <= 0:
                xtEPDiff, atEPdiff = self.saveEPDiff(self.fvInit, self.control['at']*self.ds, fileName=self.options['saveFile'])
                logging.info('EPDiff difference %f' % (np.fabs(self.control['xt'][-1, :, :] - xtEPDiff[-1, :, :]).sum()))

            if self.options['saveTrajectories']:
                saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curves.vtk', self.state['xt'])


            self.updateEndPoint(self.state['xt'])
            self.fvInit.updateVertices(self.control['x0'][:self.nvert, :])

            if self.options['affine'] == 'euclidean' or self.options['affine'] == 'translation':
                self.saveCorrectedEvolution(self.fvInit, self.state, self.control, fileName=self.saveFileList)
            self.saveEvolution(self.fvInit, self.state, fileName=self.saveFileList,
                               passenger = (self.passenger, yt))
            if self.unreduced:
                saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesSGDState.vtk',
                                           self.state['xt'])
                saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesSGDControl.vtk',
                                           self.control['ct'])
            self.saveHdf5(fileName=self.outputDir + '/output.h5')
        else:
            if self.varCounter != self.trajCounter:
                if self.unreduced:
                    self.state['xt'] = evolSR.landmarkSemiReducedEvolutionEuler(self.control['x0'], self.control['ct'],
                                                                              self.control['at']*self.ds,
                                                                              self.options['KparDiff'], affine=A,
                                                                                fidelityWeight=self.fidelityWeight)
                else:
                    self.state['xt'] = evol.landmarkDirectEvolutionEuler(self.control['x0'], self.control['at']*self.ds,
                                                                         self.options['KparDiff'], affine=A)
                self.trajCounter = self.varCounter
            self.updateEndPoint(self.state['xt'])
            if self.options['symmetric']:
                self.fvInit.updateVertices(self.control['x0'][:self.nvert, :])

        if (self.unreduced and self.unreducedResetRate > 0 and self.iter % self.unreducedResetRate == 0 and
            self.obj < self.obj_unreduced_save):
            self.obj_unreduced_save = self.obj
            dist0 = np.sqrt(((self.state['xt'][0, :, :] - self.state['xt'][-1, :, :])**2).sum(axis=-1).max())
            dist = np.sqrt(((self.control['ct'] - self.state['xt'][:-1, :, :])**2).sum(axis=-1).max())
            if dist > 0.05 * dist0:
                rho = 1.
                objDef, st = self.objectiveFun_(self.control)
                obj_ = 2*objDef+1
                control = deepcopy((self.control))
                d_ = 2*dist + 1
                control['ct'] = (1 - rho) * self.control['ct'] + rho * self.state['xt'][:-1, :, :]
                # while obj_ > 1.5*objDef and rho > 0.001:
                #     control['ct'] = (1-rho) * self.control['ct'] + rho * self.state['xt'][:-1, :, :]
                #     obj_, st = self.objectiveFun_(control)
                #     #d_ = np.sqrt(((st['xt'][-1, :, :] - self.state['xt'][-1, :, :]) ** 2).sum(axis=-1).max())
                #     #logging.info(f"d: {obj_:.4f}" )
                #     if obj_ > objDef:
                #         rho *= 0.5
                # rho = 1.
                if rho > 0.001:
                    logging.info(f'Resetting trajectories: max distance = {dist:.4f} trajectory distance = {dist0:.4f} rho = {rho:0.4f}')
                    self.control = deepcopy(control)
                    # f.at = np.zeros(f.at.shape)
                    self.controlTry['ct'] = np.copy(self.control['ct'])
                    # dist = np.sqrt(((self.control['ct'] - self.state['xt'][:-1, :, :])**2).sum(axis=-1).max())
                    # logging.info(f'max distance = {dist:.4f}')
                    self.reset = True
        if self.pplot:
            self.plotAtIteration()

        self.options['KparDiff'].pk_dtype = self.Kdiff_dtype
        self.options['KparDist'].pk_dtype = self.Kdist_dtype


    def saveHdf5(self, fileName):
        fout = h5py.File(fileName, 'w')
        LDDMMResult = fout.create_group('LDDMM Results')
        parameters = LDDMMResult.create_group('parameters')
        parameters.create_dataset('Time steps', data=self.Tsize)
        parameters.create_dataset('Deformation Kernel type', data = self.options['KparDiff'].name)
        parameters.create_dataset('Deformation Kernel width', data = self.options['KparDiff'].sigma)
        parameters.create_dataset('Deformation Kernel order', data = self.options['KparDiff'].order)
        parameters.create_dataset('Spatial Varifold Kernel type', data = self.options['KparDist'].name)
        parameters.create_dataset('Spatial Varifold width', data = self.options['KparDist'].sigma)
        parameters.create_dataset('Spatial Varifold order', data = self.options['KparDist'].order)
        template = LDDMMResult.create_group('template')
        template.create_dataset('vertices', data=self.fv0.vertices)
        template.create_dataset('faces', data=self.fv0.faces)
        target = LDDMMResult.create_group('target')
        if isinstance(self.fv1, Surface):
            target.create_dataset('vertices', data=self.fv1.vertices)
            target.create_dataset('faces', data=self.fv1.faces)
        elif isinstance(self.fv1, PointSet):
            target.create_dataset('vertices', data=self.fv1.points)
        deformedTemplate = LDDMMResult.create_group('deformedTemplate')
        deformedTemplate.create_dataset('vertices', data=self.fvDef.vertices)
        variables = LDDMMResult.create_group('variables')
        variables.create_dataset('alpha', data=self.control['at'])
        if self.control['Afft'] is not None:
            variables.create_dataset('affine', data=self.control['Afft'])
        else:
            variables.create_dataset('affine', data='None')
        descriptors = LDDMMResult.create_group('descriptors')

        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.control['Afft'][t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2 + self.dim]
        (xt, Jt) = evol.landmarkDirectEvolutionEuler(self.control['x0'], self.control['at']*self.ds, self.options['KparDiff'], affine=A,
                                                     withJacobian=True)

        AV0 = self.fv0.computeVertexArea()
        AV = self.fvDef.computeVertexArea()[0]/AV0[0]
        descriptors.create_dataset('Jacobian', data=Jt[-1,:])
        descriptors.create_dataset('Surface Jacobian', data=AV)
        descriptors.create_dataset('Displacement', data=xt[-1,...]-xt[0,...])

        fout.close()


    def endOfProcedure(self):
        if self.iter % self.saveRate != 0:
            self.endOfIteration(forceSave=True)

    def optimizeMatching(self):
        if self.unreduced:
            print(f"Unreduced weight: {self.options['unreducedWeight']:0.4f}")

        if self.options['algorithm'] in ('cg', 'bfgs'):
            self.coeffAff = self.coeffAff2
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])

            if self.gradEps < 0:
                self.gradEps = max(1e-5, np.sqrt(grd2) / 10000)
            self.epsMax = 5.
            logging.info(f'Gradient lower bound: {self.gradEps:.5f}')
            self.coeffAff = self.coeffAff1
            if self.options['algorithm'] == 'cg':
                cg.cg(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=.01,
                      Wolfe=self.options['Wolfe'])
            elif self.options['algorithm'] == 'bfgs':
                bfgs.bfgs(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=self.options['epsInit'],
                      Wolfe=self.options['Wolfe'], lineSearch=self.options['lineSearch'], memory=50)
        elif self.options['algorithm'] == 'sgd':
            logging.info('Running stochastic gradient descent')
            sgd.sgd(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                    burnIn=self.sgdBurnIn, epsInit=self.sgdEpsInit, normalization = self.sgdNormalization)

        #return self.at, self.xt

