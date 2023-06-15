import logging
import numpy.linalg as la
from copy import deepcopy
from . import surfaces
from .pointSets import *
from .secondOrderPointSetMatching import SecondOrderPointSetMatching
from .meshMatching import MeshMatching

class SecondOrderMeshMatching(MeshMatching, SecondOrderPointSetMatching):
    def __init__(self, Template=None, Target=None, options=None):
        # self.rescaleTemplate = rescaleTemplate
        MeshMatching.__init__(self, Template=Template, Target=Target, options=options)
        if self.options['internalCost'] is not None:
            self.options['internalCost'] = None
            logging.info(f'Warning: Hybrid models not implemented for second-order methods: using basic LDDMM')


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['initialMomentum'] = None
        return options


    # def setDotProduct(self, unreduced=False):
    #     self.euclideanGradient = True
    #     self.dotProduct = self.dotProduct_euclidean

    def set_parameters(self):
        super().set_parameters()
        if self.options['affine']=='euclidean' or self.options['affine']=='translation':
            self.saveCorrected = True
        else:
            self.saveCorrected = False

    def initialize_variables(self):
        SecondOrderPointSetMatching.initialize_variables(self)
        self.nvert = self.fv0.vertices.shape[0]
        # if self.match_landmarks:
        #     self.x0 = np.concatenate((self.fvInit.vertices, self.tmpl_lmk.vertices), axis=0)
        #     self.nlmk = self.tmpl_lmk.vertices.shape[0]
        # else:
        self.x0 = np.copy(self.fv0.vertices)
            # self.nlmk = 0

        # if self.match_landmarks:
        #     self.def_lmk = PointSet(data=self.tmpl_lmk)
        self.npt = self.x0.shape[0]


    def solveStateEquation(self, control= None, init_state = None, kernel = None, options=None):
        return SecondOrderPointSetMatching.solveStateEquation(self, control=control, init_state=init_state,
                                                              kernel=kernel, options=options)
    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False, display=False):
        return SecondOrderPointSetMatching.objectiveFunDef(self, control, var = var, withTrajectory = withTrajectory,
                                                           withJacobian=withJacobian, display=False)

    def updateTry(self, dr, eps, objRef=None):
        return SecondOrderPointSetMatching.updateTry(self, dr, eps, objRef=objRef)

    def getGradient(self, coeff=1.0, update=None):
        return SecondOrderPointSetMatching.getGradient(self, coeff=coeff, update=update)

    def randomDir(self):
        return SecondOrderPointSetMatching.randomDir(self)

    def dotProduct_euclidean(self, g1, g2):
        return SecondOrderPointSetMatching.dotProduct_euclidean(self, g1, g2)

    def dotProduct_Riemannian(self, g1, g2):
        return SecondOrderPointSetMatching.dotProduct_Riemannian(self, g1, g2)

    def acceptVarTry(self):
        self.obj = self.objTry
        self.control = deepcopy(self.controlTry)

    # def endOfIteration(self, forceSave = False):
    #     self.iter += 1
    #     if (self.iter % self.saveRate == 0):
    #         logging.info('Saving surfaces...')
    #         obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True, withJacobian=True,
    #                                                 display=self.options['verb'])
    #         self.fvDef.updateVertices(self.state['xt'][-1, :, :])
    #
    #
    #         dt = 1.0 / self.Tsize
    #         if self.saveCorrected:
    #             f = self.createObject(self.x0)
    #             X = self.affB.integrateFlow(self.control['Afft'])
    #             displ = np.zeros(self.x0.shape[0])
    #             atCorr = np.zeros(self.state['at'].shape)
    #             for t in range(self.Tsize+1):
    #                 U = la.inv(X[0][t,...])
    #                 yyt = self.state['xt'][t,...]
    #                 yyt = np.dot(yyt - X[1][t, ...], U.T)
    #                 scalars = dict()
    #                 scalars['displacement'] = displ
    #                 if t < self.Tsize:
    #                     a = self.state['at'][t,...]
    #                     a = np.dot(a, X[0][t,...])
    #                     atCorr[t,...] = a
    #                     vt = self.options['KparDiff'].applyK(yyt, a)
    #                     vt = np.dot(vt, U.T)
    #                     # displ += np.sqrt((vt**2).sum(axis=-1))
    #                 self.updateObject(f, yyt)
    #                 scalars = dict()
    #                 scalars['Jacobian'] = self.state['Jt'][t, :]
    #                 vectors = dict()
    #                 vectors['velocity'] = vt
    #                 nu = self.fv0 * f.computeVertexNormals()
    #                 displ += dt * (vt * nu).sum(axis=1)
    #                 vf = surfaces.vtkFields()
    #                 vf.scalars.append('Jacobian')
    #                 vf.scalars.append(displ)
    #                 vf.scalars.append('Jacobian_T')
    #                 vf.scalars.append(displ)
    #                 vf.scalars.append('Jacobian_N')
    #                 vf.scalars.append(displ)
    #                 vf.scalars.append('displacement')
    #                 vf.scalars.append(displ)
    #                 f.saveVTK2(self.options['outputDir'] +'/'+self.options['saveFile']+'Corrected'+str(t)+'.vtk', vf)
    #
    #
    #         fvDef = self.createObject(self.fv0.vertices)
    #         AV0 = fvDef.computeVertexArea()
    #         nu = self.fv0ori*self.fv0.computeVertexNormals()
    #         #v = self.v[0,...]
    #         displ = np.zeros(self.npt)
    #         v = self.options['KparDiff'].applyK(self.x0, self.control['a0'])
    #         for kk in range(self.Tsize+1):
    #             fvDef.updateVertices(self.state['xt'][kk, :, :])
    #             AV = fvDef.computeVertexArea()
    #             AV = AV[0]/AV0[0]
    #             vf = surfaces.vtkFields()
    #             vf.scalars.append('Jacobian')
    #             vf.scalars.append(self.state['Jt'][kk, :])
    #             vf.scalars.append('Jacobian_T')
    #             vf.scalars.append(AV)
    #             vf.scalars.append('Jacobian_N')
    #             vf.scalars.append(self.state['Jt'][kk, :]/AV)
    #             vf.scalars.append('displacement')
    #             vf.scalars.append(displ)
    #             if self.Tsize > 0:
    #                 displ += (v*nu).sum(axis=1) / self.Tsize
    #             if kk < self.Tsize:
    #                 nu = self.fv0ori*fvDef.computeVertexNormals()
    #                 v = self.options['KparDiff'].applyK(self.state['xt'][kk,...], self.state['at'][kk,...])
    #                 #v = self.v[kk,...]
    #
    #             vf.vectors.append('velocity')
    #             vf.vectors.append(np.copy(v))
    #             fvDef.saveVTK2(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk)+'.vtk', vf)
    #     else:
    #         obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True, display=True)
    #         self.fvDef.updateVertices(self.state['xt'][-1, :, :])


