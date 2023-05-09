import os
import numpy as np
from copy import deepcopy
import numpy.linalg as la
import logging
import h5py
import glob
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, bfgs
from . import surfaces
from . import surface_distances
from . import pointSets
from .surfaceMatching import SurfaceMatching, Control as SMControl, State as SMState
from .affineBasis import AffineBasis, getExponential, gradExponential
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Control(dict):
    def __init__(self):
        super().__init__()
        self['at0'] =None
        self['at1'] =None
        self['Afft'] = None

class State(dict):
    def __init__(self):
        super().__init__()
        self['xt0'] =None
        self['xt1'] =None


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
class SurfaceMatchingMidpoint(SurfaceMatching):
    # def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, maxIter=1000,
    #              regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
    #              affineOnly = False,
    #              rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
    #              testGradient=True, saveFile = 'evolution',
    #              saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):
    #     if param == None:
    #         self.param = SurfaceMatchingParam()
    #     else:
    #         self.param = param
    #
    #     if self.param.algorithm == 'cg':
    #         self.euclideanGradient = False
    #     else:
    #         self.euclideanGradient = True
    #
    #     self.setOutputDir(outputDir)
    #     self.set_fun(self.param.errorType,vfun=self.param.vfun)
    #
    #     self.set_template_and_target(Template, Target)
    #     self.match_landmarks = False
    #
    #     self.set_parameters(maxIter=maxIter, regWeight=regWeight, affineWeight=affineWeight,
    #                         internalWeight=internalWeight, verb=verb, affineOnly=affineOnly,
    #                         rotWeight=rotWeight, scaleWeight=scaleWeight, transWeight=transWeight,
    #                         symmetric=symmetric, testGradient=testGradient, saveFile=saveFile,
    #                         saveTrajectories=saveTrajectories, affine=affine)
    #
    #     self.initialize_variables()
    #     self.gradCoeff = self.x0.shape[0]
    #
    #     self.pplot = pplot
    #     if self.pplot:
    #         self.initial_plot()


    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.x0 = np.copy(self.fv0.vertices)
        self.fvDef0 = surfaces.Surface(surf=self.fv0)
        self.x1 = np.copy(self.fv1.vertices)
        self.fvDef1 = surfaces.Surface(surf=self.fv1)
        #self.nvert = self.fvInit.vertices.shape[0]
        self.npt0 = self.fv0.vertices.shape[0]
        self.npt1 = self.fv1.vertices.shape[0]

        self.control = Control()
        self.controlTry = Control()
        self.control['at0'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.randomInit:
            self.control['at0'] = np.random.normal(0, 1, self.control['at0'].shape)
        self.controlTry['at0'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.control['at1'] = np.zeros([self.Tsize, self.x1.shape[0], self.x1.shape[1]])
        if self.randomInit:
            self.control['at1'] = np.random.normal(0, 1, self.control['at1'].shape)
        self.controlTry['at1'] = np.zeros([self.Tsize, self.x1.shape[0], self.x1.shape[1]])
        if self.affineDim > 0:
            self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.state = State()
        self.state['xt0'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.state['xt1'] = np.tile(self.x1, [self.Tsize+1, 1, 1])
        self.v0 = np.zeros([self.Tsize+1, self.npt0, self.dim])
        self.v1 = np.zeros([self.Tsize+1, self.npt1, self.dim])


    def set_fun(self, errorType, vfun=None):
        self.options['errorType'] = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj = partial(surface_distances.currentNorm, KparDist=self.options['KparDist'], weight=1.)
            self.fun_objGrad = partial(surface_distances.currentNormGradient, KparDist=self.options['KparDist'], weight=1.)
        elif errorType=='measure':
            #print('Running Measure Matching')
            self.fun_obj = partial(surface_distances.measureNorm,KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(surface_distances.measureNormGradient,KparDist=self.options['KparDist'])
        elif errorType=='varifold':
            self.fun_obj = partial(surface_distances.varifoldNorm, KparDist=self.options['KparDist'], fun=vfun)
            self.fun_objGrad = partial(surface_distances.varifoldNormGradient, KparDist=self.options['KparDist'], fun=vfun)
        else:
            print('Unknown error Type: ', self.options['errorType'])


    def dataTerm(self, _fvDef, var=None):
        obj = self.fun_obj(_fvDef[0], _fvDef[1]) / (self.options['sigmaError']**2)
        return obj

    def  objectiveFunDef2(self, control, var = None, withTrajectory = False,
                          withJacobian=False):
        at0 = control['at0']
        at1 = control['at1']
        Afft = control['Afft']

        if var is None:
            var2 = {}
        else:
            var2 = var
        var2['fv0'] = self.fv0
        res0 = self.objectiveFunDef({'at':at0, 'Afft':Afft}, var = var2, withTrajectory=withTrajectory,
                                    withJacobian=withJacobian)
        var2['fv0'] = self.fv1
        res1 = self.objectiveFunDef({'at':at1}, var = var2, withTrajectory=withTrajectory,
                                    withJacobian=withJacobian)
        if withJacobian or withTrajectory:
            return res0[0]+res1[0], [res0[1], res1[1]]
        else:
            return res0+res1


    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = 0 #self.fun_obj0(self.fv1) / (self.param.sigmaError**2)
            (self.obj, xt) = self.objectiveFunDef2(self.control, withTrajectory=True,
                                                   var = {'regWeight':self.options['regWeight']})
            self.state['xt0'] = xt[0]['xt']
            self.state['xt1'] = xt[1]['xt']
            self.fvDef0.updateVertices(self.state['xt0'][-1, :, :])
            self.fvDef1.updateVertices(self.state['xt1'][-1, :, :])
            self.obj += self.obj0 + self.dataTerm([self.fvDef0, self.fvDef1])
        return self.obj

    def getVariable(self):
        return self.control

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        controlTry = Control()
        controlTry['at0'] = self.control['at0'] - eps * dir['at0']
        controlTry['at1'] = self.control['at1'] - eps * dir['at1']
        if self.affineDim > 0:
            controlTry['Afft'] = self.control['Afft'] - eps * dir['Afft']
        else:
            controlTry['Afft'] = self.control['Afft']

        foo = self.objectiveFunDef2(controlTry, withTrajectory=True, var={'regWeight': self.options['regWeight']})
        objTry += foo[0]

        ff0 = surfaces.Surface(surf=self.fvDef0)
        ff0.updateVertices(foo[1][0]['xt'][-1, :, :])
        ff1 = surfaces.Surface(surf=self.fvDef1)
        ff1.updateVertices(foo[1][1]['xt'][-1, :, :])
        objTry += self.dataTerm([ff0, ff1])
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) or (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry

        return objTry


    def testEndpointGradient(self):
        c0 = self.dataTerm([self.fvDef0, self.fvDef1])
        ff0 = surfaces.Surface(surf=self.fvDef0)
        ff1 = surfaces.Surface(surf=self.fvDef1)
        dff0 = np.random.normal(size=ff0.vertices.shape)
        dff1 = np.random.normal(size=ff1.vertices.shape)
        eps = 1e-6
        ff0.updateVertices(ff0.vertices+eps*dff0)
        ff1.updateVertices(ff1.vertices+eps*dff1)
        c1 = self.dataTerm([ff0, ff1])
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps,
                                                                      (grd[0]*dff0).sum()
                                                                      + (grd[1]*dff1).sum()) )

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint0 = self.fvDef0
            endPoint1 = self.fvDef1
        else:
            endPoint0 = endPoint[0]
            endPoint1 = endPoint[1]
        px0 = self.fun_objGrad(endPoint0, endPoint1)
        px1 = self.fun_objGrad(endPoint1, endPoint0)
        return px0 / self.options['sigmaError']**2, px1 / self.options['sigmaError']**2


    # def hamiltonianCovector2(self, x0, x1, at0, at1, px01, px11, KparDiff, regWeight, affine = None):
    #     pxt0, xt0 = self.hamiltonianCovector(at0, px01, KparDiff, regWeight, affine = affine)
    #     pxt1, xt1 = self.hamiltonianCovector(at1, px11, KparDiff, regWeight)
    #     return [pxt0, pxt1], [xt0, xt1]

    def hamiltonianGradient2(self, px01, px11, kernel = None, regWeight=None, control = None):
        if control is None:
            control = self.control
        control0 = SMControl()
        control0['at'] = control['at0']
        control0['Afft'] = control['Afft']
        control1 = SMControl()
        control1['at'] = control['at1']
        res0 = self.hamiltonianGradient(px01, kernel=kernel, regWeight=regWeight,
                                                       fv0= self.fv0, control=control0)
        res1 = self.hamiltonianGradient(px11, kernel = kernel, regWeight=regWeight,
                                                   fv0=self.fv1, control=control1)

        return res0, res1
        # if control0['Afft'] is None:
        #     return [res0[0], res1[0]], [res0[1], res1[1]], [res0[2], res1[2]]
        # else:
        #     return [res0[0], res1[0]], res0[1], res0[2], [res0[3], res1[3]], [res0[4], res1[4]]


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            at0 = self.control['at0']
            endPoint0 = self.fvDef0
            at1 = self.control['at1']
            Afft = self.control['Afft']
            endPoint1 = self.fvDef1
            if self.affineDim > 0:
                A = self.affB.getTransforms(self.control['Afft'])
            else:
                A = None
        else:
            if self.affineDim > 0:
                Afft = self.control['Afft'] - update[1]*update[0]['Afft']
                A = self.affB.getTransforms(Afft)
            else:
                A = None
                Afft = None
            at0 = self.control['at0'] - update[1] *update[0]['at0']
            xt0 = evol.landmarkDirectEvolutionEuler(self.x0, at0, self.options['KparDiff'], affine=A)
            endPoint0 = surfaces.Surface(surf=self.fv0)
            endPoint0.updateVertices(xt0[-1, :, :])
            at1 = self.control['at1'] - update[1] *update[0]['at1']
            xt1 = evol.landmarkDirectEvolutionEuler(self.x1, at1, self.options['KparDiff'])
            endPoint1 = surfaces.Surface(surf=self.fv1)
            endPoint1.updateVertices(xt1[-1, :, :])


        control = Control()
        control['at0'] = at0
        control['at1'] = at1
        control['Afft'] = Afft
        px1 = self.endPointGradient(endPoint=[endPoint0, endPoint1])
        px01 = -px1[0]
        px11 = -px1[1]
        foo = self.hamiltonianGradient2(px01, px11, control=control, regWeight=self.options['regWeight'])
        grd = Control()
        # if self.euclideanGradient:
        #     grd['diff0'] = np.zeros(foo[0][0].shape)
        #     grd['diff1'] = np.zeros(foo[0][1].shape)
        #     for t in range(self.Tsize):
        #         z = self.xt0[t, :, :]
        #         grd['diff0'][t,:,:] = self.options['KparDiff'].applyK(z, foo[0][0][t, :,:])/(coeff*self.Tsize)
        #         z = self.xt1[t, :, :]
        #         grd['diff1'][t,:,:] = self.options['KparDiff'].applyK(z, foo[0][1][t, :,:])/(coeff*self.Tsize)
        # else:
        grd['at0'] = foo[0]['dat']/(coeff*self.Tsize)
        grd['at1'] = foo[1]['dat'] / (coeff * self.Tsize)
        if self.affineDim > 0:
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            dim2 = self.dim ** 2
            dA = foo[0]['dA']
            db = foo[0]['db']
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*self.control['Afft']
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] /= (self.coeffAff*coeff*self.Tsize)
        return grd



    # def addProd(self, dir1, dir2, beta):
    #     dr = Direction()
    #     for k in dir1.keys():
    #         if k != 'aff' or self.affineDim > 0:
    #             dr[k] = dir1[k] + beta * dir2[k]
    #     return dr
    #
    # def prod(self, dir1, beta):
    #     dr = Direction()
    #     for k in dir1.keys():
    #         if k != 'aff' or self.affineDim > 0:
    #             dr[k] = beta * dir1[k]
    #     return dr
    #
    # def copyDir(self, dir0):
    #     dr = Direction()
    #     for k in dir0.keys():
    #         dr[k] = np.copy(dir0[k])
    #     return dr


    def randomDir(self):
        dirfoo = Control()
        if self.affineOnly:
            dirfoo['at0'] = np.zeros((self.Tsize, self.npt0, self.dim))
            dirfoo['at1'] = np.zeros((self.Tsize, self.npt1, self.dim))
        else:
            dirfoo['at0'] = np.random.randn(self.Tsize, self.npt0, self.dim)
            dirfoo['at1'] = np.random.randn(self.Tsize, self.npt1, self.dim)
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z0 = np.squeeze(self.state['xt0'][t, :, :])
            gg0 = np.squeeze(g1['at0'][t, :, :])
            z1 = np.squeeze(self.state['xt1'][t, :, :])
            gg1 = np.squeeze(g1['at1'][t, :, :])
            u0 = self.options['KparDiff'].applyK(z0, gg0)
            u1 = self.options['KparDiff'].applyK(z1, gg1)
            #uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            uu = g1['aff'][t]
            ll = 0
            for gr in g2:
                ggOld0 = np.squeeze(gr['at0'][t, :, :])
                ggOld1 = np.squeeze(gr['at1'][t, :, :])
                res[ll]  += (ggOld0*u0).sum() + (ggOld1*u1).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['Afft'][t]).sum() * self.coeffAff
                ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            u0 = np.squeeze(g1['at0'][t, :, :])
            u1 = np.squeeze(g1['at1'][t, :, :])
            if self.affineDim > 0:
                uu = (g1['Afft'][t, :, :]*self.options['affineWeight'])#.reshape(g1['aff'][t].shape))
            ll = 0
            for gr in g2:
                ggOld0 = np.squeeze(gr['at0'][t, :, :])
                ggOld1 = np.squeeze(gr['at1'][t, :, :])
                res[ll]  += (ggOld0*u0).sum() + (ggOld1*u1).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['aff'][t, :, :]).sum()
                ll = ll + 1
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.control = deepcopy(self.controlTry)


    def endOfIteration(self, forceSave=False):
        self.iter += 1
        if self.options['testGradient']:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        if (self.iter % self.saveRate == 0) :
            logging.info('Saving surfaces...')
            (obj1, st) = self.objectiveFunDef2(self.control, withTrajectory=True,
                                               var = {'regWeight':self.options['regWeight']})
            self.state['xt0'] = st[0]['xt']
            self.state['xt1'] = st[1]['xt']

            if self.options['saveTrajectories']:
                pointSets.saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curves0.vtk', self.state['xt0'])
                pointSets.saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curves1.vtk', self.state['xt1'])

            self.fvDef0.updateVertices(np.squeeze(self.state['xt0'][-1, :, :]))
            self.fvDef1.updateVertices(np.squeeze(self.state['xt1'][-1, :, :]))
            dim2 = self.dim**2
            if self.affineDim > 0:
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.control['Afft'][t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            else:
                A = None

            ct0 = SMControl()
            ct0['at'] = self.control['at0']
            ct0['Afft'] = self.control['Afft']
            ct1 = SMControl()
            ct1['at'] = self.control['at1']

            st0 = SMState()
            (st0['xt'], st0['Jt'])  = evol.landmarkDirectEvolutionEuler(self.x0, self.control['at0'],
                                                                        self.options['KparDiff'], affine=A,
                                                                        withJacobian=True)
            st1 = SMState()
            (st1['xt'], st1['Jt'])  = evol.landmarkDirectEvolutionEuler(self.x1, self.control['at1'],
                                                                        self.options['KparDiff'], withJacobian=True)
            if self.options['affine']=='euclidean' or self.options['affine']=='translation':
                self.saveCorrectedEvolution(self.fv0, st0, ct0, fileName=self.options['saveFile'] +'0')
            self.saveEvolution(self.fv0, st0, fileName=self.options['saveFile']+'_0_', velocity=self.v0)
            self.saveEvolution(self.fv1, st1, fileName=self.options['saveFile']+'_1_', velocity=self.v1,
                               orientation=1)
        else:
            (obj1, st) = self.objectiveFunDef2(self.control, withTrajectory=True,
                                               var = {'regWeight':self.options['regWeight']})
            self.state['xt0'] = st[0]['xt']
            self.state['xt1'] = st[1]['xt']
            self.fvDef0.updateVertices(np.squeeze(self.state['xt0'][-1, :, :]))
            self.fvDef1.updateVertices(np.squeeze(self.state['xt1'][-1, :, :]))

        if self.options['pplot']:
            fig=plt.figure(4)
            #fig.clf()
            ax = Axes3D(fig)
            lim0 = self.addSurfaceToPlot(self.fvDef0, ax, ec = 'k', fc = 'b')
            lim1 = self.addSurfaceToPlot(self.fvDef1, ax, ec='k', fc='r')
            ax.set_xlim(min(lim0[0][0],lim1[0][0]), max(lim0[0][1],lim1[0][1]))
            ax.set_ylim(min(lim0[1][0],lim1[1][0]), max(lim0[1][1],lim1[1][1]))
            ax.set_zlim(min(lim0[2][0],lim1[2][0]), max(lim0[2][1],lim1[2][1]))
            fig.canvas.flush_events()
            #plt.axis('equal')
            #plt.pause(0.1)


    def endOfProcedure(self):
        self.endOfIteration()

    def optimizeMatching(self):
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        self.epsMax = 5.
        logging.info('Gradient lower bound: %f' %(self.gradEps))
        self.coeffAff = self.coeffAff1
        if self.options['algorithm'] == 'cg':
            cg.cg(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                  TestGradient=self.options['testGradient'], Wolfe=self.options['Wolfe'], epsInit=0.01)
        elif self.options['algorithm'] == 'bfgs':
            bfgs.bfgs(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=1., Wolfe=self.options['Wolfe'], memory=50)

