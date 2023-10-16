import logging
import numpy as np
import numpy.linalg as la
from .pointSets import PointSet, savePoints
from .pointSetMatching import PointSetMatching
from . import pointEvolution as evol
# from .affineBasis import getExponential, gradExponential

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'

class State(dict):
    def __init__(self):
        super().__init__()
        self['xt'] = None
        self['at'] = None
        self['yt'] = None
        self['Jt'] = None


class Control(dict):
    def __init__(self):
        super().__init__()
        self['a0'] = None
        self['Afft'] = None


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
class SecondOrderPointSetMatching(PointSetMatching):
    def __init__(self, Template=None, Target=None, options=None):
        # self.rescaleTemplate = rescaleTemplate
        super().__init__(Template=Template, Target=Target, options=options)


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['controlWeight'] = 1.0
        options['initialMomentum'] = None
        return options


    # def setDotProduct(self, unreduced=False):
    #     self.euclideanGradient = True
    #     self.dotProduct = self.dotProduct_euclidean

    def set_parameters(self):
        super().set_parameters()
        # if self.options['affine']=='euclidean' or self.options['affine']=='translation':
        #     self.saveCorrected = True
        # else:
        #     self.saveCorrected = False
        #


    def initialize_variables(self):
        self.x0 = self.fv0.vertices
        self.fvDef = self.createObject(self.x0)
        self.npt = self.x0.shape[0]

        self.Tsize = int(round(1/self.options['timeStep']))
        self.state = State()
        self.control = Control()
        if self.options['initialMomentum']==None:
            self.state['xt'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
            self.control['a0'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.state['at'] = np.tile(self.control['a0'], [self.Tsize+1, 1, 1])
        else:
            self.control['a0'] = self.options['initialMomentum']
            self.state = self.solveStateEquation()
            # (self.state['xt'], self.state['at'])  = evol.secondOrderEvolution(self.x0[-1,...], self.control['a0'],
            #                                                                   self.options['KparDiff'],
            #                                                                   self.options['timeStep'])

        #self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.controlTry = Control()
        self.controlTry['a0'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
        self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])


    def solveStateEquation(self, control= None, init_state = None, kernel = None, options=None):
        if control is None:
            control = self.control
        if init_state is None:
            init_state = self.x0
        if kernel is None:
            kernel = self.options['KparDiff']

        A = self.affB.getTransforms(control['Afft'])

        return evol.secondOrderEvolution(init_state, control['a0'], kernel, self.options['timeStep'],
                                         affine=A, options=options)

    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False, display=False):
        if var is None or 'Init' not in var:
            x0 = self.x0
        else:
            x0 = var['Init'][0]
            
        a0 = control['a0']
        Afft = control['Afft']

        timeStep = self.options['timeStep']
        dim2 = self.dim**2

        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        #print a0.shape
        st = self.solveStateEquation(control=control, init_state=x0, options={'withJacobian':withJacobian})
        #print xt[-1, :, :]
        #print obj
        obj0 = 0.5 * (a0 * self.options['KparDiff'].applyK(x0,a0)).sum()
        obj2 = 0
        for t in range(self.Tsize):
            if self.affineDim > 0:
                obj2 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()/2
            #print xt.sum(), at.sum(), obj
        obj = obj2+obj0
        if display:
            logging.info(f'deformation terms: init {obj0:.4f}, aff {obj2:.4f}')
        if withJacobian or withTrajectory:
            return obj, st
        else:
            return obj


    # def objectiveFun(self):
    #     if self.obj == None:
    #         control = {'a0': self.a0, 'rhot': self.rhot, 'Afft':self.Afft}
    #         (self.obj, self.xt, self.at) = self.objectiveFunDef(control, withTrajectory=True)
    #         self.obj0 = 0
    #         if self.options['errorType'] == 'L2Norm':
    #                 self.obj0 += L2Norm0(self.fv1[k]) / (self.options['sigmaError'] ** 2)
    #             else:
    #                 self.obj0 += self.fun_obj0(self.fv1[k]) / (self.options['sigmaError']**2)
    #             #foo = surfaces.Surface(surf=self.fvDef[k])
    #             self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))
    #             #foo.computeCentersAreas()
    #         self.obj += self.obj0 + self.dataTerm(self.fvDef)
    #         #print self.obj0,  self.dataTerm(self.fvDef)
    #     return self.obj

    def initVariable(self):
        return Control()
    def getVariable(self):
        return self.control
    
    # def updateTry(self, dr, eps, objRef=None):
    #     objTry = self.obj0
    #     #print self.options['typeRegression']
    #     if self.options['typeRegression'] == 'spline':
    #         a00Try = self.a00
    #         rhot0Try = self.rhot0 - eps * dr['rhot0']
    #         a0Try = self.a0
    #         rhotTry = self.rhot - eps * dr['rhot']
    #     elif self.options['typeRegression'] == 'geodesic':
    #         a00Try = self.a00 - eps * dr['a00']
    #         rhot0Try = self.rhot0
    #         a0Try = self.a0 - eps * dr['a0']
    #         rhotTry = self.rhot
    #     elif self.options['typeRegression'] == "affine":
    #         a0Try = self.a0
    #         rhotTry = self.rhot
    #         a00Try = self.a00
    #         rhot0Try = self.rhot0
    #     else:
    #         a00Try = self.a00 - eps * dr['a00']
    #         rhot0Try = self.rhot0 - eps * dr['rhot0']
    #         a0Try = self.a0 - eps * dr['a0']
    #         rhotTry = self.rhot - eps * dr['rhot']
    #
    #     if self.affineDim > 0 and self.options['typeRegression']=="affine":
    #         Afft0Try = self.Afft0 - eps * dr['aff0']
    #         AfftTry = self.Afft - eps * dr['Afft']
    #     else:
    #         Afft0Try = self.Afft0
    #         AfftTry = self.Afft
    #     control = {'a00': a00Try, 'rhot0': rhot0Try, 'Afft0': Afft0Try,
    #                'a0': a0Try, 'rhot': rhotTry, 'Afft': AfftTry}
    #     foo = self.objectiveFunDef(control,  withTrajectory=True)
    #     objTry += foo[0]
    #
    #     ff = []
    #     for k in range(self.nTarg):
    #         ff.append(surfaces.Surface(surf=self.fvDef[k]))
    #         ff[k].updateVertices(np.squeeze(foo[3][self.jumpIndex[k], :, :]))
    #     objTry += self.dataTerm(ff)
    #     if np.isnan(objTry):
    #         print('Warning: nan in updateTry')
    #         return 1e500
    #
    #     if (objRef == None) or (objTry < objRef):
    #         self.a00Try = a00Try
    #         self.rhot0Try = rhot0Try
    #         self.Afft0Try = Afft0Try
    #         self.a0Try = a0Try
    #         self.rhotTry = rhotTry
    #         self.AfftTry = AfftTry
    #         self.objTry = objTry
    #         #print 'objTry=',objTry, dir.diff.sum()
    #
    #     return objTry



    # def endPointGradient(self, endPoint=None):
    #     if endPoint is None:
    #         endPoint = self.fvDef
    #     px = []
    #     for k in range(self.nTarg):
    #         if self.options['errorType'] == 'L2Norm':
    #             targGradient = -L2NormGradient(endPoint[k], self.fv1[k].vfld) / (self.options['sigmaError'] ** 2)
    #         else:
    #             targGradient = -self.fun_objGrad(endPoint[k], self.fv1[k])/(self.options['sigmaError']**2)
    #         px.append(targGradient)
    #     return px

    # def secondOrderCovector(self, x0, control, px1, pa1, affine = None):
    #     a0 = control['a0']
    #     rhot = self.rhot
    #
    #     N = x0.shape[0]
    #     dim = x0.shape[1]
    #     if affine is not None:
    #         aff_ = True
    #         A = affine[0]
    #     else:
    #         aff_ = False
    #
    #     T = self.Tsize
    #
    #     timeStep = self.options['timeStep']
    #     xt, at = evol.secondOrderEvolution(x0, a0, rhot, self.options['KparDiff'], timeStep, affine=affine)
    #     pxt = np.zeros([T+1, N, dim])
    #     pxt[T, :, :] = px1
    #     pat = np.zeros([T+1, N, dim])
    #     pat[T, :, :] = pa1
    #     KparDiff = self.options['KparDiff']
    #     for t in range(T):
    #         px = pxt[T-t, :, :]
    #         pa = pat[T-t, :, :]
    #         x = xt[T-t-1, :, :]
    #         a = at[T-t-1, :, :]
    #         #rho = np.squeeze(rhot[T-t-1, :, :])
    #
    #         if aff_:
    #             U = getExponential(timeStep * A[T-t-1])
    #             px_ = np.dot(px, U)
    #             Ui = la.inv(U)
    #             pa_ = np.dot(pa,Ui.T)
    #         else:
    #             px_ = px
    #             pa_ = pa
    #
    #         zpx = KparDiff.applyDiffKT(x, px_, a, sym=True) - KparDiff.applyDDiffK11and12(x, a, a, pa_)
    #         zpa = KparDiff.applyK(x, px_) - KparDiff.applyDiffK1and2(x, pa_, a)
    #
    #         pxt[T-t-1, :, :] = px_ + timeStep * zpx
    #         pat[T-t-1, :, :] = pa_ + timeStep * zpa
    #
        #####
        #
        # return pxt, pat, xt, at

    # Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
    # def secondOrderGradient(self, x0, control, px1, pa1, getCovector = False, affine=(None, None), controlWeight=1.0):
    #     a0 = control['a0']
    #     pxt, pat, xt, at = evol.secondOrderCovector(x0, a0, px1, pa1, affine=affine)
    #     timeStep = self.options['timeStep']
    #     # KparDiff = self.options['KparDiff0']
    #     # if affine[0] is not None:
    #     #     for t in range(Tsize0):
    #     #         x = np.squeeze(xt0[t, :, :])
    #     #         a = np.squeeze(at0[t, :, :])
    #     #         rho = np.squeeze(rhot0[t, :, :])
    #     #         px = np.squeeze(pxt0[t+1, :, :])
    #     #         pa = np.squeeze(pat0[t+1, :, :])
    #     #         zx = x + timeStep * KparDiff.applyK(x, a)
    #     #         za = a + timeStep * (-KparDiff.applyDiffKT(x, a, a) + rho)
    #     #         U = getExponential(timeStep * affine[0][0][t])
    #     #         #U = np.eye(dim) + timeStep * affine[0][k]
    #     #         Ui = la.inv(U)
    #     #         pa = np.dot(pa, Ui.T)
    #     #         za = np.dot(za, Ui)
    #     #         dA0[t,...] =  (gradExponential(timeStep*affine[0][0][t], px, zx)
    #     #                         - gradExponential(timeStep*affine[0][0][t], za, pa))
    #     #         drhot0[t,...] = rho*controlWeight - pa
    #     #     db0 = pxt0[1:Tsize0+1,...].sum(axis=1)
    #     # else:
    #     #     for t in range(Tsize0):
    #     #         rho = rhot0[t, :, :]
    #     #         pa = pat0[t+1, :, :]
    #     #         drhot0[t,...] = rho*controlWeight - pa
    #     #
    #     # da00 = KparDiff.applyK(x00, a00) - pat0[0,...]
    #
    #
    #     if affine is not None:
    #         dA = np.zeros(affine[0].shape)
    #         db = np.zeros(affine[1].shape)
    #     else:
    #         dA = None
    #         db = None
    #     Tsize = self.Tsize
    #     KparDiff = self.options['KparDiff']
    #     if affine is not None:
    #         for t in range(Tsize):
    #             x = xt[t, :, :]
    #             a = at[t, :, :]
    #             px = pxt[t+1, :, :]
    #             pa = pat[t+1, :, :]
    #             zx = x + timeStep * KparDiff.applyK(x, a)
    #             za = a + timeStep * (-KparDiff.applyDiffKT(x, a, a))
    #             U = getExponential(timeStep * affine[0][t])
    #             Ui = la.inv(U)
    #             pa = np.dot(pa, Ui.T)
    #             za = np.dot(za, Ui)
    #             dA[t,...] =  (gradExponential(timeStep*affine[0][t], px, zx)
    #                             - gradExponential(timeStep*affine[0][t], za, pa))
    #         db = pxt[1:Tsize+1,...].sum(axis=1)
    #
    #     da0 = KparDiff.applyK(xt[0,...], a0) - pat[0,...]
    #
    #     if affine is None:
    #         if getCovector == False:
    #             return da0, xt, at
    #         else:
    #             return da0, xt, at, pxt, pat
    #     else:
    #         if getCovector == False:
    #             return da0, dA, db, xt, at
    #         else:
    #             return da0, dA, db, xt, at, pxt, pat
    #
    #


    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]
        # A = self.affB.getTransforms(control['Afft'])
        st = self.solveStateEquation(control=control)
        # xt, at = evol.secondOrderEvolution(self.x0, control['a0'], self.options['KparDiff'],
        #                                    self.options['timeStep'], affine=A)
        endPoint = self.createObject(self.fv0.vertices)
        self.updateObject(endPoint, st['xt'][-1, :, :])
        # st = State()
        # st['xt'] = xt
        # st['at'] = at

        return control, st, endPoint


    def getGradient(self, coeff=1.0, update=None):
        A = None
        A0 = None
        #logging.info('Computing gradient')
        if update is None:
            control = self.control
            endPoint = self.fvDef
            state = self.state
        else:
            control, state, endPoint = self.setUpdate(update)
        # if update is None:
        #     a0 = self.a0
        #     a00 = self.a00
        #     rhot = self.rhot
        #     rhot0 = self.rhot0
        #     endPoint = self.fvDef
        #     if len(self.Afft) > 0:
        #         A = self.affB.getTransforms(self.Afft)
        #     if len(self.Afft0) > 0:
        #         A0 = self.affB.getTransforms(self.Afft0)
        # else:
        #     eps = update[1]
        #     dr = update[0]
        #     if self.options['typeRegression'] == 'spline':
        #         a00 = self.a00
        #         rhot0 = self.rhot0 - eps * dr['rhot0']
        #         a0 = self.a0
        #         rhot = self.rhot - eps * dr['rhot']
        #     elif self.options['typeRegression'] == 'geodesic':
        #         a00 = self.a00 - eps * dr['a00']
        #         rhot0 = self.rhot0
        #         a0 = self.a0 - eps * dr['a0']
        #         rhot = self.rhot
        #     elif self.options['typeRegression'] == "affine":
        #         a0 = self.a0
        #         rhot = self.rhot
        #         a00 = self.a00
        #         rhot0 = self.rhot0
        #     else:
        #         a00 = self.a00 - eps * dr['a00']
        #         rhot0 = self.rhot0 - eps * dr['rhot0']
        #         a0 = self.a0 - eps * dr['a0']
        #         rhot = self.rhot - eps * dr['rhot']
        #
        #     if self.affineDim > 0 and self.options['typeRegression'] == "affine":
        #         Afft0 = self.Afft0 - eps * dr['aff0']
        #         Afft = self.Afft - eps * dr['Afft']
        #     else:
        #         Afft0 = self.Afft0
        #         Afft = self.Afft
        #
        #     if len(update[0]['aff0']) > 0:
        #         A0 = self.affB.getTransforms(Afft0)
        #     if len(update[0]['Afft']) > 0:
        #         A = self.affB.getTransforms(Afft)
        #
        #     # a0 = self.a0 - update[1] * update[0]['a0']
        #     # a00 = self.a0 - update[1] * update[0]['a00']
        #     # rhot = self.rhot - update[1] * update[0]['rhot']
        #     # rhot0 = self.rhot0 - update[1] * update[0]['rhot0']
        #     # if len(update[0]['aff0']) > 0:
        #     #     A0 = self.affB.getTransforms(self.Afft0 - update[1]*update[0]['aff0'])
        #     # if len(update[0]['Afft']) > 0:
        #     #     A = self.affB.getTransforms(self.Afft - update[1]*update[0]['Afft'])
        #     (xt0, at0)  = evol.secondOrderEvolution(self.x0, a00, rhot0, self.options['KparDiff0'],
        #                                             self.options['timeStep'])
        #     (xt, at)  = evol.secondOrderEvolution(xt0[-1,:,:], a0, rhot, self.options['KparDiff'],
        #                                           self.options['timeStep'])
        #     endPoint = []
        #     for k in range(self.nTarg):
        #         fvDef = surfaces.Surface(surf=self.fv0)
        #         fvDef.updateVertices(xt[self.jumpIndex[k], :, :])
        #         endPoint.append(fvDef)

        A = self.affB.getTransforms(control['Afft'])
        px1 = -self.endPointGradient(endPoint=endPoint)
        pa1 = np.zeros(self.control['a0'].shape)

        foo = evol.secondOrderGradient(self.x0, control['a0'], px1, pa1, self.options['KparDiff'],
                                       self.options['timeStep'], affine=A)
        grd = Control()
        # # if self.euclideanGradient:
        # #     grd['a0'] = self.options['KparDiff'].applyK(self.x0, foo['da0'])/coeff
        # else:
        grd['a0'] = foo['da0']/coeff
        grd['Afft'] = np.zeros(self.control['Afft'].shape)
        if self.affineDim > 0:
            dim2 = self.dim**2
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*control['Afft']
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] /= (self.coeffAff*coeff*self.Tsize)

        # if self.options['typeRegression'] == 'affine':
        #     grd['a00'] = np.zeros(foo[0][0].shape)
        #     grd['rhot0'] = np.zeros(foo[0][1].shape)
        # else:
        #     grd['a00'] = foo[0][0] / coeff
        #     grd['rhot0'] = foo[0][1] / coeff
        
        return grd



    def randomDir(self):
        dirfoo = Control()
        dirfoo['a0'] = np.random.randn(self.npt, self.dim)
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        else:
            dirfoo['Afft'] = None
        return dirfoo


    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        z = self.x0
        gg = g1['a0']
        u = self.options['KparDiff'].applyK(z, gg)
        ll=0
        for gr in g2:
            ggOld = gr['a0']
            res[ll] = res[ll] + (ggOld * u).sum()
            ll = ll + 1

        if self.affineDim > 0:
            for t in range(self.Tsize):
                uu = g1['Afft'][t]
                ll = 0
                for gr in g2:
                    res[ll] += (uu*gr['Afft'][t]).sum() * self.coeffAff
                    ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        u = g1['a0']
        ll=0
        for gr in g2:
            ggOld = gr['a0']
            res[ll] = res[ll] + (ggOld * u).sum()
            ll = ll + 1

        if self.affineDim > 0:
            for t in range(self.Tsize):
                uu = g1['Afft'][t]
                ll = 0
                for gr in g2:
                    res[ll] += (uu*gr['Afft'][t]).sum()
                    ll = ll + 1
        return res





    def endOfIteration(self, forceSave = False):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.affine = 'none'
            #self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0):
            logging.info('Saving surfaces...')
            control = self.control
            obj1, self.state = self.objectiveFunDef(control, withTrajectory=True, display=self.options['verb'])
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])
            # dim2 = self.dim**2
            # if self.affineDim > 0:
            #     A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            #     for t in range(self.Tsize):
            #         AB = np.dot(self.affineBasis, self.control['Afft'][t])
            #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
            #         A[1][t] = AB[dim2:dim2+self.dim]
            # else:
            #     A = None

            dt = 1.0 / self.Tsize
            if self.options['saveCorrected']:
                f = self.createObject(self.x0)
                X = self.affB.integrateFlow(self.control['Afft'])
                displ = np.zeros(self.x0.shape[0])
                atCorr = np.zeros(self.state['at'].shape)
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t,...])
                    yyt = self.state['xt'][t,...]
                    yyt = np.dot(yyt - X[1][t, ...], U.T)
                    scalars = dict()
                    scalars['displacement'] = displ
                    if t < self.Tsize:
                        a = self.state['at'][t,...]
                        a = np.dot(a, X[0][t,...])
                        atCorr[t,...] = a
                        vt = self.options['KparDiff'].applyK(yyt, a)
                        vt = np.dot(vt, U.T)
                        displ += dt*np.sqrt((vt**2).sum(axis=-1))
                    self.updateObject(f, yyt)
                    scalars = dict()
                    scalars['Jacobian'] = self.state['Jt'][t, :]
                    vectors = dict()
                    vectors['velocity'] = vt
                    savePoints(self.options['outputDir'] +'/'+self.options['saveFile']+'Corrected'+str(t+self.Tsize)+'.vtk',
                               f.vertices, vectors = vectors, scalars=scalars)
#                (foo,zt) = evol.landmarkDirectEvolutionEuler(self.x0, atCorr, self.options['KparDiff'], withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.options['outputDir'] +'/'+self.saveFile+'CorrectedCheck'+str(t)+'.vtk')
#                (foo,foo2,zt) = evol.secondOrderEvolution(self.x0, atCorr[0,...], self.rhot, self.options['KparDiff'], withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.options['outputDir'] +'/'+self.saveFile+'CorrectedCheckBis'+str(t)+'.vtk')
                 

                f = PointSet(data = self.fv1)
                U = la.inv(X[0][-1])
                yyt = f.vertices
                yyt = np.dot(yyt - X[1][-1, ...], U.T)
                f.updateVertices(yyt)
                savePoints(self.options['outputDir'] +'/TargetCorrected.vtk', f)
            
            fvDef = self.createObject(self.x0)
            #v = self.v[0,...]
            displ = np.zeros(self.npt)
            # dt = 1.0 /self.Tsize
            v = self.options['KparDiff'].applyK(self.x0, self.state['at'][0,...])
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(self.state['xt'][-1, :, :])
                scalars = dict()
                scalars['Jacobian'] = self.state['Jt'][kk, :]
                scalars['displacement'] = displ
                if self.Tsize > 0:
                    displ += np.sqrt((v ** 2).sum(axis=-1))
                if kk < self.Tsize:
                    v = self.options['KparDiff'].applyK(self.state['xt'][kk,...], self.state['at'][kk,...])
                    #v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                vectors = dict()
                vectors['velocity'] = v
                savePoints(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk+self.Tsize)+'.vtk', fvDef,
                           vectors=vectors, scalars=scalars)
        else:
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True, display=True)
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])


    # def optimizeMatching(self):
    #     #print 'dataterm', self.dataTerm(self.fvDef)
    #     #print 'obj fun', self.objectiveFun(), self.obj0
    #     grd = self.getGradient(self.gradCoeff)
    #     [grd2] = self.dotProduct(grd, [grd])
    #
    #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
    #     logging.info('Gradient lower bound: {0:f}'.format(self.gradEps))
    #     #print 'x0:', self.x0
    #     #print 'y0:', self.y0
    #     self.cgBurnIn = self.affBurnIn
    #
    #     if self.options['algorithm'] == 'cg':
    #         cg.cg(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
    #               TestGradient=self.options['testGradient'], epsInit=.01,
    #               Wolfe=self.options['Wolfe'])
    #     elif self.options['algorithm'] == 'bfgs':
    #         bfgs.bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
    #                   TestGradient=self.options['testGradient'], epsInit=1.,
    #                   Wolfe=self.options['Wolfe'], lineSearch=self.options['lineSearch'], memory=50)
    #     #return self.at, self.xt
