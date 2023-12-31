import logging
import numpy.linalg as la
from copy import deepcopy
from . import surfaces
from .pointSets import *
from .surfaceTimeSeries import SurfaceTimeMatching
from .secondOrderPointSetMatching import SecondOrderPointSetMatching
from . import pointEvolution as evol
from .affineBasis import getExponential, gradExponential
from .surfaceDistancess import L2Norm0

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'

class Control(dict):
    def __init__(self):
        super().__init__()
        self['a0'] = None
        self['rhot'] = None
        self['aff'] = None
        self['a00'] = None
        self['rhot0'] = None
        self['aff0'] = None

class State(dict):
    def __init__(self):
        super().__init__()
        self['xt0'] = None
        self['Jt0'] = None
        self['xt'] = None
        self['Jt'] = None


class SecondOrderSurfaceTimeMatching(SurfaceTimeMatching, SecondOrderPointSetMatching):
    def __init__(self, Template=None, Target=None, options=None):
        # self.rescaleTemplate = rescaleTemplate
        super().__init__(Template=Template, Target=Target, options=options)


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['typeRegression'] = False
        options['controlWeight'] = 1.0
        options['initialMomentum'] = None
        options['KparDiff0'] = None
        return options


    def setDotProduct(self, unreduced=False):
        self.euclideanGradient = True
        self.dotProduct = self.dotProduct_euclidean

    def set_parameters(self):
        super().set_parameters()
        if self.options['affine']=='euclidean' or self.options['affine']=='translation':
            self.saveCorrected = True
        else:
            self.saveCorrected = False

        self.typeRegressionSave = self.options['typeRegression']

        if self.options['affine'] != 'none':
            self.options['typeRegression'] = 'affine'

        if self.options['KparDiff0'] is None:
            self.options['KparDiff0'] = self.options['KparDiff']

        self.isActive = dict()
        if self.options['typeRegression'] == 'spline':
            self.isActive['a00'] = False
            self.isActive['rhot0'] = True
            self.isActive['a0'] = False
            self.isActive['rhot'] = True
        elif self.options['typeRegression'] == 'geodesic':
            self.isActive['a00'] = True
            self.isActive['rhot0'] = False
            self.isActive['a0'] = True
            self.isActive['rhot'] = False
        elif self.options['typeRegression'] == "affine":
            self.isActive['a00'] = False
            self.isActive['rhot0'] = False
            self.isActive['a0'] = False
            self.isActive['rhot'] = False
        else:
            self.isActive['a00'] = True
            self.isActive['rhot0'] = True
            self.isActive['a0'] = True
            self.isActive['rhot'] = True

    def initialize_variables(self):
        self.x0 = self.fv0.vertices
        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.npt = self.x0.shape[0]
        self.nvert = self.fvInit.vertices.shape[0]
        if self.options['times'] is None:
            self.times = np.arange(1, self.nTarg+1)
        else:
            self.times = np.array(self.options['times'])

        self.Tsize0 = int(round(1./self.options['timeStep']))
        self.Tsize = int(round(self.times[-1]/self.options['timeStep']))
        self.jumpIndex = np.round(self.times/self.options['timeStep']).astype(int)
        #print self.jumpIndex
        self.isjump = np.zeros(self.Tsize+1, dtype=bool)
        for k in self.jumpIndex:
            self.isjump[k] = True

        self.control['rhot0'] = np.zeros([self.Tsize0, self.x0.shape[0], self.x0.shape[1]])
        self.control['rhot'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.options['initialMomentum']==None:
            self.state['xt0'] = np.tile(self.x0, [self.Tsize0+1, 1, 1])
            self.state['xt'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
            self.control['a00'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.control['a0'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.state['at'] = np.tile(self.control['a0'], [self.Tsize+1, 1, 1])
            self.state['at0'] = np.tile(self.control['a00'], [self.Tsize0+1, 1, 1])
        else:
            self.control['a00'] = self.options['initialMomentum'][0]
            self.control['a0'] = self.options['initialMomentum'][1]
            self.state = self.solveStateEquation()
            # (self.control['xt0'], self.at0)  = evol.secondOrderEvolution(self.x0, self.a00, self.options['KparDiff0'],
            #                                                   self.options['timeStep'], withSpline=self.rhot0)
            # (self.xt, self.at)  = evol.secondOrderEvolution(self.x0[-1,...], self.a0, self.options['KparDiff'],
            #                                                 self.options['timeStep'], withSpline=self.rhot)

        self.control['Afft0'] = np.zeros([self.Tsize0, self.affineDim])
        self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
        #self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.controlTry = deepcopy(self.control)

    # def set_template_and_target(self, Template, Target, subsampleTargetSize=-1):
    #     if Template is None:
    #         logging.error('Please provide a template surface')
    #         return
    #     else:
    #         self.fv0 = surfaces.Surface(surf=Template)
    # 
    #     if Target is None:
    #         logging.error('Please provide a list of target surfaces')
    #         return
    #     else:
    #         self.fv1 = []
    #         if self.options['errorType'] == 'L2Norm':
    #             for f in Target:
    #                 fv1 = surfaces.Surface()
    #                 fv1.readFromImage(f)
    #                 self.fv1.append(fv1)
    #         else:
    #             for f in Target:
    #                 self.fv1.append(surfaces.Surface(surf=f))
    #     self.fix_orientation()
    #     self.dim = self.fv0.vertices.shape[1]
    # 


    def solveStateEquation(self, control= None, init_state = None, kernel = None, options=None):
        if control is None:
            control = self.control
        if init_state is None:
            init_state = self.x0

        A0 = self.affB.getTransforms(control['Afft0'])
        A = self.affB.getTransforms(control['Afft'])

        st0 = evol.secondOrderEvolution(init_state, control['a00'], self.options['KparDiff0'], self.options['timeStep'],
                                        affine=A0, withSpline=control['rhot0'], options=options)
        st = evol.secondOrderEvolution(st0['xt'][-1, ...], self.control['a0'], self.options['KparDiff'],
                                        self.options['timeStep'], affine=A, withSpline=self.control['rhot'],
                                       options=options)
        st['xt0'] = st0['xt']
        st['Jt0'] = st0['Jt']
        return  st


    def fix_orientation(self, fv1=None):
        if fv1 is None:
            fv1 = self.fv1

        self.fv0.getEdges()
        self.closed = self.fv0.bdry.max() == 0
        v0 = self.fv0.surfVolume()
        if self.closed:
            if self.options['errorType'] == 'L2Norm' and v0 < 0:
                self.fv0.flipFaces()
                v0 = -v0
            if v0 < 0:
                self.fv0ori = -1
            else:
                self.fv0ori = 1
        else:
            self.fv0ori = 1
        if fv1:
            self.fv1ori = []
            for f1 in fv1:
                f1.getEdges()
                closed = self.closed and f1.bdry.max() == 0
                if closed:
                    v1 = f1.surfVolume()
                    if v0*v1 < 0:
                        f1.flipFaces()
                        v1 = -v1
                        if v1 < 0:
                            self.fv1ori.append(-1)
                        else:
                            self.fv1ori.append(1)
            else:
                self.fv0ori = 1
                self.fv1ori = 1
        else:
            self.fv1ori = 1
        #self.fv0Fine = surfaces.Surface(surf=self.fv0)
        logging.info('orientation: {0:d}'.format(self.fv0ori))


    # def dataTerm(self, _fvDef, fv1=None, fvInit = None, _lmk_def = None, lmk1 = None):
    #     obj = 0
    #     if fv1 is None:
    #         fv1 = self.fv1
    #     for k,s in enumerate(_fvDef):
    #         obj += super().dataTerm(s, {'fv1' = fv1[k])
    #         # if self.options['errorType'] == 'L2Norm':
    #         #     obj += surfaces.L2Norm(s, self.fv1[k].vfld) / (self.options['sigmaError'] ** 2)
    #         # else:
    #         #     obj += self.param.fun_obj(s, self.fv1[k], self.param.KparDist) / (self.options['sigmaError']**2)
    #     return obj


    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False, display=False):
        if var is None or 'Init' not in var:
            x0 = self.x0
        else:
            x0 = var['Init'][0]
            
        a00 = control['a00']
        a0 = control['a0']
        Afft0 = control['Afft0']
        rhot0 = control['rhot0']
        Afft = control['Afft']
        rhot = control['rhot']
            
        timeStep = self.options['timeStep']
        # dim2 = self.dim**2
        # A0 = [np.zeros([self.Tsize0, self.dim, self.dim]), np.zeros([self.Tsize0, self.dim])]
        # if self.affineDim > 0:
        #     for t in range(self.Tsize0):
        #         AB = np.dot(self.affineBasis, Afft0[t])
        #         A0[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A0[1][t] = AB[dim2:dim2+self.dim]
        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
        #print a0.shape
        st = self.solveStateEquation(control=control, options={'withJacobian':withJacobian})
        # if withJacobian:
        #     (xt0, at0, Jt0) = evol.secondOrderEvolution(x0, a00, self.options['KparDiff0'], timeStep,
        #                                                 withSpline=self.rhot0, withJacobian=True,
        #                                                 affine=A0)
        #     (xt, at, Jt) = evol.secondOrderEvolution(xt0[-1,...], a0, self.options['KparDiff'], timeStep,
        #                                              withSpline=self.rhot, withJacobian=True,
        #                                              affine=A)
        # else:
        #     (xt0, at0) = evol.secondOrderEvolution(x0, a00, self.options['KparDiff0'], timeStep,
        #                                            withSpline=rhot0, affine=A0)
        #     (xt, at) = evol.secondOrderEvolution(xt0[-1,...], a0, self.options['KparDiff'], timeStep,
        #                                          withSpline=rhot, affine=A)
        #print xt[-1, :, :]
        #print obj

        xt = st['xt']
        obj00 = 0.5 * (a00 * self.options['KparDiff0'].applyK(x0,a00)).sum() 
        obj0 = 0.5 * (a0 * self.options['KparDiff'].applyK(xt[0,...],a0)).sum()
        obj10 = 0
        obj1 = 0
        obj20 = 0
        obj2 = 0
        for t in range(self.Tsize0):
            rho = np.squeeze(rhot0[t, :, :])            
            obj10 += timeStep* self.options['controlWeight'] * (rho**2).sum()/2
            if self.affineDim > 0:
                obj20 +=  timeStep * (self.affineWeight.reshape(Afft0[t].shape) * Afft0[t]**2).sum()/2
        for t in range(self.Tsize):
            rho = np.squeeze(rhot[t, :, :])            
            obj1 += timeStep* self.options['controlWeight'] * (rho**2).sum()/2
            if self.affineDim > 0:
                obj2 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()/2
            #print xt.sum(), at.sum(), obj
        obj = obj1+obj2+obj0+obj10+obj20+obj00
        if display:
            logging.info(f'deformation terms: init {obj00:.4f}, rho {obj10:.4f}, aff {obj20:.4f}; init {obj0:.4f}, rho {obj1:.4f}, aff {obj2:.4f}')

        return obj, st


    def objectiveFun(self):
        if self.obj == None:
            self.obj, self.state = self.objectiveFunDef(self.control)
            self.obj0 = 0
            for k in range(self.nTarg):
                if self.options['errorType'] == 'L2Norm':
                    self.obj0 += L2Norm0(self.fv1[k]) / (self.options['sigmaError'] ** 2)
                else:   
                    self.obj0 += self.fun_obj0(self.fv1[k]) / (self.options['sigmaError']**2)
                #foo = surfaces.Surface(surf=self.fvDef[k])
                self.fvDef[k].updateVertices(self.state['xt'][self.jumpIndex[k], :, :])
                #foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return self.control
    
    def updateTry(self, dr, eps, objRef=None):
        controlTry = Control()
        for k in dr.keys():
            if self.isActive[k]:
                if dr[k] is not None:
                    controlTry[k] = self.control[k] - eps * dr[k]
            else:
                controlTry[k] = self.control[k]
        #print self.options['typeRegression']

        # if self.affineDim > 0 and self.options['typeRegression']=="affine":
        #     Afft0Try = self.Afft0 - eps * dr['aff0']
        #     AfftTry = self.Afft - eps * dr['aff']
        # else:
        #     Afft0Try = self.Afft0
        #     AfftTry = self.Afft
        # control = {'a00': a00Try, 'rhot0': rhot0Try, 'Afft0': Afft0Try,
        #            'a0': a0Try, 'rhot': rhotTry, 'Afft': AfftTry}
        objTry, stTry = self.objectiveFunDef(controlTry,  withTrajectory=True)
        objTry += self.obj0

        ff = [] 
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(stTry['xt'][self.jumpIndex[k], :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) or (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry



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

    def secondOrderCovector(self, x00, control, px1, pa1,  affine = (None, None)):
        a00 = control['a00'] 
        rhot0 = control['rhot0']
        a0 = control['a0']
        rhot = control['rhot']

        nTarg = len(px1)
        N = x00.shape[0]
        A = self.affB.getTransforms(control['Afft'])
        A0 = self.affB.getTransforms(control['Afft0'])
        dim = x00.shape[1]
        # if affine[1] is not None:
        #     aff_ = True
        #     A = affine[1][0]
        # else:
        #     aff_ = False
            
        T = self.Tsize
        # if isjump is None:
        #     isjump = np.zeros(T, dtype=bool)
        #     for t in range(T):
        #         if t%nTarg == 0:
        #             isjump[t] = True
    
        timeStep = self.options['timeStep']
        st = self.solveStateEquation(control=control, init_state=x00)
        xt0 = st['xt0']
        at0 = st['at0']
        xt = st['xt']
        at = st['at']
        # [xt0, at0] = evol.secondOrderEvolution(x00, a00, self.options['KparDiff0'], timeStep,
        #                                        withSpline=rhot0, affine=affine[0])
        # x0 = xt0[-1,...]
        # [xt, at] = evol.secondOrderEvolution(x0, a0, self.options['KparDiff'], timeStep, affine=affine[1],
        #                                      withSpline=rhot)

        pxt, pat = evol.secondOrderCovector(xt[0, :, :], at[0, :, :], px1, pa1, self.options['Kpardiff'],
                                       self.options['timeStep'], affine=A, withSpline=rhot, isjump=self.isjump,
                                       forwardState=(xt, at))
        # pxt = np.zeros([T+1, N, dim])
        # pxt[T, :, :] = px1[nTarg-1]
        # pat = np.zeros([T+1, N, dim])
        # pat[T, :, :] = pa1[nTarg-1]
        # jIndex = nTarg - 2
        # KparDiff = self.options['KparDiff']
        # for t in range(T):
        #     px = pxt[T-t, :, :]
        #     pa = pat[T-t, :, :]
        #     x = xt[T-t-1, :, :]
        #     a = at[T-t-1, :, :]
        #     #rho = np.squeeze(rhot[T-t-1, :, :])
        #
        #     if aff_:
        #         U = getExponential(timeStep * A[T-t-1])
        #         px_ = np.dot(px, U)
        #         Ui = la.inv(U)
        #         pa_ = np.dot(pa,Ui.T)
        #     else:
        #         px_ = px
        #         pa_ = pa
        #
        #     # a1 = np.concatenate((px_[np.newaxis,...], a[np.newaxis,...]))
        #     # a2 = np.concatenate((a[np.newaxis,...], px_[np.newaxis,...]))
        #     # zpx = KparDiff.applyDiffKT(x, px_, a) + KparDiff.applyDiffKT(x, a, px_) \
        #     #       - KparDiff.applyDDiffK11and12(x, a, a, pa_)
        #     zpx = KparDiff.applyDiffKT(x, px_, a, sym=True) - KparDiff.applyDDiffK11and12(x, a, a, pa_)
        #     zpa = KparDiff.applyK(x, px_) - KparDiff.applyDiffK1and2(x, pa_, a)
        #
        #     pxt[T-t-1, :, :] = px_ + timeStep * zpx
        #     pat[T-t-1, :, :] = pa_ + timeStep * zpa
        #     if isjump[T-t-1]:
        #         pxt[T-t-1, :, :] += px1[jIndex]
        #         pat[T-t-1, :, :] += pa1[jIndex]
        #         jIndex -= 1

        #####


        # if not(affine[0] is None):
        #     aff_ = True
        #     A0 = affine[0][0]
        # else:
        #     aff_ = False
        px01 = pxt[0,:,:] - self.options['KparDiff'].applyDiffKT(self.x0, control['a0'], control['a0'])
        pa01 = np.zeros([N, dim])
        pxt0, pat0 = evol.secondOrderCovector(x00, self.a0, px01, pa01, self.options['Kpardiff0'],
                                              self.options['timeStep'], affine=A0, withSpline=rhot0,
                                              forwardState=(xt0, at0))
        # T = self.Tsize0
        # pxt0 = np.zeros([T+1, N, dim])
        # pxt0[T, :, :] = pxt[0,:,:] - KparDiff.applyDiffKT(x0, a0, a0)
        # pat0 = np.zeros([T+1, N, dim])
        #
        # KparDiff = self.options['KparDiff0']
        # for t in range(T):
        #     px = pxt0[T-t, :, :]
        #     pa = pat0[T-t, :, :]
        #     x = xt0[T-t-1, :, :]
        #     a = at0[T-t-1, :, :]
        #     #rho = np.squeeze(rhot[T-t-1, :, :])
        #
        #     if aff_:
        #         U = getExponential(timeStep * A0[T-t-1])
        #         px_ = np.dot(px, U)
        #         Ui = la.inv(U)
        #         pa_ = np.dot(pa,Ui.T)
        #     else:
        #         px_ = px
        #         pa_ = pa
        #
        #     # a1 = np.concatenate((px_[np.newaxis,...], a[np.newaxis,...]))
        #     # a2 = np.concatenate((a[np.newaxis,...], px_[np.newaxis,...]))
        #     zpx = KparDiff.applyDiffKT(x, px_, a) + KparDiff.applyDiffKT(x, a, px_) - KparDiff.applyDDiffK11and12(x, a, a, pa_)
        #     zpa = KparDiff.applyK(x, px_) - KparDiff.applyDiffK1and2(x, pa_, a)
        #
        #     pxt0[T-t-1, :, :] = px_ + timeStep * zpx
        #     pat0[T-t-1, :, :] = pa_ + timeStep * zpa
        #
        return [[pxt0, pat0, xt0, at0], [pxt, pat, xt, at]]

    # Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
    def secondOrderGradient(self, x00, control, px1, pa1, controlWeight=1.0):
        a00 = control['a00'] 
        rhot0 = control['rhot0']
        a0 = control['a0']
        rhot = control['rhot']
        foo = self.secondOrderCovector(x00, control, px1, pa1)
        (pxt0, pat0, xt0, at0) = foo[0]
        A = self.affB.getTransforms(control['Afft'])
        A0 = self.affB.getTransforms(control['Afft0'])

        if A0 is not None:
            dA0 = np.zeros(A0[0].shape)
            db0 = np.zeros(A0[1].shape)
        else:
            dA0 = None
            db0 = None

        if A is not None:
            dA = np.zeros(A[0].shape)
            db = np.zeros(A[1].shape)
        else:
            dA = None
            db = None

        Tsize0 = self.Tsize0
        timeStep = self.options['timeStep']
        drhot0 = np.zeros(rhot0.shape)
        KparDiff = self.options['KparDiff0']
        if A0 is not None:
            for t in range(Tsize0):
                x = xt0[t, :, :]
                a = at0[t, :, :]
                rho = rhot0[t, :, :]
                px = pxt0[t+1, :, :]
                pa = pat0[t+1, :, :]
                zx = x + timeStep * KparDiff.applyK(x, a)
                za = a + timeStep * (-KparDiff.applyDiffKT(x, a, a) + rho)
                U = getExponential(timeStep * A0[0][t])
                #U = np.eye(dim) + timeStep * affine[0][k]
                Ui = la.inv(U)
                pa = np.dot(pa, Ui.T)
                za = np.dot(za, Ui)
                dA0[t,...] =  (gradExponential(timeStep*A0[0][t], px, zx)
                                - gradExponential(timeStep*A0[0][t], za, pa))
                drhot0[t,...] = rho*controlWeight - pa
            db0 = pxt0[1:Tsize0+1,...].sum(axis=1)
        else:
            for t in range(Tsize0):
                rho = rhot0[t, :, :]
                pa = pat0[t+1, :, :]
                drhot0[t,...] = rho*controlWeight - pa

        da00 = KparDiff.applyK(x00, a00) - pat0[0,...]
        
        
        (pxt, pat, xt, at) = foo[1]
        Tsize = self.Tsize
        drhot = np.zeros(rhot.shape)
        KparDiff = self.options['KparDiff']
        if A is not None:
            for t in range(Tsize):
                x = xt[t, :, :]
                a = at[t, :, :]
                rho = rhot[t, :, :]
                px = pxt[t+1, :, :]
                pa = pat[t+1, :, :]
                zx = x + timeStep * KparDiff.applyK(x, a)
                za = a + timeStep * (-KparDiff.applyDiffKT(x, a, a) + rho)
                U = getExponential(timeStep * A[0][t])
                Ui = la.inv(U)
                pa = np.dot(pa, Ui.T)
                za = np.dot(za, Ui)
                dA[t,...] =  (gradExponential(timeStep*A[0][t], px, zx)
                                - gradExponential(timeStep*A[0][t], za, pa))
                drhot[t,...] = rho*controlWeight - pa
            db = pxt[1:Tsize+1,...].sum(axis=1)
        else:
            for t in range(Tsize):
                rho = rhot[t, :, :]
                pa = pat[t+1, :, :]
                drhot[t,...] = rho*controlWeight - pa

        da0 = KparDiff.applyK(xt[0,...], a0) - pat[0,...]
        res0 = dict()
        res0['a0'] = da00
        res0['rhot'] = drhot0
        res0['dA'] = dA0
        res0['db'] = db0
        res0['xt'] = xt0
        res0['at'] = at0
        res0['pxt'] = pxt0
        res0['pat'] = pat0

        res = dict()
        res['a0'] = da0
        res['rhot'] = drhot
        res['dA'] = dA
        res['db'] = db
        res['xt'] = xt
        res['at'] = at
        res['pxt'] = pxt
        res['pat'] = pat

        return res, res0

        # if affine is None:
        #     if getCovector == False:
        #         return [[da00, drhot0, xt0, at0], [da0, drhot, xt, at]]
        #     else:
        #         return [[da00, drhot0, xt0, at0, pxt0, pat0], [da0, drhot, xt, at, pxt, pat]]
        # else:
        #     if getCovector == False:
        #         return [[da00, drhot0, dA0, db0, xt0, at0], [da0, drhot, dA, db, xt, at]]
        #     else:
        #         return [[da00, drhot0, dA0, db0, xt0, at0, pxt0, pat0], [da0, drhot, dA, db, xt, at, pxt, pat]]

    def getGradient(self, coeff=1.0, update=None):
        A = None
        A0 = None
        #logging.info('Computing gradient')
        if update is None:
            control = self.control
            # a0 = self.a0
            # a00 = self.a00
            # rhot = self.rhot
            # rhot0 = self.rhot0
            endPoint = self.fvDef
        else:
            eps = update[1]
            dr = update[0]
            control = Control()
            for k in dr.keys():
                if self.isActive[k]:
                    if dr[k] is not None:
                        control[k] = self.control[k] - eps * dr[k]
                else:
                    control[k] = self.control[k]
            # if self.options['typeRegression'] == 'spline':
            #     a00 = self.a00
            #     rhot0 = self.rhot0 - eps * dr['rhot0']
            #     a0 = self.a0
            #     rhot = self.rhot - eps * dr['rhot']
            # elif self.options['typeRegression'] == 'geodesic':
            #     a00 = self.a00 - eps * dr['a00']
            #     rhot0 = self.rhot0
            #     a0 = self.a0 - eps * dr['a0']
            #     rhot = self.rhot
            # elif self.options['typeRegression'] == "affine":
            #     a0 = self.a0
            #     rhot = self.rhot
            #     a00 = self.a00
            #     rhot0 = self.rhot0
            # else:
            #     a00 = self.a00 - eps * dr['a00']
            #     rhot0 = self.rhot0 - eps * dr['rhot0']
            #     a0 = self.a0 - eps * dr['a0']
            #     rhot = self.rhot - eps * dr['rhot']
            #
            # if self.affineDim > 0 and self.options['typeRegression'] == "affine":
            #     Afft0 = self.Afft0 - eps * dr['aff0']
            #     Afft = self.Afft - eps * dr['aff']
            # else:
            #     Afft0 = self.Afft0
            #     Afft = self.Afft

            # if len(update[0]['aff0']) > 0:
            #     A0 = self.affB.getTransforms(Afft0)
            # if len(update[0]['aff']) > 0:
            #     A = self.affB.getTransforms(Afft)

            # a0 = self.a0 - update[1] * update[0]['a0']
            # a00 = self.a0 - update[1] * update[0]['a00']
            # rhot = self.rhot - update[1] * update[0]['rhot']
            # rhot0 = self.rhot0 - update[1] * update[0]['rhot0']
            # if len(update[0]['aff0']) > 0:
            #     A0 = self.affB.getTransforms(self.Afft0 - update[1]*update[0]['aff0'])
            # if len(update[0]['aff']) > 0:
            #     A = self.affB.getTransforms(self.Afft - update[1]*update[0]['aff'])
            st = self.solveStateEquation(control=control)
            # (xt0, at0)  = evol.secondOrderEvolution(self.x0, a00, self.options['KparDiff0'],
            #                                         self.options['timeStep'], withSpline=rhot0)
            # (xt, at)  = evol.secondOrderEvolution(xt0[-1,:,:], a0, self.options['KparDiff'],
            #                                       self.options['timeStep'], withSpline=rhot)
            endPoint = []
            for k in range(self.nTarg):
                fvDef = surfaces.Surface(surf=self.fv0)
                fvDef.updateVertices(st['xt'][self.jumpIndex[k], :, :])
                endPoint.append(fvDef)

        if control['Afft'] is not None:
            A = self.affB.getTransforms(control['Afft'])
        if control['Afft0'] is not None:
            A0 = self.affB.getTransforms(control['Afft0'])

        px1 = self.endPointGradient(endPoint=endPoint)
        pa1 = []
        for k in range(self.nTarg):
            pa1.append(np.zeros(self.a0.shape))

        foo0, foo1 = self.secondOrderGradient(self.x0, control, px1, pa1,
                                              controlWeight=self.options['controlWeight'])
        grd = Control()
        # if self.options['typeRegression'] == 'affine':
        #     grd['a00'] = np.zeros(foo[0][0].shape)
        #     grd['rhot0'] = np.zeros(foo[0][1].shape)
        # else:
        #     grd['a00'] = foo[0][0] / coeff
        #     grd['rhot0'] = foo[0][1] / coeff
        
        if self.options['typeRegression'] == 'spline':
            grd['rhot'] = foo1['drot']/(coeff)
            grd['rhot0'] = foo0['drhot']/(coeff)
            #grd.rhot = foo[1]/(coeff*self.rhot.shape[0])
        elif self.options['typeRegression'] == 'geodesic':
            grd['a0'] = foo1['da0'] / coeff
            grd['a00'] = foo0['da0'] / coeff
        elif self.options['typeRegression'] == 'affine':
            pass
            # grd['a0'] = np.zeros(foo[1][0].shape)
            # grd['rhot'] = np.zeros(foo[1][1].shape)
            # grd['a00'] = np.zeros(foo[0][0].shape)
            # grd['rhot0'] = np.zeros(foo[0][1].shape)
        else:
            grd['rhot'] = foo1['drot']/(coeff)
            grd['rhot0'] = foo0['drhot']/(coeff)
            grd['a0'] = foo1['da0'] / coeff
            grd['a00'] = foo0['da0'] / coeff

        if self.affineDim > 0 and self.iter < self.affBurnIn:
            dim2 = self.dim ** 2
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            grd['Afft0'] = np.zeros(self.control['Afft0'].shape)
            dA0 = foo0['dA']
            db0 = foo0['db']
            dA = foo1['dA']
            db = foo1['db']
            grd['Afft0'] = np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.control['Afft0'])
            for t in range(self.Tsize0):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA0[t].reshape([dim2,1]), db0[t].reshape([self.dim, 1])]))
               grd['Afft0'][t] -=  dAff.reshape(grd['Afft0'][t].shape)
            grd['Afft0'] *= self.options['timeStep']/(self.coeffAff*coeff)
            grd['Afft'] = np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.control['Afft'])
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] *= self.options['timeStep']/(self.coeffAff*coeff)
        
        #print (grd.a00**2).sum(),(grd.rhot0**2).sum(),(grd.aff0**2).sum(),(grd.a0**2).sum(),(grd.rhot**2).sum(),(grd.aff**2).sum() 
        return grd

    # def addProd(self, dir1, dir2, beta):
    #     dr = Direction()
    #     for k in dir1.keys():
    #         if dir1[k] is not None:
    #             dr[k] = dir1[k] + beta * dir2[k]
    #     return dr
    #
    # def prod(self, dir1, beta):
    #     dr = Direction()
    #     for k in dir1.keys():
    #         if dir1[k] is not None:
    #             dr[k] = beta * dir1[k]
    #     return dr


    def randomDir(self):
        dirfoo = Control()
        if self.options['typeRegression'] == 'spline':
            dirfoo['a0'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot'] = np.random.randn(self.Tsize, self.npt, self.dim)
            dirfoo['a00'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot0'] = np.random.randn(self.Tsize0, self.npt, self.dim)
        elif self.options['typeRegression'] == 'geodesic':
            dirfoo['a0'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot'] = np.zeros([self.Tsize, self.npt, self.dim])
            dirfoo['a00'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot0'] = np.zeros([self.Tsize0, self.npt, self.dim])
        elif self.options['typeRegression'] == 'affine':
            dirfoo['a0'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot'] = np.zeros([self.Tsize, self.npt, self.dim])
            dirfoo['a00'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot0'] = np.zeros([self.Tsize0, self.npt, self.dim])
        else:
            dirfoo['a0'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot'] = np.random.randn(self.Tsize, self.npt, self.dim)
            dirfoo['a00'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot0'] = np.random.randn(self.Tsize0, self.npt, self.dim)


        if self.iter < self.affBurnIn:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
            dirfoo['Afft0'] = np.random.randn(self.Tsize0, self.affineDim)
        else:
            dirfoo['Afft'] = np.zeros((self.Tsize, self.affineDim))
            dirfoo['Afft0'] = np.zeros((self.Tsize0, self.affineDim))
        return dirfoo

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        gg = g1['rhot']
        gga = g1['a0']
        uu = g1['Afft']
        ll = 0
        for gr in g2:
            ggOld = gr['rhot']
            res[ll]  = (ggOld*gg).sum()*self.options['timeStep']
            res[ll] += (gr['a0'] * gga).sum()
            res[ll] += (uu * gr['Afft']).sum() * self.coeffAff
            ll = ll+1

        gg = g1['rhot0']
        gga = g1['a00']
        uu = g1['Afft0']
        ll = 0
        for gr in g2:
            ggOld = gr['rhot0']
            res[ll] += (ggOld*gg).sum()*self.options['timeStep']
            res[ll] += (gr['a00'] * gga).sum()
            res[ll] += (uu * gr['Afft0']).sum() * self.coeffAff
            ll = ll+1

        return res

    # def acceptVarTry(self):
    #     self.obj = self.objTry
    #     self.control =
    #     self.a0 = np.copy(self.a0Try)
    #     self.rhot = np.copy(self.rhotTry)
    #     self.Afft = np.copy(self.AfftTry)
    #     self.a00 = np.copy(self.a00Try)
    #     self.rhot0 = np.copy(self.rhot0Try)
    #     self.Afft0 = np.copy(self.Afft0Try)
    #     #print self.at

    def endOfIteration(self, forceSave = False):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.options['typeRegression'] = self.typeRegressionSave
            self.affine = 'none'
            #self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0):
            logging.info('Saving surfaces...')
            # control = {'a00': self.a00, 'rhot0': self.rhot0, 'Afft0':self.Afft0,
            #            'a0': self.a0, 'rhot': self.rhot, 'Afft':self.Afft}
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True, display=self.options['verb'])
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.state['xt'][self.jumpIndex[k], :, :]))
            # dim2 = self.dim**2
            # if self.affineDim > 0:
            #     A0 = getTransforms(self.control['Afft0'])
            #     A = getTransforms(self.control['Afft'])
            #     # A0 = [np.zeros([self.Tsize0, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            #     # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            #     # for t in range(self.Tsize0):
            #     #     AB = np.dot(self.affineBasis, self.Afft0[t])
            #     #     A0[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
            #     #     A0[1][t] = AB[dim2:dim2+self.dim]
            #     # for t in range(self.Tsize):
            #     #     AB = np.dot(self.affineBasis, self.Afft[t])
            #     #     A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
            #     #     A[1][t] = AB[dim2:dim2+self.dim]
            # else:
            #     A0 = None
            #     A = None
                    
            # (xt0, at0, ft0)  = evol.secondOrderEvolution(self.x0, self.a00, self.options['KparDiff0'],
            #                                              self.options['timeStep'], affine=A0,
            #                                                withPointSet = self.fv0.vertices, withSpline=self.rhot0)
            # (xt, at, ft, Jt)  = evol.secondOrderEvolution(xt0[-1,...], self.a0, self.options['KparDiff'],
            #                                               self.options['timeStep'], affine=A, withSpline=self.rhot,
            #                                                withPointSet = ft0[-1,...], withJacobian=True)

            if self.saveCorrected:
                f = surfaces.Surface(surf=self.fv0)
                X0 = self.affB.integrateFlow(self.control['Afft0'])
                X = self.affB.integrateFlow(self.control['Afft'])
                displ = np.zeros(self.x0.shape[0])
                at0Corr = np.zeros(self.state['at0'].shape)
                for t in range(self.Tsize0+1):
                    R0 = X0[0][t,...]
                    U0 = la.inv(R0)
                    b0 = X0[1][t,...]
                    yyt = np.dot(self.state['xt0'][t,...] - b0, U0.T)
                    zt = np.dot(self.state['xt0'][t,...] - b0, U0.T)
                    if t < self.Tsize0:
                        a = np.dot(self.state['at0'][t,...], R0)
                        at0Corr[t,...] = a
                        vt = self.options['KparDiff0'].applyK(yyt, a, firstVar=zt)
                        vt = np.dot(vt, U0.T)
                    f.updateVertices(zt)
                    vf = surfaces.vtkFields()
                    vf.scalars.append('Jacobian')
                    vf.scalars.append(displ)
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity')
                    vf.vectors.append(vt)
                    f.saveVTK2(self.options['outputDir'] +'/'+self.options['saveFile']+'Corrected'+str(t)+'.vtk', vf)

                dt = 1.0 /self.Tsize0
                atCorr = np.zeros(self.state['at'].shape)
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t,...])
                    yyt = np.dot(self.state['xt'][t,...] - b0, U0.T)
                    yyt = np.dot(yyt - X[1][t, ...], U.T)
                    zt = np.dot(self.state['xt'][t,...] - b0, U0.T)
                    zt = np.dot(zt - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        a = np.dot(self.state['at'][t,...], R0)
                        a = np.dot(a, X[0][t,...])
                        atCorr[t,...] = a
                        vt = self.options['KparDiff'].applyK(yyt, a, firstVar=zt)
                        vt = np.dot(vt, U.T)
                    f.updateVertices(zt)
                    vf = surfaces.vtkFields()
                    vf.scalars.append('Jacobian')
                    vf.scalars.append(self.state['Jt'][t, :])
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity')
                    vf.vectors.append(vt)
                    nu = self.fv0ori*f.computeVertexNormals()
                    displ += dt * (vt*nu).sum(axis=1)
                    f.saveVTK2(self.options['outputDir'] +'/'+self.options['saveFile']+'Corrected'+str(t+self.Tsize0)+'.vtk', vf)
#                (foo,zt) = evol.landmarkDirectEvolutionEuler(self.x0, atCorr, self.options['KparDiff'], withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.options['outputDir'] +'/'+self.saveFile+'CorrectedCheck'+str(t)+'.vtk')
#                (foo,foo2,zt) = evol.secondOrderEvolution(self.x0, atCorr[0,...], self.rhot, self.options['KparDiff'], withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.options['outputDir'] +'/'+self.saveFile+'CorrectedCheckBis'+str(t)+'.vtk')
                 

                for k,fv in enumerate(self.fv1):
                    f = surfaces.Surface(surf=fv)
                    U = la.inv(X[0][self.jumpIndex[k]])
                    yyt = np.dot(f.vertices - b0, U0.T)
                    yyt = np.dot(yyt - X[1][self.jumpIndex[k], ...], U.T)
                    f.updateVertices(yyt)
                    f.saveVTK(self.options['outputDir'] +'/Target'+str(k)+'Corrected.vtk')
            
            fvDef = surfaces.Surface(surf=self.fv0)
            AV0 = fvDef.computeVertexArea()
            nu = self.fv0ori*self.fv0.computeVertexNormals()
            #v = self.v[0,...]
            displ = np.zeros(self.npt)
            # dt = 1.0 /self.Tsize
            v = self.options['KparDiff0'].applyK(self.x0, self.state['at0'][0,...])
            for kk in range(self.Tsize0+1):
                fvDef.updateVertices(np.squeeze(self.state['xt0'][kk, :, :]))
                AV = fvDef.computeVertexArea()
                AV = AV[0]/AV0[0]
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(self.state['Jt0'][kk, :])
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(self.state['Jt0'][kk, :]/AV)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if kk < self.Tsize:
                    v = self.options['KparDiff0'].applyK(self.state['xt0'][kk,...], self.state['at0'][kk,...])

                vf.vectors.append('velocity')
                vf.vectors.append(np.copy(v))
                fvDef.saveVTK2(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk)+'.vtk', vf)

            #dt = 1.0 /self.Tsize
            v = self.options['KparDiff'].applyK(self.state['xt'][0,...], self.state['at'][0,...])
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(self.state['xt'][kk, :, :]))
                AV = fvDef.computeVertexArea()
                AV = AV[0]/AV0[0]
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(self.state['Jt'][kk, :])
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(self.state['Jt'][kk, :]/AV)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if self.Tsize > 0:
                    displ += (v*nu).sum(axis=1) / self.Tsize
                if kk < self.Tsize:
                    nu = self.fv0ori*fvDef.computeVertexNormals()
                    v = self.options['KparDiff'].applyK(self.state['xt'][kk,...], self.state['at'][kk,...])
                    #v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                vf.vectors.append('velocity')
                vf.vectors.append(np.copy(v))
                fvDef.saveVTK2(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk+self.Tsize0)+'.vtk', vf)
        else:

            obj1, self.state= self.objectiveFunDef(self.control, withTrajectory=True, display=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.state['xt'][self.jumpIndex[k], :, :]))


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
