import os
from copy import deepcopy
import logging
from . import curves, curveDistancess as cd
from . import pointSets
from .pointSetMatching import PointSetMatching, Control, State
from . import conjugateGradient as cg, grid, pointEvolution as evol, bfgs
from .affineBasis import *
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import partial


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class CurveMatchingParam(matchingParam.MatchingParam):
#     def __init__(self, timeStep = .1, algorithm = 'bfgs', Wolfe=True, KparDiff = None, KparDist = None,
#                  sigmaKernel = 6.5, sigmaDist=2.5, sigmaError=1.0,
#                  errorType = 'measure', typeKernel='gauss', internalCost=None):
#         super().__init__(timeStep=timeStep, algorithm = algorithm, Wolfe=Wolfe,
#                         KparDiff = KparDiff, KparDist = KparDist, sigmaError=sigmaError,
#                         errorType = errorType)
#           
#         self.internalCost = internalCost
#         self.errorType = errorType
                                         

# class Direction:
#     def __init__(self):
#         self.diff = []
#         self.aff = []


## Main class for curve matching
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
class CurveMatching(PointSetMatching):
    # def __init__(self, Template=None, Target=None, options = None):
    #     super().__init__(Template=Template, Target=Target, options=options)
                # fileTempl=None, fileTarg=None, param=None,
                #  maxIter=1000, regWeight = 1.0, affineWeight = 1.0,
                #  verb=True, gradLB = 0.001, saveRate=10, saveTrajectories=False,
                #  rotWeight = None, scaleWeight = None, transWeight = None, internalWeight=-1.0, 
                #  testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.', pplot=True):

            
        # self.fvDef = curves.Curve(curve=self.fv0)
        # self.iter = 0
        # self.maxIter = maxIter
        # self.verb = verb
        # self.options['testGradient'] = testGradient
        # self.regweight = regWeight
        # self.internalWeight = internalWeight
        # self.affine = affine
        # self.affB = AffineBasis(self.dim, affine)
        # self.affineDim = self.affB.affineDim
        # self.affineBasis = self.affB.basis
        # self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        # if (len(self.affB.rotComp) > 0) & (rotWeight != None):
        #     self.affineWeight[self.affB.rotComp] = rotWeight
        # if (len(self.affB.simComp) > 0) & (scaleWeight != None):
        #     self.affineWeight[self.affB.simComp] = scaleWeight
        # if (len(self.affB.transComp) > 0) & (transWeight != None):
        #     self.affineWeight[self.affB.transComp] = transWeight
        # 
        # if param==None:
        #     self.param = CurveMatchingParam()
        # else:
        #     self.param = param

        # if self.param.algorithm == 'bfgs':
        #      self.euclideanGradient = True
        # else:
        #     self.euclideanGradient = False
        # 
        # self.set_fun(self.param.errorType)

            
        # self.obj = None
        # self.objTry = None
        # self.gradCoeff = self.fv0.vertices.shape[0]
        # self.options['saveFile'] = saveFile
        # self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        # self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        # self.gradLB = gradLB
        # self.saveRate = saveRate 
        # self.options['saveTrajectories'] = saveTrajectories
        #self.pplot = pplot
        # if self.options['pplot']:

#        om = np.random.uniform(-1,1,[1,self.fv0.vertices.shape[1]])
#        v1 = np.cross(om, self.fv0.vertices)
#        #om = np.random.uniform(-1,1,[1,3])
#        #v11 = np.cross(om[np.newaxis,:], fv11.vertices)
#        dtest = self.internalCost(self.fv0, v1)
#        print 'dtest= ', dtest

    def set_parameters(self):
        super().set_parameters()
        self.gradEps = -1
        self.lineSearch = "Weak_Wolfe"
        self.randomInit = False
        self.iter = 0
        self.reset = True

        if self.options['internalCost'] == 'h1':
            self.internalCost = partial(curves.normGrad, weight=0.0)
            self.internalCostGrad = partial(curves.diffNormGrad, weight=0.0)
        elif self.options['internalCost'] == 'h1Alpha':
            self.internalCost = curves.h1AlphaNorm
            self.internalCostGrad = curves.diffH1Alpha
            #self.internalWeight *= self.fv0.length()/(self.fv0.component.max()+1)
        elif self.options['internalCost'] == 'h1AlphaInvariant':
            self.internalCost = curves.h1AlphaNormInvariant
            self.internalCostGrad = curves.diffH1AlphaInvariant
            #self.internalWeight *= self.fv0.length()/(self.fv0.component.max()+1)
        elif self.options['internalCost'] == 'h1Invariant':
            if self.fv0.vertices.shape[1] == 2:
                self.internalCost = curves.normGradInvariant
                self.internalCostGrad = curves.diffNormGradInvariant
            else:
                self.internalCost = curves.normGradInvariant3D
                self.internalCostGrad = curves.diffNormGradInvariant3D
        else:
            self.internalCost = None


    def set_template_and_target(self, Template, Target, misc=None):
        if Template is None:
            logging.info('Please provide a template curve')
            return
        else:
            self.fv0 = curves.Curve(curve=Template)


        if Target is None:
            logging.info('Please provide a target curve')
            return
        else:
            self.fv1 = curves.Curve(curve=Target)

        self.fv0.saveVTK(self.outputDir + '/Template.vtk')
        self.fv1.saveVTK(self.outputDir + '/Target.vtk')
        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]

    def initialize_variables(self):
        self.x0 = self.fv0.vertices
        self.fvDef = deepcopy(self.fv0)
        self.Tsize = int(round(1.0/self.options['timeStep']))

        self.control['at'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.controlTry['at'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.state['xt'] = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])

        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        if self.dim == 2:
            xmin = min(self.fv0.vertices[:,0].min(), self.fv1.vertices[:,0].min())
            xmax = max(self.fv0.vertices[:,0].max(), self.fv1.vertices[:,0].max())
            ymin = min(self.fv0.vertices[:,1].min(), self.fv1.vertices[:,1].min())
            ymax = max(self.fv0.vertices[:,1].max(), self.fv1.vertices[:,1].max())
            dx = 0.01*(xmax-xmin)
            dy = 0.01*(ymax-ymin)
            dxy = min(dx,dy)
            #print xmin,xmax, dxy
            [x,y] = np.mgrid[(xmin-10*dxy):(xmax+10*dxy):dxy, (ymin-10*dxy):(ymax+10*dxy):dxy]
            #print x.shape
            self.gridDef = grid.Grid(gridPoints=[x, y])
            self.gridxy = np.copy(self.gridDef.vertices)


    def set_fun(self, errorType, vfun = None):
        self.options['errorType'] = errorType
        if errorType == 'current':
            print('Running Current Matching')
            weight = None
            self.fun_obj0 = partial(cd.currentNorm0, KparDist=self.options['KparDist'], weight=weight)
            self.fun_obj = partial(cd.currentNormDef, KparDist=self.options['KparDist'], weight=weight)
            self.fun_objGrad = partial(cd.currentNormGradient, KparDist=self.options['KparDist'], weight=weight)
            # self.fun_obj0 = curves.currentNorm0
            # self.fun_obj = curves.currentNormDef
            # self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            print('Running Measure Matching')
            self.fun_obj0 = partial(cd.measureNorm0, KparDist=self.options['KparDist'])
            self.fun_obj = partial(cd.measureNormDef,KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(cd.measureNormGradient,KparDist=self.options['KparDist'])
        elif errorType=='varifold':
            self.fun_obj0 = partial(cd.varifoldNorm0, KparDist=self.options['KparDist'], weight=1.)
            self.fun_obj = partial(cd.varifoldNormDef, KparDist=self.options['KparDist'], weight=1.)
            self.fun_objGrad = partial(cd.varifoldNormGradient, KparDist=self.options['KparDist'], weight=1.)
        elif errorType == 'varifoldComponent':
            self.fun_obj0 = partial(cd.varifoldNormComponent0, KparDist=self.options['KparDist'])
            self.fun_obj = partial(cd.varifoldNormComponentDef, KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(cd.varifoldNormComponentGradient, KparDist=self.options['KparDist'])
        elif errorType == 'landmarks':
            self.fun_obj0 = cd.L2Norm0
            self.fun_obj = cd.L2NormDef
            self.fun_objGrad = cd.L2NormGradient
            if self.fv1.vertices.shape[0] != self.fvDef.vertices.shape[0]:
                sdef = self.fvDef.arclength()
                s1 = self.fv1.arclength()
                x1 = np.zeros(self.fvDef.vertices.shape)
                x1[:,0] = np.interp(sdef, s1, self.fv1.vertices[:,0])
                x1[:,1] = np.interp(sdef, s1, self.fv1.vertices[:,1])
                self.fv1 = curves.Curve(curve=(self.fvDef.faces,x1))
            bestk = 0
            minL2 = cd.L2Norm(self.fvDef, self.fv1)
            fvTry = curves.Curve(curve=self.fv1)
            for k in range(1,self.fv1.vertices.shape[0]):
                fvTry.updateVertices(np.roll(self.fv1.vertices, k, axis=0))
                L2 = cd.L2Norm(self.fvDef, fvTry)
                if L2 < minL2:
                    bestk = k
                    minL2 = L2
            if bestk>0:
                self.fv1.updateVertices(np.roll(self.fv1.vertices, bestk, axis=0))

        else:
            print('Unknown error Type: ', self.options['errorType'])


    def initial_plot(self):
        self.cmap = cm.get_cmap('hsv', self.fvDef.faces.shape[0])
        self.cmap1 = cm.get_cmap('hsv', self.fv1.faces.shape[0])
        self.lw = 3
        if self.dim == 2:
            fig = plt.figure(2)
            fig.clf()
            ax = fig.gca()
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 1], color='r', linewidth=self.lw)
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                        color=[0, 0, 1], linewidth=self.lw)
            plt.axis('equal')
            plt.axis('off')
            self.axis = plt.axis()
            plt.savefig(self.outputDir + '/Template_Target.png')
            fig = plt.figure(3)
            fig.clf()
            ax = fig.gca()
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 1], color=self.cmap(kf), linewidth=self.lw)
            plt.axis('equal')
        elif self.dim == 3:
            fig = plt.figure(2)
            fig.clf()
            ax = fig.gca(projection='3d')
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                        self.fv1.vertices[self.fv1.faces[kf, :], 2], color=[0, 0, 1])
            fig = plt.figure(3)
            fig.clf()
            ax = fig.gca(projection='3d')
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 2], color=[1, 0, 0], marker='*')
            # plt.axis('equal')
        plt.pause(0.1)

    def dataTerm(self, _fvDef, var=None):
        obj = self.fun_obj(_fvDef, self.fv1) / (self.options['sigmaError']**2)
        return obj

    def objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False):
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

        timeStep = 1.0/self.Tsize
        # A = self.affB.getTransforms(Afft)
        st = self.solveStateEquation(control, init_state=x0, kernel=kernel, options={'withJacobian':withJacobian})
        # if withJacobian:
        #     (xt,Jt)  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A, withJacobian=True)
        #     st['Jt'] = Jt
        # else:
        #     xt  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A)
        # st['xt'] = xt
        #print xt[-1, :, :]
        #print obj
        obj=0
        xt = st['xt']
        foo = curves.Curve(curve=self.fv0)
        for t in range(self.Tsize):
            z = xt[t, :, :]
            a = at[t, :, :]
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            ra = kernel.applyK(z, a)
            obj = obj + regWeight*timeStep*np.multiply(a, (ra)).sum()
            if self.internalCost:
                foo.updateVertices(z)
                obj += self.options['internalWeight']*self.internalCost(foo, ra)*timeStep
            if self.affineDim > 0:
                obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withJacobian or withTrajectory:
            return obj, st
        else:
            return obj

    def _objectiveFun(self, control, withTrajectory = False):
        (obj, st) = self.objectiveFunDef(control, withTrajectory=True)
        self.fvDef.updateVertices(st['xt'][-1, :, :])
        obj0 = self.dataTerm(self.fvDef)

        if withTrajectory:
            return obj+obj0, st
        else:
            return obj+obj0

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError']**2)
            (self.obj, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.state['xt'][-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return self.control

    def makeTryInstance(self, state):
        ff = curves.Curve(curve=self.fvDef)
        ff.updateVertices(state['xt'][-1,:,:])
        return ff

    # def updateTry(self, dir, eps, objRef=None):
    #     objTry = self.obj0
    #     atTry = self.at - eps * dir['diff']
    #     if self.affineDim > 0:
    #         AfftTry = self.Afft - eps * dir['aff']
    #     else:
    #         AfftTry = self.Afft
    #     foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
    #     objTry += foo[0]
    # 
    #     ff = curves.Curve(curve=self.fvDef)
    #     ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
    #     objTry += self.dataTerm(ff)
    #     if np.isnan(objTry):
    #         print('Warning: nan in updateTry')
    #         return 1e500
    # 
    #     if (objRef == None)  or (objTry < objRef):
    #         self.atTry = atTry
    #         self.objTry = objTry
    #         self.AfftTry = AfftTry
    # 
    #     return objTry

    def hamiltonianCovector(self, px1, control = None):
        x0 = self.x0
        if control is None:
            control = self.control
        KparDiff = self.options['KparDiff']
        N = x0.shape[0]
        dim = x0.shape[1]
        M = control['at'].shape[0]
        timeStep = 1.0/M
        st = self.solveStateEquation(control)
        # xt = evol.landmarkDirectEvolutionEuler(x0, control['at'], KparDiff, affine=affine)

        xt = st['xt']
        affine = self.affB.getTransforms(control['Afft'])
        if affine is not None:
            A0 = affine[0]
            A = np.zeros([M,dim,dim])
            for k in range(A0.shape[0]):
                A[k,...] = getExponential(timeStep*A0[k]) 
        else:
            A = None
            # A = np.zeros([M,dim,dim])
            # for k in range(M-1):
            #     A[k,...] = np.eye(dim)
                
        pxt = np.zeros([M+1, N, dim])
        pxt[M, :, :] = px1
        foo = curves.Curve(curve=self.fv0)
        for t in range(M):
            px = pxt[M-t, :, :]
            z = xt[M-t-1, :, :]
            a = control['at'][M-t-1, :, :]
            if self.internalCost:
                foo.updateVertices(z)
                v = KparDiff.applyK(z, a)
                grd = self.internalCostGrad(foo, v)
                Lv =  grd[0]
                DLv = self.options['internalWeight']*grd[1]
                zpx = KparDiff.applyDiffKT(z, px, a, regweight=self.options['regWeight'], lddmm=True,
                                                      extra_term = -self.options['internalWeight']*Lv) - DLv
            else:
                zpx = KparDiff.applyDiffKT(z, px, a, regweight=self.options['regWeight'], lddmm=True)
                
            if affine is not None:
                pxt[M-t-1, :, :] = np.dot(px, A[M-t-1]) + timeStep * zpx
            else:
                pxt[M-t-1, :, :] = px + timeStep * zpx
        return pxt, xt


    def hamiltonianGradient(self, px1, control = None, kernel = None, regWeight=None, fv0=None):
        if control is None:
            control = self.control
        if regWeight is None:
            regWeight = self.options['regWeight']
        if fv0 is None:
            fv0 = self.fv0
            x0 = self.x0
        else:
            x0 = self.fv0.vertices
        if kernel is None:
            kernel  = self.options['KparDiff']

        affine = self.affB.getTransforms(control['Afft'])
        if not self.internalCost:
            return evol.landmarkHamiltonianGradient(x0, control['at'], px1, kernel, regWeight, affine=affine,
                                                    getCovector=True)
                                                    
        foo = curves.Curve(curve=fv0)
        (pxt, xt) = self.hamiltonianCovector(px1, control=control)
        #at = self.at
        dat = np.zeros(control['at'].shape)
        timeStep = 1.0/control['at'].shape[0]
        if affine is not None:
            A = affine[0]
            dA = np.zeros(affine[0].shape)
            db = np.zeros(affine[1].shape)
        else:
            A = None
            dA = None
            db = None

        for k in range(control['at'].shape[0]):
            z = np.squeeze(xt[k,...])
            foo.updateVertices(z)
            a = np.squeeze(control['at'][k, :, :])
            px = np.squeeze(pxt[k+1, :, :])
            #print 'testgr', (2*a-px).sum()
            v = kernel.applyK(z,a)
            if self.internalCost:
                Lv = self.internalCostGrad(foo, v, variables='phi') 
                dat[k, :, :] = 2*regWeight*a-px + self.options['internalWeight'] * Lv
            else:
                dat[k, :, :] = 2*regWeight*a-px

            if affine is not None:
                dA[k] = gradExponential(A[k]*timeStep, px, xt[k]) #.reshape([self.dim**2, 1])/timeStep
                db[k] = pxt[k+1].sum(axis=0) #.reshape([self.dim,1]) 
    
        if affine is None:
            return dat, xt, pxt
        else:
            return dat, dA, db, xt, pxt


    def endPointGradient(self, endPoint = None):
        if endPoint is None:
            endPoint = self.fvDef
        px = self.fun_objGrad(endPoint, self.fv1)
        return px / self.options['sigmaError']**2

    def testEndpointGradient(self):
        ff = curves.Curve(curve=self.fvDef)
        dff = np.random.normal(size=ff.vertices.shape)
        eps = 1e-6
        ff.updateVertices(ff.vertices+eps*dff)
        c1 = self.dataTerm(ff)
        ff.updateVertices(ff.vertices-2*eps*dff)
        c2 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c2)/(2*eps), (grd*dff).sum()) )

    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]
        # A = self.affB.getTransforms(control['Afft'])
        # at = self.at - update[1] * update[0]['diff']
        #
        # Afft = self.Afft - update[1] * update[0]['aff']
        # if len(update[0]['aff']) > 0:
        #     A = self.affB.getTransforms(Afft)
        # else:
        #     A = None
        st = self.solveStateEquation(control)
        # xt = evol.landmarkDirectEvolutionEuler(self.x0, control['at'], self.options['KparDiff'], affine=A)
        xt = st['xt']
        endPoint = curves.Curve(curve=self.fv0)
        endPoint.updateVertices(xt[-1, :, :])
        # st = State()
        # st['xt'] = xt

        return control, st, endPoint

    # def getGradient(self, coeff=1.0, update=None):
    #     if update is None:
    #         at = self.at
    #         endPoint = self.fvDef
    #         A = self.affB.getTransforms(self.Afft)
    #     else:
    #         A = self.affB.getTransforms(self.Afft - update[1]*update[0].aff)
    #         at = self.at - update[1] *update[0].diff
    #         xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine=A)
    #         endPoint = curves.Curve(curve=self.fv0)
    #         endPoint.updateVertices(xt[-1, :, :])
    # 
    #     px1 = -self.endPointGradient(endPoint=endPoint)
    #     # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
    #     dim2 = self.dim**2
    #     # if self.affineDim > 0:
    #     #     for t in range(self.Tsize):
    #     #         AB = np.dot(self.affineBasis, self.Afft[t])
    #     #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
    #     #         A[1][t] = AB[dim2:dim2+self.dim]
    #     foo = self.hamiltonianGradient(px1, at=at, affine=A)
    #     grd = Direction()
    #     if self.euclideanGradient:
    #         grd.diff = np.zeros(foo[0].shape)
    #         for t in range(self.Tsize):
    #             z = self.xt[t, :, :]
    #             grd.diff[t,:,:] = self.param.KparDiff.applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
    #     else:
    #         grd.diff = foo[0]/(coeff*self.Tsize)
    #     grd.aff = np.zeros(self.Afft.shape)
    #     if self.affineDim > 0:
    #         dA = foo[1]
    #         db = foo[2]
    #         #dAfft = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft)
    #         grd.aff = 2 * self.Afft
    #         for t in range(self.Tsize):
    #            dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
    #            grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
    #         grd.aff /= (coeff*self.Tsize)
    #         #            dAfft[:,0:self.dim**2]/=100
    #     return grd
    # 


    # def addProd(self, dir1, dir2, beta):
    #     dir = Direction()
    #     dir.diff = dir1.diff + beta * dir2.diff
    #     dir.aff = dir1.aff + beta * dir2.aff
    #     return dir
    # 
    # def prod(self, dir1, beta):
    #     dir = Direction()
    #     dir.diff = beta * dir1.diff
    #     dir.aff = beta * dir1.aff
    #     return dir
    # 
    # def copyDir(self, dir0):
    #     dir = Direction()
    #     dir.diff = np.copy(dir0.diff)
    #     dir.aff = np.copy(dir0.aff)
    # 
    #     return dir
    # 
    # 
    # def randomDir(self):
    #     dirfoo = Direction()
    #     dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
    #     dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
    #     return dirfoo
    # 
    # def dotProduct(self, g1, g2):
    #     res = np.zeros(len(g2))
    #     #dim2 = self.dim**2
    #     for t in range(self.Tsize):
    #         z = np.squeeze(self.xt[t, :, :])
    #         gg = np.squeeze(g1.diff[t, :, :])
    #         u = self.param.KparDiff.applyK(z, gg)
    #         uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
    #         ll = 0
    #         for gr in g2:
    #             ggOld = np.squeeze(gr.diff[t, :, :])
    #             res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
    #             if self.affineDim > 0:
    #                 #print np.multiply(np.multiply(g1[1][t], gr[1][t]), self.affineWeight).shape
    #                 res[ll] += np.multiply(uu, gr.aff[t]).sum()
    #                 #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
    #             ll = ll + 1
    #     return res
    # 
    # 
    # def dotProduct_euclidean(self, g1, g2):
    #     res = np.zeros(len(g2))
    #     for t in range(self.Tsize):
    #         u = np.squeeze(g1.diff[t, :, :])
    #         uu = (g1.aff[t]*self.affineWeight.reshape(g1.aff[t].shape))
    #         ll = 0
    #         for gr in g2:
    #             ggOld = np.squeeze(gr.diff[t, :, :])
    #             res[ll]  += (ggOld*u).sum()
    #             if self.affineDim > 0:
    #                 res[ll] += (uu*gr.aff[t]).sum()
    #             ll = ll + 1
    #     return res


    # def acceptVarTry(self):
    #     self.obj = self.objTry
    #     self.at = np.copy(self.atTry)
    #     self.Afft = np.copy(self.AfftTry)

    def endOfIteration(self, endP=False):
        (obj1, self.state) = self.objectiveFunDef(self.control, withTrajectory=True, withJacobian=True)
        #logging.info('Distance {0:f}'.format(np.sqrt(obj1/2)))
        self.iter += 1
        if self.options['testGradient']:
            self.testEndpointGradient()

        if self.internalCost and self.options['testGradient']:
            Phi = np.random.normal(size=self.x0.shape)
            dPhi1 = np.random.normal(size=self.x0.shape)
            dPhi2 = np.random.normal(size=self.x0.shape)
            eps = 1e-10
            fv22 = curves.Curve(curve=self.fvDef)
            fv22.updateVertices(self.fvDef.vertices+eps*dPhi2)
            e0 = self.internalCost(self.fvDef, Phi)
            e1 = self.internalCost(self.fvDef, Phi+eps*dPhi1)
            e2 = self.internalCost(fv22, Phi)
            grad = self.internalCostGrad(self.fvDef, Phi)
            logging.info(f'Laplacian: {(e1-e0)/eps:.5f} {(grad[0]*dPhi1).sum():.5f};  '+
                         f'Gradient: {(e2-e0)/eps:.5f} {(grad[1]*dPhi2).sum():.5f}\n')

        if self.saveRate > 0 and self.iter%self.saveRate==0:
            if self.dim==2:
                # A = self.affB.getTransforms(self.control['Afft'])
                st = self.solveStateEquation(init_state=self.fv0.vertices, options={'withPointSet':self.gridxy})
                # (xt,yt) = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, self.control['at'], self.options['KparDiff'],
                #                                             affine=A, withPointSet=self.gridxy)
                xt = st['xt']
                yt = st['yt']
            if self.options['saveTrajectories']:
                pointSets.saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curves.vtk', self.state['xt'])

            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.state['xt'][kk, :, :]))
                if self.pplot and self.dim==2:
                    fig = plt.figure(4)
                    fig.clf()
                    ax = fig.gca()
                    for kf in range(self.fvDef.faces.shape[0]):
                        #ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
                        ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1],
                                color=self.cmap(kf), linewidth=self.lw)
                    plt.axis('off')
                    plt.axis('equal')
                    plt.axis(self.axis)
                    plt.savefig(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.png')
                    fig.canvas.flush_events()
                    # plt.pause(0.01)
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                self.fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk')
                if self.dim == 2:
                    self.gridDef.vertices = np.copy(yt[kk, :, :])
                    self.gridDef.saveVTK(self.outputDir +'/grid'+str(kk)+'.vtk')
        else:
            self.fvDef.updateVertices(np.squeeze(self.state['xt'][self.Tsize, :, :]))

        if self.pplot:
            if self.dim==2:
                # fig = plt.figure(4)
                # fig.clf()
                if self.saveRate == 0 or self.iter % self.saveRate != 0:
                    fig = plt.figure(4)
                    fig.clf()
                    ax = fig.gca()
                    if self.options['errorType'] == 'landmarks':
                        for kv in range(self.fvDef.vertices.shape[0]):
                            ax.plot((self.fvDef.vertices[kv, 0], self.fv1.vertices[kv, 0]),
                                    (self.fvDef.vertices[kv, 1], self.fv1.vertices[kv, 1]),
                                    color=[0.7,0.7,0.7], linewidth=1)
                    for kf in range(self.fv1.faces.shape[0]):
                        ax.plot(self.fv1.vertices[self.fv1.faces[kf,:],0], self.fv1.vertices[self.fv1.faces[kf,:],1], color=self.cmap1(kf))
                    for kf in range(self.fvDef.faces.shape[0]):
                        #ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
                        ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], color=self.cmap(kf), linewidth=self.lw)
            elif self.dim==3:
                fig = plt.figure(4)
                fig.clf()
                ax = fig.gca(projection='3d')
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf,:],0], self.fv1.vertices[self.fv1.faces[kf,:],1], 
                               self.fv1.vertices[self.fv1.faces[kf,:],2], color=[0,0,1])
                for kf in range(self.fvDef.faces.shape[0]):
                    ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], 
                               self.fvDef.vertices[self.fv1.faces[kf,:],2], color=[1,0,0], marker='*')
            #plt.axis('equal')
            fig.canvas.flush_events()
            #plt.pause(0.1)
                

    def endOptim(self):
        if self.saveRate==0 or self.iter%self.saveRate > 0:
            if self.dim==2:
                st = self.solveStateEquation(init_state=self.fv0.vertices, options={'withPointSet':self.gridxy})
                # (xt,yt) = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, self.control['at'], self.options['KparDiff'],
                #                                             affine=A, withPointSet=self.gridxy)
                yt = st['yt']
                # A = self.affB.getTransforms(self.control['Afft'])
                # (xt,yt) = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, self.control['at'], self.options['KparDiff'],
                #                                             affine=A, withPointSet=self.gridxy)
                for kk in range(self.Tsize+1):
                    self.gridDef.vertices = np.copy(yt[kk, :, :])
                    self.gridDef.saveVTK(self.outputDir + '/grid' + str(kk) + '.vtk')
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.state['xt'][kk, :, :]))
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                self.fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk')
        self.defCost = self.obj - self.obj0 - self.dataTerm(self.fvDef)


    def optimizeMatching(self):
        print('obj0', self.fun_obj0(self.fv1))
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(self.options['gradLB'], np.sqrt(grd2) / 10000)
        print('Gradient bound:', self.gradEps)
        kk = 0
        while os.path.isfile(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk'):
            os.remove(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk')
            kk += 1
        if self.options['algorithm'] == 'cg':
            cg.cg(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=.01,
                  Wolfe=self.options['Wolfe'])
        elif self.options['algorithm'] == 'bfgs':
            bfgs.bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=1.,
                      Wolfe=self.options['Wolfe'], lineSearch=self.options['lineSearch'], memory=50)
        # if self.param.algorithm == 'cg':
        #     cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.options['testGradient'], epsInit=0.01)
        # elif self.param.algorithm == 'bfgs':
        #     bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.options['testGradient'], epsInit=1.,
        #               Wolfe=self.param.wolfe, memory=25)
        #return self.at, self.xt

