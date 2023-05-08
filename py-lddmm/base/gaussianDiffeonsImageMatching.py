import numpy as np
import numpy.linalg as LA
import scipy.ndimage as Img
import scipy.stats.mstats as stats
from . import gaussianDiffeons as gd
from .gridscalars import GridScalars
from.diffeo import imageGradient, multilinInterp, multilinInterpGradient
from . import conjugateGradient as cg, diffeonEvolution as evol, bfgs
from.pointSetMatching import PointSetMatching, Control
from .affineBasis import *
from PIL import Image
import logging

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normalization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'


def ImageMatchingDist(xDef, im0, im1):
    #print gr.shape
    #print gr[10:20, 10:20, 0]
    #imdef0 = Img.interpolation.map_coordinates(im1.data, gr.transpose(range(-1, gr.ndim-1)), order=1, mode='nearest')
    gr = xDef[0]
    J = xDef[1]
    imdef = multilinInterp(im1.data, gr.transpose(range(-1, gr.ndim - 1)))
    #print 'diff:', (imdef-imdef0).max()
    # print im1.data.min(), im1.data.max()
    # print imdef.min(), imdef.max()
    res = (((im0.data - imdef)**2)*np.exp(J)).sum()
    return res

def ImageMatchingGradient(xDef, im0, im1, gradient=None):
    gr = xDef[0]
    J = xDef[1]
    imdef = multilinInterp(im1.data, gr.transpose(range(-1, gr.ndim - 1)))
    #imdef = Img.interpolation.map_coordinates(im1.data, gr.transpose(range(-1, gr.ndim-1)), order=1, mode='nearest')
    # if not (gradient==None):
    #     gradIm1 = gradient
    # else:
    #     gradIm1 = diffeo.gradient(im1.data, im1.resol)
    # gradDef = np.zeros(gradIm1.shape)
    # for k in range(gradIm1.shape[0]):
    #     gradDef[k,...] = diffeo.multilinInterp(gradIm1[k, ...], gr.transpose(range(-1, gr.ndim-1)))
        #gradDef[k,...] = Img.interpolation.map_coordinates(gradIm1[k, ...], gr.transpose(range(-1, gr.ndim-1)), order=1, mode='nearest')
    gradDef = multilinInterpGradient(im1.data, gr.transpose(range(-1, gr.ndim - 1)))
    
    expJ = np.exp(J)
    pgr = ((-2*(im0.data-imdef)*expJ)*gradDef).transpose(np.append(range(1, gr.ndim), 0))
    pJ =  ((im0.data - imdef)**2)*expJ
    return pgr, pJ


class State(dict):
    def __init__(self):
        super().__init__()
        self['imt'] = None
        self['Jt'] = None
        self['ct'] = None
        self['grt'] = None
        self['St'] = None


# class ImageMatchingParam:
#     def __init__(self, ):
#         self.timeStep = timeStep
#         self.sigmaKernel = sigmaKernel
#         self.sigmaError = sigmaError
#         self.typeKernel = typeKernel
#         self.errorType = errorType
#         self.dimension = dimension
#         self.algorithm = algorithm
#         self.wolfe = Wolfe
#         if errorType == 'L2':
#             self.fun_obj = ImageMatchingDist
#             self.fun_objGrad = ImageMatchingGradient
#         else:
#             print('Unknown error Type: ', self.errorType)
#         if KparDiff == None:
#             self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
#         else:
#             self.KparDiff = KparDiff



# class Direction:
#     def __init__(self):
#         self.diff = []
#         self.aff = []


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
class ImageMatchingDiffeons(PointSetMatching):
    def __init__(self, Template=None, Target=None, options=None):
        super().__init__(Template, Target, options)
        #

            #zoom = np.minimum(np.array(self.im0.data.shape).astype(float)/np.array(self.im1.data.shape), np.ones(self.im0.data.ndim))

        # Include template in bigger image

        # Include target in bigger image
        #print self.im1.data.shape
        # self.saveRate = 5
        self.iter = 0
        # self.gradEps = -1
        self.dim = self.im0.data.ndim
        # self.setOutputDir(outputDir)
        # self.maxIter = maxIter
        # self.verb = verb
        # self.testGradient = testGradient
        # self.regweight = regWeight
        # self.affine = affine

        #self.x0 = self.fv0.vertices

        if self.options['zeroVar']:
            self.S0 = np.zeros(self.S0.shape)

            #print self.c0
            #print self.S0
        #print self.gr0.shape
        # if self.dim == 1:
        #     self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate]
        #     self.gr0 = range(self.im0.data.shape[0])
        # elif self.dim == 2:
        #     self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate, 0:self.im0.data.shape[1]:subsampleTemplate]
        #     self.gr0 = np.mgrid[0:self.im0.data.shape[0], 0:self.im0.data.shape[1]].transpose((1, 2,0))
        # elif self.dim == 3:
        #     self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate, 0:self.im0.data.shape[1]:subsampleTemplate, 0:self.im0.data.shape[2]:subsampleTemplate]
        #     self.gr0 = np.mgrid[0:self.im0.data.shape[0], 0:self.im0.data.shape[1], 0:self.im0.data.shape[2]].transpose((1,2, 3, 0))
        # print('error type:', self.param.errorType)
        # self.gradCoeff = self.ndf
        self.im0.saveImg(self.options['outputDir']+'/Template.png', normalize=True)
        self.im1.saveImg(self.options['outputDir']+'/Target.png', normalize=True)

    # def setOutputDir(self, outputDir):
    #     self.outputDir = outputDir
    #     if not os.access(outputDir, os.W_OK):
    #         if os.access(outputDir, os.F_OK):
    #             print('Cannot save in ' + outputDir)
    #             return
    #         else:
    #             os.makedirs(outputDir)

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['dimension'] = 2
        options['errorType'] = 'L2'
        options['Diffeons'] = None
        options['EpsilonNet'] = None
        options['DecimationTarget'] = 1
        options['subsampleTemplate'] = 1
        options['targetMargin'] = 10
        options['templateMargin']=0
        options['DiffeonEpsForNet']=None
        options['DiffeonSegmentationRatio']=None
        options['zeroVar']=False
        options['gradEps'] = -1
        return options

    def set_template_and_target(self, Template, Target, misc=None):
        if Template == None:
            print('Please provide a template image')
            return
        else:
            self.im0 = GridScalars(grid=Template)

        self.dim = self.im0.data.ndim

            # print self.im0.data.shape, Template.data.shape

        if Target == None:
            print('Please provide a target image')
            return
        else:
            self.im1 = GridScalars(grid=Target)
        zoom = np.array(self.im0.data.shape).astype(float)/np.array(self.im1.data.shape)
        self.im1.data = Img.interpolation.zoom(self.im1.data, zoom, order=0)

        templateMargin = self.options['templateMargin']
        if templateMargin > 0:
            #self.im0.zeroPad(templateMargin)
            I = range(- templateMargin, self.im0.data.shape[0]+ templateMargin)
            for k in range(1, self.im0.data.ndim):
                I = (I, range(-templateMargin, self.im0.data.shape[k]+ templateMargin))
            self.gr1 = np.array(np.meshgrid(*I, indexing='ij'))
            #print self.im1.data.shape
            self.im0.data = multilinInterp(self.im0.data, self.gr1)

        targetMargin = self.options['targetMargin']
        #self.im1.zeroPad(targetMargin)
        I = range(-targetMargin, self.im1.data.shape[0]+targetMargin)
        for k in range(1, self.im1.data.ndim):
            I = (I, range(-targetMargin, self.im1.data.shape[k]+targetMargin))
        self.gr1 = np.array(np.meshgrid(*I, indexing='ij'))
        #print self.im1.data.shape
        self.im1.data = multilinInterp(self.im1.data, self.gr1)
        self.gr1 = self.gr1.transpose(np.append(range(1,self.gr1.ndim), 0)) + targetMargin

        self.im0Fine = GridScalars(grid=self.im0)

        if self.options['Diffeons'] is None:
            gradIm0 = np.sqrt((imageGradient(self.im0.data, self.im0.resol) ** 2).sum(axis=0))
            m0 = stats.mquantiles(gradIm0, 0.75)/10. + 1e-5
            if self.options['DecimationTarget'] is None:
                DecimationTarget = 1
            else:
                DecimationTarget = self.options['DecimationTarget']
            gradIm0 = Img.filters.maximum_filter(gradIm0, DecimationTarget)
            I = range(templateMargin, self.im0.data.shape[0]-templateMargin, DecimationTarget)
            for k in range(1, self.im0.data.ndim):
                I = (I, range(templateMargin, self.im0.data.shape[k]-templateMargin, DecimationTarget))
            u = np.meshgrid(*I, indexing='ij')
            self.c0 = np.zeros([u[0].size, self.dim])
            for k in range(self.dim):
                self.c0[:,k] = u[k].flatten()

            gradIm0 = multilinInterp(gradIm0, np.array(u)).flatten()

            jj = 0
            for kk in range(self.c0.shape[0]):
                if gradIm0[kk] > m0:
                    self.c0[jj, :] = self.c0[kk, :]
                    jj += 1
            self.c0 = self.c0[0:jj, :]
            print('keeping ', jj, ' diffeons')
                #print self.im0.resol
            self.c0 = targetMargin - templateMargin + self.im0.origin + self.c0 * self.im0.resol
            self.S0 = np.tile( (DecimationTarget*np.diag(self.im0.resol)/2)**2, [self.c0.shape[0], 1, 1])
        else:
            (self.c0, self.S0, self.idx) = self.options['Diffeons']


        if self.options['subsampleTemplate'] is None:
            subsampleTemplate = 1
        else:
            subsampleTemplate = self.options['subsampleTemplate']
        self.im0.resol *= subsampleTemplate
        self.im0.data = Img.filters.median_filter(self.im0.data, size=subsampleTemplate)
        I = range(0, self.im0.data.shape[0], subsampleTemplate)
        II = range(0, self.im0.data.shape[0])
        for k in range(1, self.im0.data.ndim):
            I = (I, range(0, self.im0.data.shape[k], subsampleTemplate))
            II = (II, range(0, self.im0.data.shape[k]))
        self.gr0 = np.array(np.meshgrid(*I, indexing='ij'))
        self.gr0Fine = np.array(np.meshgrid(*II, indexing='ij'))
        self.im0.data = multilinInterp(self.im0.data, self.gr0)
        self.gr0 = self.gr0.transpose(np.append(range(1,self.gr0.ndim), 0))
        self.gr0Fine = self.gr0Fine.transpose(np.append(range(1,self.gr0Fine.ndim), 0))

        self.gr0 = targetMargin-templateMargin+self.im1.origin + self.gr0 * self.im1.resol
        self.gr0Fine = targetMargin-templateMargin+self.im1.origin + self.gr0Fine * self.im1.resol
        self.J0 = np.log(self.im0.resol.prod()) * np.ones(self.im0.data.shape)
        self.ndf = self.c0.shape[0]

    def set_fun(self, errorType, vfun=None):
        if errorType == 'L2':
            self.fun_obj = ImageMatchingDist
            self.fun_objGrad = ImageMatchingGradient
        else:
            print('Unknown error Type: ', errorType)


    def set_parameters(self):
        if self.options['mode'] in ('normal', 'debug'):
            self.options['verb'] = True
            if self.options['mode'] == 'debug':
                self.options['testGradient'] = True
            else:
                self.options['testGradient'] = False
        else:
            self.options['verb'] = False

        affB = AffineBasis(self.dim, self.options['affine'])
        self.affB = affB
        self.affineDim = affB.affineDim
        self.affineBasis = affB.basis
        self.affineWeight = self.options['affineWeight'] * np.ones([self.affineDim, 1])
        if (len(affB.rotComp) > 0) & (self.options['rotWeight'] is not None):
            self.affineWeight[affB.rotComp] = self.options['rotWeight']
        if (len(affB.simComp) > 0) & (self.options['scaleWeight'] is not None):
            self.affineWeight[affB.simComp] = self.options['scaleWeight']
        if (len(affB.transComp) > 0) & (self.options['transWeight'] != None):
            self.affineWeight[affB.transComp] = self.options['transWeight']

    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.control = Control()
        self.controlTry = Control()
        self.state = State()
        self.control['at'] = np.zeros([self.Tsize, self.c0.shape[0], self.dim])
        self.controlTry['at'] = np.zeros([self.Tsize, self.c0.shape[0], self.dim])
        if self.affineDim > 0:
            self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.state['imt'] = np.tile(self.im0.data, np.insert(np.ones(self.dim, dtype=int), 0, self.Tsize+1))
        self.state['Jt'] = np.tile(self.J0, np.insert(np.ones(self.dim, dtype=int), 0, self.Tsize+1))
        self.state['grt'] = np.tile(self.gr0, np.insert(np.ones(self.dim+1, dtype=int), 0, self.Tsize+1))
        self.state['ct'] = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.state['St'] = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradIm1 = imageGradient(self.im1.data, self.im1.resol)



    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian = True):
        if var is None or 'initial' not in var:
            c0 = self.c0
            S0 = self.S0
            gr0 = self.gr0
            J0 = self.J0
        else:
            gr0 = self.gr0
            J0 = self.J0
            (c0, S0) = var['initial']

        timeStep = 1.0/self.Tsize
        if self.affineDim > 0:
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            for t in range(self.Tsize):
                AB = self.affineBasis @ control['Afft'][t]
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        else:
            A = None
        st = State()
        (st['ct'], st['St'], st['grt'], st['Jt'])  = \
            evol.gaussianDiffeonsEvolutionEuler(c0, S0, control['at'], self.options['sigmaKernel'], affine=A,
                                                withPointSet = gr0, withJacobian=J0)

        #print xt[-1, :, :]
        #print obj
        obj=0
        #print St.shape
        for t in range(self.Tsize):
            c = st['ct'][t, :, :]
            S = st['St'][t, :, :, :]
            a = control['at'][t, :, :]
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            rcc = gd.computeProducts(c, S, self.options['sigmaKernel'])
            obj = obj + self.options['regWeight']*timeStep*(a * (rcc@a)).sum()
            if self.affineDim > 0:
                obj +=  timeStep * (self.affineWeight.reshape(control['Afft'][t].shape) * control['Afft'][t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withTrajectory:
            return obj, st
        else:
            return obj

    def dataTerm(self, stDef, var = None):
        obj = self.fun_obj([stDef[0], stDef[1]], self.im0, self.im1) / (self.options['sigmaError']**2)
        return obj

    def objectiveFun(self):
        if self.obj is None:
            (self.obj, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
            #data = (self.xt[-1,:,:], self.Jt[-1,:])
            self.obj += self.dataTerm([self.state['grt'][-1,...], self.state['Jt'][-1,...]])

        return self.obj

    # def getVariable(self):
    #     return self.control

    def updateTry(self, dr, eps, objRef=None):
        controlTry = Control()
        for k in dr.keys():
            if dr[k] is not None:
                controlTry[k] = self.control[k] - eps * dr[k]
        # atTry = self.at - eps * dir.diff
        # if self.affineDim > 0:
        #     AfftTry = self.Afft - eps * dir.aff
        # else:
        #     AfftTry = self.Afft
        objTry, state = self.objectiveFunDef(controlTry, withTrajectory=True)
        #data = (grt[-1,:,:], Jt[-1,:])
        objTry += self.dataTerm([state['grt'][-1,...], state['Jt'][-1,...]])

        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if objRef is None or objTry < objRef:
            self.controlTry = controlTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()
        return objTry



    def endPointGradient(self, endPoint = None):
        if endPoint is None:
            endPoint = [self.state['grt'][-1, ...], self.state['Jt'][-1,...]]
        (pg, pJ) = self.fun_objGrad(endPoint, self.im0, self.im1, gradient=self.gradIm1)
        pc = np.zeros(self.c0.shape)
        pS = np.zeros(self.S0.shape)
        #gd.testDiffeonCurrentNormGradient(self.ct[-1, :, :], self.St[-1, :, :, :], self.bt[-1, :, :],
        #                               self.fv1, self.param.KparDist.sigma)
        pg = pg / self.options['sigmaError']**2
        pJ = pJ / self.options['sigmaError']**2
        return pc, pS, pg, pJ


    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]
        A = self.affB.getTransforms(control['Afft'])
        st = State()
        (st['ct'], st['St'], st['grt'], st['Jt'])  = \
            evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, control['at'], self.options['sigmaKernel'], affine=A,
                                                withPointSet = self.gr0, withJacobian=self.J0)

        endPoint = [st['grt'][-1, ...], st['Jt'][-1, ...]]
        return control, st, endPoint

    def getGradient(self, coeff=1.0, update = None):
        if update is None:
            control = self.control
            endPoint = [self.state['grt'][-1, ...], self.state['Jt'][-1,...]]
            # A = self.affB.getTransforms(self.control['Afft'])
            state = self.state
        else:
            control, state, endPoint = self.setUpdate(update)
            
        if self.affineDim > 0:
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            dim2 = self.dim ** 2
            for t in range(self.Tsize):
                AB = (self.affineBasis @ control['Afft'][t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        else:
            A = None

        (pc1, pS1, pg1, pJ1) = self.endPointGradient(endPoint=endPoint)
        foo = evol.gaussianDiffeonsGradientPset(self.c0, self.S0, self.gr0, control['at'], -pc1, -pS1, -pg1,
                                                self.options['sigmaKernel'], self.options['regWeight'],
                                                affine=A, withJacobian = (self.J0, -pJ1), euclidean=self.euclideanGradient)

        grd = Control()
        # if self.euclideanGradient:
        #     grd['at'] = np.zeros(foo['dat'].shape)
        #     for t in range(self.Tsize):
        #         c = state['ct'][t, :, :]
        #         S = state['St'][t, :, :, :]
        #         gg = foo['dat'][t, :, :]
        #         rcc = gd.computeProducts(c, S, self.options['sigmaKernel'])
        #         (L, W) = LA.eigh(rcc)
        #         rcc += (L.max() / 10000) * np.eye(rcc.shape[0])
        #         grd['at'][t, :, :] = rcc @ gg / (coeff*self.Tsize)
        #         # z = state['xt'][t, :, :]
        #         # grd['at'][t,:,:] = self.options['KparDiff'].applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
        # else:
        grd['at'] = foo['dat']/(coeff*self.Tsize)

        if self.affineDim > 0:
            grd['Afft'] = np.zeros(control['Afft'].shape)
            dim2 = self.dim ** 2
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2 * control['Afft']
            for t in range(self.Tsize):
               dAff = self.affineBasis.T @ np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])])
               grd['Afft'][t] -=  np.divide(dAff.reshape(grd['Afft'][t].shape), self.affineWeight.reshape(grd['Afft'][t].shape))
            grd['Afft'] /= (coeff*self.Tsize)
        return grd



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
    #     ddir = Direction()
    #     ddir.diff = np.copy(dir0.diff)
    #     ddir.aff = np.copy(dir0.aff)
    #     return ddir


    def randomDir(self):
        dirfoo = Control()
        dirfoo['at'] = np.random.randn(self.Tsize, self.ndf, self.dim)
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    # def getBGFSDir(Var, oldVar, grd, grdOld):
    #     s = (Var[0] - oldVar[0]).unravel()
    #     y = (grd.diff - grdOld.diff).unravel()
    #     if skipBGFS==0:
    #         rho = max(0, (s*y).sum())
    #     else:
    #         rho = 0
    #     Imsy = np.eye((s.shape[0], s.shape[0])) - rho*np.dot(s, y.T)
    #     H0 = np.dot(Imsy, np.dot(H0, Imsy)) + rho * np.dot(s, s.T)
    #     dir0.diff = (np.dot(H0, grd.diff.unravel()))

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
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
                uu = 0
            ll = 0
            for gr in g2:
                ggOld = gr['at'][t, :, :]
                res[ll]  = res[ll] + (ggOld * u).sum()
                if self.affineDim > 0:
                    res[ll] += np.multiply(uu, gr['Afft'][t]).sum()
                ll = ll + 1

        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            u = g1['at'][t, :, :]
            if self.affineDim > 0:
                uu = (g1['Afft'][t]*self.affineWeight.reshape(g1['Afft'][t].shape))
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

    def setDotProduct(self, unreduced=False):
        if self.options['algorithm'] == 'cg' and not unreduced:
             self.euclideanGradient = False
             self.dotProduct = self.dotProduct_Riemannian
        else:
            self.euclideanGradient = True
            self.dotProduct = self.dotProduct_euclidean


    # def acceptVarTry(self):
    #     self.obj = self.objTry
    #     self.at = np.copy(self.atTry)
    #     self.Afft = np.copy(self.AfftTry)
    #     #print self.at

    # def startOfIteration(self):
    #     u0 = self.dataTerm(self.grt[-1, ...], self.Jt[-1, ...])
    #     (pg, pJ) = self.param.fun_objGrad(self.grt[-1,...], self.Jt[-1, ...], self.im0, self.im1, gradient=self.gradIm1)
    #     eps = 1e-8
    #     dg = np.random.normal(size=self.grt[-1,...].shape)
    #     dJ = np.random.normal(size=self.Jt[-1,...].shape)
    #     ug = self.dataTerm(self.grt[-1, ...]+eps*dg, self.Jt[-1, ...])
    #     uJ = self.dataTerm(self.grt[-1, ...], self.Jt[-1, ...]+eps*dJ)
    #     print 'Test end point gradient: grid ', (ug-u0)/eps, (pg*dg).sum(), 'jacobian', (uJ-u0)/eps, (pJ*dJ).sum() 

    def startOfIteration(self):
        pass

    def endOfIteration(self, forceSave = False):
        #print self.obj0
        self.iter += 1
        if forceSave or (self.iter % self.options['saveRate']) == 0:
            logging.info('saving...')
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True)
            dim2 = self.dim**2
            if self.affineDim > 0:
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.control['Afft'][t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            else:
                A = None
            (ct, St, grt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.control['at'], 
                                                                 self.options['sigmaKernel'], affine=A, 
                                                                 withPointSet= self.gr0Fine)
            # (ct, St, grt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.at, self.param.sigmaKernel, affine=A,
            #                                                         withPointSet = self.fv0Fine.vertices, withJacobian=True)
            imDef = GridScalars(grid = self.im1)
            for kk in range(self.Tsize+1):
                imDef.data = multilinInterp(self.im1.data, grt[kk, ...].transpose(range(-1, self.state['grt'].ndim - 2)))
                imDef.saveImg(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk)+'.png', normalize=True)
                if self.dim==3:
                    gd.saveDiffeons(self.options['outputDir'] +'/'+ self.options['saveFile']+'Diffeons'+str(kk)+'.vtk',
                                    self.state['ct'][kk,...], self.state['St'][kk,...])
                elif self.dim==2:
                    (R, detR) = gd.multiMatInverse1(self.state['St'][kk,...], isSym=True)
                    diffx = self.gr1[..., np.newaxis, :] - self.state['ct'][kk, ...]
                    betax = (R*diffx[..., np.newaxis, :]).sum(axis=-1)
                    dst = (betax * diffx).sum(axis=-1)
                    diffIm = np.minimum((255*(1-dst)*(dst < 1)).astype(float).sum(axis=-1), 255)
                    out = Image.fromarray(diffIm.astype(np.uint8))
                    out.save(self.options['outputDir'] +'/'+ self.options['saveFile']+'Diffeons'+str(kk)+'.png')
        else:
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True)



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

        if self.options['gradEps'] < 0:
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])

            self.gradEps = max(0.001, np.sqrt(grd2) / 10000)

        print('Gradient lower bound: ', self.gradEps)
        if self.options['algorithm'] == 'cg':
            cg.cg(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=0.1)
        elif self.options['algorithm'] == 'bfgs':
            bfgs.bfgs(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=1.,
                      Wolfe=self.options['Wolfe'], memory=50)

        #return self.at, self.xt

