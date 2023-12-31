import numpy as np
import logging
from copy import deepcopy
from . import conjugateGradient as cg
from . import bfgs
from .gridscalars import GridScalars, saveImage
from .diffeo import multilinInterp, multilinInterpGradient, jacobianDeterminant, \
    multilinInterpDual, idMesh
from .imageMatchingBase import ImageMatchingBase

# from functools import partial
# import matplotlib
# #matplotlib.use("QT5Agg")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
        self['Lv'] = None
        self['Z'] = None

class Metamorphosis(ImageMatchingBase):
    def __init__(self, Template=None, Target=None, options = None):
        super().__init__(Template=Template, Target=Target, options=options)
        self.initialize_variables()
        #self.gradCoeff = np.array(self.shape).prod()


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['imgCoeff'] = 1.
        return options


    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.vfShape = [self.dim] + list(self.im0.data.shape)
        self.imShape = self.im0.data.shape
        self.timeImageShape = (self.Tsize+1,) + self.imShape
        self.timeVfShape = (self.Tsize,) + tuple(self.vfShape)
        self.ZShape = (self.Tsize,) + self.imShape
        self.imDef = np.zeros(self.timeImageShape)
        self.imDef[0,...] = self.im0.data
        self.imDef[-1,...] = self.im1.data

        self.control = Control()
        self.controlTry = Control()
        self.v = np.zeros(self.timeVfShape)
        self.control['Lv'] = np.zeros(self.timeVfShape)
        self.controlTry['Lv'] = np.zeros(self.timeVfShape)
        self.control['Z'] = np.zeros(self.ZShape)
        self.control['Z'][:,...] = ((self.im1.data - self.im0.data))[None,...]
        self.controlTry['Z'] = np.zeros(self.ZShape)
        # if self.randomInit:
        #     self.at = np.random.normal(0, 1, self.at.shape)

    def flowImage(self, control):
        if self.options['metaDirection'] == 'FWD':
            return self.flowImage_fwd(control)
        elif self.options['metaDirection'] == 'BWD':
            return self.flowImage_bwd(control)
        else:
            logging.error('Unknown direction in flow')
            return

    def flowImage_fwd(self, control):
        Lv = control['Lv']
        Z = control['Z']
        id = idMesh(self.imShape)
        v = np.zeros(Lv.shape)
        dt = 1/self.Tsize
        imDef = np.zeros(self.timeImageShape)
        imDef[0,...] = self.im0.data
        imDef[-1,...] = self.im1.data

        for t in range(self.Tsize - 1):
            v[t, ...] = self.kernel(Lv[t, ...])
            foo = id + dt * v[t, ...]
            imDef[t + 1, ...] = multilinInterp(imDef[t, ...], foo) + Z[t, ...] * dt
        t = self.Tsize - 1
        v[t, ...] = self.kernel(Lv[t, ...])
        foo = id + dt * v[t, ...]
        ener = ((imDef[self.Tsize, ...] - multilinInterp(imDef[t, ...], foo)) ** 2).sum() \
               / (2 * dt * self.options['sigmaError'] ** 2)
        ener += ((Lv * v).sum() + (Z[:self.Tsize - 1, ...] ** 2).sum() / self.options['sigmaError'] ** 2) * dt / 2
        return ener, imDef

    def flowImage_bwd(self, control):
        Lv = control['Lv']
        Z = control['Z']
        id = idMesh(self.imShape)
        v = np.zeros(Lv.shape)
        dt = 1/self.Tsize
        imDef = np.zeros(self.timeImageShape)

        for t in range(self.Tsize, 1, -1):
            v[t, ...] = self.kernel(Lv[t, ...])
            foo = id - dt * v[t - 1, ...]
            imDef[t - 1, ...] = multilinInterp(imDef[t, ...], foo) - Z[t - 1, ...] * dt
        t = 1
        v[0, ...] = self.kernel(Lv[0, ...])
        foo = id - dt * v[0, ...]
        ener = ((imDef[0, ...] - multilinInterp(imDef[t, ...], foo)) ** 2).sum() \
               / (2 * dt * self.options['sigmaError'] ** 2)
        ener += ((Lv * v).sum() + (Z[1:, ...] ** 2).sum() / self.options['sigmaError'] ** 2) * dt / 2
        return ener, imDef

    def initial_plot(self):
        pass
        # fig = plt.figure(3)
        # ax = Axes3D(fig)
        # lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r')
        # if self.fv1:
        #     lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
        # else:
        #     lim0 = lim1
        # ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
        # ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
        # ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
        # fig.canvas.flush_events()

    def initialSave(self):
        saveImage(self.im0.data, self.outputDir + '/Template')
        saveImage(self.im1.data, self.outputDir + '/Target')
        saveImage(self.KparDiff.K, self.outputDir + '/Kernel', normalize=True)
        if self.smoothKernel:
            saveImage(self.smoothKernel.K, self.outputDir + '/smoothKernel', normalize=True)
        saveImage(self.mask.min(axis=0), self.outputDir + '/Mask', normalize=True)


    def objectiveFun(self, control = None):
        if control is None:
            ener = self.flowImage(self.control)[0]
            self.obj = ener
        else:
            ener = self.flowImage(control)[0]
        return ener

    def gradient_bwd(self, control):
        Lv = control['Lv']
        Z = control['Z']
        dt = 1/self.Tsize
        id = idMesh(self.imShape)
        ener, imDef = self.flowImage_bwd(control)
        v = np.zeros(Lv.shape)
        p = np.zeros(Lv.shape)

        v[0, ...] = self.kernel(Lv[0, ...])
        p[0,...] = (multilinInterp(imDef[1, ...], id - dt * v[0, ...]) - imDef[0, ...] )/self.options['sigmaError'] ** 2
        for t in range(1, self.Tsize):
            p[t,...] = multilinInterpDual(p[t-1,...], id - dt * v[t-1, ...])
            v[t, ...] = self.kernel(Lv[t, ...])
        grad = Control()
        grad['Lv'] = np.zeros(Lv.shape)
        for t in range(self.Tsize):
            grad['Lv'][t,...] = Lv[t,...] - multilinInterpGradient(imDef[t+1,...], id - dt*v[t,...])*p[t, None, ...]
        grad['Z'] = np.zeros(Z.shape)
        for t in range(1,self.Tsize):
            grad['Z'][t,...] = Z[t,...]/self.options['sigmaError'] ** 2 - p[t,...]
        return grad

    def gradient_fwd(self, control):
        Lv = control['Lv']
        Z = control['Z']
        dt = 1/self.Tsize
        id = idMesh(self.imShape)
        ener, imDef = self.flowImage_fwd(control)
        v = np.zeros(Lv.shape)
        p = np.zeros(Z.shape)

        v[self.Tsize-1, ...] = self.kernel(Lv[self.Tsize-1, ...])
        p[self.Tsize-1,...] = (imDef[self.Tsize, ...] - multilinInterp(imDef[self.Tsize-1, ...], id + dt *v[self.Tsize-1,...]) )\
                              /(dt*self.options['sigmaError'] ** 2)
        for t in range(self.Tsize-1, 0, -1):
            p[t-1,...] = multilinInterpDual(p[t,...], id + dt * v[t, ...])
            v[t - 1, ...] = self.kernel(Lv[t - 1, ...])
        grad = Control()
        grad['Lv'] = np.zeros(Lv.shape)
        for t in range(self.Tsize):
            grad['Lv'][t,...] = Lv[t,...] - multilinInterpGradient(imDef[t,...], id + dt*v[t,...])*p[t, None, ...]
        grad['Z'] = np.zeros(Z.shape)
        for t in range(self.Tsize-1):
            grad['Z'][t,...] = Z[t,...]/self.options['sigmaError'] ** 2 - p[t,...]
        return grad


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            # Lv = self.Lv
            # Z = self.Z
        else:
            control = Control()
            control['Lv'] = self.control['Lv'] - update[1]*update[0]['Lv']
            control['Z'] = self.control['Z'] - update[1]*update[0]['Z']

        if self.options['metaDirection'] == 'FWD':
            grad = self.gradient_fwd(control)
        elif self.options['metaDirection'] == 'BWD':
            grad =  self.gradient_bwd(control)
        else:
            logging.error('Unknown direction in getGradient')
            return

        if self.euclideanGradient:
            for t in range(grad['Lv'].shape[0]):
                grad['Lv'][t,...] = self.kernel(grad['Lv'][t, ...])
        grad['Lv'] /= coeff
        grad['Z'] /= coeff * self.options['imgCoeff']
        return grad


    def getVariable(self):
        return self.control

    def updateTry(self, dir, eps, objRef=None):
        controlTry = Control()
        controlTry['Lv'] = self.control['Lv'] - eps * dir['Lv']
        controlTry['Z'] = self.control['Z'] - eps * dir['Z']
        objTry = self.objectiveFun(controlTry)
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry


    # def addProd(self, dir1, dir2, beta):
    #     dir = Direction()
    #     dir.diff = dir1.diff + beta * dir2.diff
    #     dir.Z = dir1.Z + beta * dir2.Z
    #     return dir

    # def prod(self, dir1, beta):
    #     dir = Direction()
    #     dir.diff = beta * dir1.diff
    #     dir.Z = beta * dir1.Z
    #     return dir

    # def copyDir(self, dir0):
    #     dir = Direction()
    #     dir.diff = np.copy(dir0.diff)
    #     dir.Z = np.copy(dir0.Z)
    #     return dir


    def randomDir(self):
        dirfoo = Control()
        dirfoo['Lv'] = np.random.normal(size=self.control['Lv'].shape)
        dirfoo['Z'] = np.random.normal(size=self.control['Z'].shape)
        if self.options['metaDirection'] == 'FWD':
            dirfoo['Z'][self.Tsize-1,...] = 0
        elif self.options['metaDirection'] == 'BWD':
            dirfoo['Z'][0,...] = 0
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        u = np.zeros(g1['Lv'].shape)
        for k in range(u.shape[0]):
            u[k,...] = self.kernel(g1['Lv'][k,...])
        ll = 0
        for gr in g2:
            ggOld = gr['Lv']
            res[ll]  = ((ggOld*u).sum() + self.options['imgCoeff']*(g1['Z']*gr['Z']).sum())/u.shape[0]
            ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        u = g1['Lv']
        ll = 0
        for gr in g2:
            ggOld = gr['Lv']
            res[ll]  = ((ggOld*u).sum() + (g1['Z']*gr['Z']).sum())/u.shape[0]
            ll = ll + 1
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.control = deepcopy(self.controlTry)
        #print self.at



    def endOfIteration(self, forceSave =  False):
        self.iter += 1
        if self.iter % 10 == 0:
            print('Saving')
            ener, imDef = self.flowImage(self.control)
            self.initFlow()
            for t in range(self.Tsize+1):
                if t < self.Tsize:
                    self.updateFlow(self.control['Lv'][t,...], 1.0/self.Tsize)
                GridScalars(grid=imDef[t,...], dim=self.dim).saveImg(self.outputDir + f'/movie{t+1:03d}.png', normalize=True)

            I1 = multilinInterp(self.im0.data, self._psi)
            saveImage(I1, self.outputDir + f'/deformedTemplate')
            I1 = multilinInterp(self.im1.data, self._phi)
            saveImage(I1, self.outputDir + f'/deformedTarget')
            saveImage(np.squeeze(self.control['Z'][0,...]), self.outputDir + f'/initialMomentum', normalize=True)
            dphi = np.log(jacobianDeterminant(self._phi, self.resol))
            saveImage(dphi, self.outputDir + f'/logJacobian', normalize=True)

            # self.GeodesicDiffeoEvolution(self.Lv[0,...])
            # I1 = multilinInterp(self.im0.data, self._psi)
            # saveImage(I1, self.outputDir + f'/EPDiffTemplate')
            # self.psi = np.copy(self._psi)
      #
      # sprintf(file, "%s/template2targetMap", path) ;
      # //if (param.verb)
      # //cout << "writing " << file << endl ;
      # _phi.write(file) ;
      #
      # sprintf(file, "%s/target2templateMap", path) ;
      # //if (param.verb)
      # //cout << "writing " << file << endl ;
      # _psi.write(file) ;

# if (param.saveProjectedMomentum) {
#     //ImageEvolution mo ;
#     Vector Z0, Z ;
#     VectorMap v0 ;
#     kernel(Lv0[0], v0) ;
#     v0.scalProd(Template.normal(), Z) ;
#     Z *= -1 ;
#     //    Template.infinitesimalAction(v0, Z) ;
#     // cout << "projection " << Z.d.length << endl ;
#     imageTangentProjection(Template, Z, Z0) ;
#     sprintf(file, "%s/initialScalarMomentum", path) ;
#     //if (param.verb)
#     //cout << "writing " << file << endl ;
#     Z0.write(file) ;
#
#
#     sprintf(file, "%s/scaledScalarMomentum", path) ;
#     //if (param.verb)
#     //cout << "writing " << file << endl ;
#     Z0.writeZeroCentered(file) ;
#
#
#
#     //  Z0 *= -1 ;
#     Template.getMomentum(Z0, Lwc) ;
#     GeodesicDiffeoEvolution(Lwc) ;
#     _psi.multilinInterp(Template.img(), I1) ;
#     sprintf(file, "%s/Z0ShootedTemplate", path) ;
#     //if (param.verb)
#     //cout << "writing " << file << endl ;
#     I1.write_image(file) ;
#   }

    def endOfProcedure(self):
        self.endOfIteration()

    def optimizeMatching(self):
        # print 'dataterm', self.dataTerm(self.fvDef)
        # print 'obj fun', self.objectiveFun(), self.obj0
        grd = self.getGradient(coeff=self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        self.epsMax = 5.
        logging.info('Gradient lower bound: %f' % (self.gradEps))
        if self.options['algorithm'] == 'cg':
            cg.cg(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=0.1,
                  lineSearch=self.options['lineSearch'])
        elif self.options['algorithm'] == 'bfgs':
            bfgs.bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=1.,
                  lineSearch=self.options['lineSearch'], memory=50)

