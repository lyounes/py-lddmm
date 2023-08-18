import numpy as np
import logging
from copy import deepcopy
from . import conjugateGradient as cg
from . import bfgs
from .gridscalars import GridScalars, saveImage
from .diffeo import multilinInterp, multilinInterpGradient, multilinInterpGradientVectorField, jacobianDeterminant, \
    multilinInterpDual, imageGradient, idMesh
from .imageMatchingBase import ImageMatchingBase


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


class ImageMatching(ImageMatchingBase):
    def __init__(self, Template=None, Target=None, options = None):
        super().__init__(Template=Template, Target=Target, options=options)
        self.initialize_variables()
        self.gradCoeff = np.array(self.shape).prod()



    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.imDef = GridScalars(grid=self.im0)

        vfShape = [self.Tsize, self.dim] + list(self.im0.data.shape)
        self.control = Control()
        self.controlTry = Control()
        self.v = np.zeros(vfShape)
        self.control['Lv'] = np.zeros(vfShape)
        self.controlTry['Lv'] = np.zeros(vfShape)
        self.Lv0 = np.zeros(vfShape[1:])
        # if self.randomInit:
        #     self.at = np.random.normal(0, 1, self.at.shape)


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

    # def initialSave(self):
    #     if len(self.im0.data.shape) == 3:
    #         ext = '.vtk'
    #     else:
    #         ext = ''
    #     saveImage(self.im0.data, self.outputDir + '/Template' + ext)
    #     saveImage(self.im1.data, self.outputDir + '/Target' + ext)
    #     saveImage(self.KparDiff.K, self.outputDir + '/Kernel' + ext, normalize=True)
    #     saveImage(self.smoothKernel.K, self.outputDir + '/smoothKernel' + ext, normalize=True)
    #     saveImage(self.mask.min(axis=0), self.outputDir + '/Mask' + ext, normalize=True)



    def dataTerm(self, IDef_, var = None):
        if var is None or 'I1' not in var:
            I1 = self.im1.data
        else:
            I1 = var['I1']
        # I = multilinInterp(I0, psi)
        return ((IDef_-I1)**2).sum()/2

    def objectiveFun(self, control = None):
        if control is None:
            Lv = self.control['Lv']
        else:
            Lv = control['Lv']
        dt = 1.0 / self.Tsize
        self.initFlow()
        ener = 0
        for t in range(Lv.shape[0]):
            ener += self.updateFlow(Lv[t,...], dt) * dt/2
        self.IDef = multilinInterp(self.im0.data, self._psi)
        ener += self.dataTerm(self.IDef,  {'I1':self.im1.data}) / self.options['sigmaError']**2
        if control is None:
            self.obj = ener
        return ener

    def LDDMMTimegradient(self, phi, psi, I0, I1, resol):
        I = multilinInterp(I0, phi)
        DI = multilinInterp(I1, psi) - I
        jj = jacobianDeterminant(phi, self.resol)
        b =  imageGradient(I, resol) * DI
        return b

    def LDDMMgradientInPsi(self, psi, I0, I1):
        DI = multilinInterp(I0, psi) - I1
        #g = imageGradient(I0)
        b = multilinInterpGradient(I0, psi) * DI[None,...]
        #b = multilinInterpVectorField(g, psi) * DI
        return b

    def testDataTermGradient(self, psi, I0, I1):
        g = self.LDDMMgradientInPsi(psi, I0, I1)
        eps = 1e-6
        # en0 = self.dataTerm(multilinInterp(I0, psi), {'I1':I1})
        dpsi = np.random.normal(size=psi.shape)
        dpsi[0,...] = 0
        newpsi = np.maximum(psi + eps*dpsi , 0)
        for k in range(psi.shape[0]):
            newpsi[k,...] = np.minimum(newpsi[k,...], I0.shape[k]-1)
        dpsi = (newpsi-psi)/eps
        #dpsi = np.random.normal(size=psi.shape)
        en0 = self.dataTerm(multilinInterp(I0, (psi-eps*dpsi)), {'I1':I1})
        en1 = self.dataTerm(multilinInterp(I0, (psi+eps*dpsi)), {'I1':I1})
        # en1 = self.dataTerm(psi+eps*dpsi, I0, I1)
        print(f'Test data gradient {(en1-en0)/(2*eps):0.4f}, {(g*dpsi).sum():.4f}')


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            Lvt = self.control['Lv']
        else:
            Lvt = self.control['Lv'] - update[1]*update[0]['Lv']
        dt = 1.0 / self.Tsize

        grad = np.zeros([self.Tsize] + self.vfShape)
        psi = np.zeros([self.Tsize+1] + self.vfShape)
        self.initFlow()
        psi[0,...] = self._psi.copy()
        for t in range(self.Tsize):
            self.updateFlow(Lvt[t,...], dt)
            psi[t+1,...] = self._psi.copy()

        pp = -self.LDDMMgradientInPsi(self._psi, self.im0.data, self.im1.data) / (self.options['sigmaError'] **2)
        # self.testDataTermGradient(self._psi, self.im0.data, self.im1.data)
        id = idMesh(self.imShape)
        ng2 = 0
        self._epsBound = self.options['epsMax']
        for t in range(self.Tsize-1, -1, -1):
            vt = self.kernel(Lvt[t, ...])
            c0 = np.sqrt((vt**2).sum(axis=0)).max()
            foo = id - dt*vt
            Dpsi = multilinInterpGradientVectorField(psi[t,...], foo)
            foo2 = (Dpsi * pp[:, None, ...]).sum(axis=0)
            grad[t, ...] = Lvt[t,...] + foo2
            for i in range(pp.shape[0]):
                # ppTest = pp[i].copy()
                # pp2 = self.im1.data.copy()
                pp[i] = multilinInterpDual(pp[i], foo)
                # test1 = (pp2*pp[i]).sum()
                # pp2 = multilinInterp(pp2, foo)
                # test2 = (ppTest*pp2).sum()
                # print(f'test dual: {test1: .4f}, {test2:.4f}')

            foo = self.kernel(grad[t,...])
            ng2 += (grad[t, ...] * foo).sum()
            c1 = np.sqrt((foo ** 2).sum(axis=0)).max()
            epsTmp = (1+c0) / (1+c1)
            if epsTmp < self._epsBound:
                self._epsBound = epsTmp
            if self.euclideanGradient:
                grad[t,...] = foo
        #self.epsBig = 10*self._epsBound
        res = Control()
        res['Lv'] = grad/coeff
        return res



    def getVariable(self):
        return self.control


    def updateTry(self, dir, eps, objRef=None):
        controlTry = Control()
        controlTry['Lv'] = self.control['Lv'] - eps * dir['Lv']
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
    #     dir = Control()
    #     dir[''] = dir1.diff + beta * dir2.diff
    #     return dir
    #
    # def prod(self, dir1, beta):
    #     dir = Direction()
    #     dir.diff = beta * dir1.diff
    #     return dir
    #
    # def copyDir(self, dir0):
    #     dir = Direction()
    #     dir.diff = np.copy(dir0.diff)
    #     return dir
    #

    def randomDir(self):
        dirfoo = Control()
        dirfoo['Lv'] = np.random.normal(size=self.control['Lv'].shape)
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        u = np.zeros(g1['Lv'].shape)
        for k in range(u.shape[0]):
            u[k,...] = self.kernel(g1['Lv'][k,...])
        ll = 0
        for gr in g2:
            ggOld = gr['Lv']
            res[ll]  = (ggOld*u).sum()/u.shape[0]
            ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        u = g1['Lv']
        ll = 0
        for gr in g2:
            ggOld = gr['Lv']
            res[ll]  = (ggOld*u).sum()/u.shape[0]
            ll = ll + 1
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.control = deepcopy(self.controlTry)
        #print self.at



    def endOfIteration(self, forceSave=False):
        self.iter += 1
        if len(self.im0.data.shape) == 3:
            ext = '.vtk'
        else:
            ext = ''
        if self.iter % 10 == 0:
            _psi = np.copy(self.psi)
            self.initFlow()
            for t in range(self.Tsize):
                self.updateFlow(self.control['Lv'][t,...], 1.0/self.Tsize)
                if (self.saveMovie):
                    I1 = multilinInterp(self.im0.data, self._psi)
                    saveImage(I1, self.outputDir + f'/movie{t+1:03d}' + ext)

            I1 = multilinInterp(self.im0.data, self._psi)
            saveImage(I1, self.outputDir + f'/deformedTemplate' + ext)
            I2 = multilinInterp(I1, self._phi)
            I1 = multilinInterp(self.im1.data, self._phi)
            saveImage(I1, self.outputDir + f'/deformedTarget' + ext)
            saveImage(np.squeeze(self.control['Lv'][0,...]), self.outputDir + f'/initialMomentum' + ext, normalize=True)
            dphi = np.log(jacobianDeterminant(self._phi, self.resol))
            saveImage(dphi, self.outputDir + f'/logJacobian' + ext, normalize=True)

            self.GeodesicDiffeoEvolution(self.control['Lv'][0,...])
            I1 = multilinInterp(self.im0.data, self._psi)
            saveImage(I1, self.outputDir + f'/EPDiffTemplate' + ext)
            self.psi = np.copy(self._psi)
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

