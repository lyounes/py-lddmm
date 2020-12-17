import numpy as np
import logging
from . import pointEvolution as evol
from .surfaces import Surface
from .curves import Curve
from .curves import measureNorm0, measureNormDef, measureNormGradient
from .curves import currentNorm0, currentNormDef, currentNormGradient
from .curves import varifoldNorm0, varifoldNormDef, varifoldNormGradient
from . import pointSets
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .surfaceMatching import SurfaceMatchingParam, SurfaceMatching
from .surfaceSection import Surf2SecDist, Surf2SecGrad




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
class SurfaceToSectionsMatching(SurfaceMatching):
    def __init__(self, Template=None, Target=None, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
                 subsampleTargetSize=-1, affineOnly = False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):
        if param is None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        if self.param.algorithm == 'cg':
             self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.setOutputDir(outputDir)
        self.fv0 = None
        self.fv1 = None
        self.fvInit = None
        self.dim = 0
        self.fun_obj = None
        self.fun_obj0 = None
        self.fun_objGrad = None
        self.obj0 = 0
        self.coeffAff = 1
        self.obj = 0
        self.xt = None

        self.set_fun(self.param.errorType)
        self.set_template_and_target(Template, Target, subsampleTargetSize)



        self.set_parameters(maxIter=maxIter, regWeight = regWeight, affineWeight = affineWeight,
                            internalWeight=internalWeight, verb=verb, affineOnly = affineOnly,
                            rotWeight = rotWeight, scaleWeight = scaleWeight, transWeight = transWeight,
                            symmetric = symmetric, testGradient=testGradient, saveFile = saveFile,
                            saveTrajectories = saveTrajectories, affine = affine)
        self.initialize_variables()
        self.gradCoeff = self.x0.shape[0]

        self.pplot = pplot
        self.colors = ('b', 'm', 'g', 'r', 'y', 'k')
        if self.pplot:
            self.initial_plot()
        self.saveRate = 10
        self.forceLineSearch = True



    def set_template_and_target(self, Template, Target, subsampleTargetSize=-1):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = Surface(surf=Template)


        if type(Target) in (list, tuple):
            self.fv1 = Target
        else:
            logging.error('Target must be a list or tuple of SurfaceSection')
            return

        self.fvInit = Surface(surf=self.fv0)
        self.fix_orientation()
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        if self.fv1:
            outCurve = ()
            for k,f in enumerate(self.fv1):
                outCurve += (f.curve,)
            c = Curve(outCurve)
            c.saveVTK(self.outputDir+f'/TargetCurves.vtk')
        self.dim = self.fv0.vertices.shape[1]


    def fix_orientation(self):
        self.fv0ori = 1
        self.fv1ori = 1



    def initial_plot(self):
        fig = plt.figure(3)
        ax = Axes3D(fig)
        lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r', al=0.2)
        for k,f in enumerate(self.fv1):
            lim0 = self.addCurveToPlot(f, ax, ec=self.colors[k%len(self.colors)], fc='b', lw=5)
            for i in range(3):
                lim1[i][0] = min(lim0[i][0], lim1[i][0])
                lim1[i][1] = max(lim0[i][1], lim1[i][1])
        ax.set_xlim(lim1[0][0], lim1[0][1])
        ax.set_ylim(lim1[1][0], lim1[1][1])
        ax.set_zlim(lim1[2][0], lim1[2][1])
        fig.canvas.flush_events()

    def set_fun(self, errorType):
        self.param.errorType = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj0 = partial(currentNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(currentNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(currentNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType=='measure':
            #print('Running Measure Matching')
            self.fun_obj0 = partial(measureNorm0, KparDist=self.param.KparDist)
            self.fun_obj = partial(measureNormDef,KparDist=self.param.KparDist)
            self.fun_objGrad = partial(measureNormGradient,KparDist=self.param.KparDist)
        elif errorType=='varifold':
            self.fun_obj0 = partial(varifoldNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(varifoldNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(varifoldNormGradient, KparDist=self.param.KparDist, weight=1.)
        else:
            print('Unknown error Type: ', self.param.errorType)

    def addCurveToPlot(self, fv1, ax, ec = 'b', fc = 'r', al=1., lw=1):
        return fv1.curve.addToPlot(ax, ec = ec, fc = fc, al=al, lw=lw)


    def dataTerm(self, _fvDef, _fvInit = None):
        obj = 0
        for k,f in enumerate(self.fv1):
            obj += Surf2SecDist(_fvDef, f, self.fun_obj, curveDist0=self.fun_obj0)# plot=101+k)
        obj /= self.param.sigmaError**2
        return obj


    def objectiveFun(self):
        if self.obj is None:
            self.obj0 = 0
            for f in self.fv1:
                self.obj0 += self.fun_obj0(f.curve) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
        return self.obj


    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
        px = np.zeros(endPoint.vertices.shape)
        for f in self.fv1:
            px += Surf2SecGrad(endPoint, f, self.fun_objGrad)
        return px / self.param.sigmaError**2

    def saveCorrectedTarget(self, U, b):
        for k, f0 in enumerate(self.fv1):
            fc = Curve(curve=f0.curve)
            yyt = np.dot(fc.vertices - b, U)
            fc.updateVertices(yyt)
            fc.saveVTK(self.outputDir + f'/TargetCurveCorrected{k:03d}.vtk')


    def endOfIteration(self):
        self.iter += 1
        if self.testGradient:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        if self.iter % self.saveRate == 0:
            logging.info('Saving surfaces...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            if not self.internalCost and self.affineDim <= 0:
                xtEPDiff, atEPdiff = self.saveEPDiff(self.fvInit, self.at, fileName=self.saveFile)
                logging.info('EPDiff difference %f' % (np.fabs(self.xt[-1, :, :] - xtEPDiff[-1, :, :]).sum()))

            if self.saveTrajectories:
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curves.vtk', self.xt)

            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)
            dim2 = self.dim ** 2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2 + self.dim]
            (xt, Jt) = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                         withJacobian=True)
            if self.affine == 'euclidean' or self.affine == 'translation':
                self.saveCorrectedEvolution(self.fvInit, xt, self.at, self.Afft, fileName=self.saveFile,
                                            Jacobian=Jt)
            self.saveEvolution(self.fvInit, xt, Jacobian=Jt, fileName=self.saveFile)
        else:
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)

        if self.pplot:
            fig = plt.figure(4)
            fig.clf()
            ax = Axes3D(fig)
            lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r', al=0.2)
            for k,f in enumerate(self.fv1):
                lim0 = self.addCurveToPlot(f, ax, ec=self.colors[k%len(self.colors)], fc='b', lw=5)
                for i in range(3):
                    lim1[i][0] = min(lim0[i][0], lim1[i][0])
                    lim1[i][1] = max(lim0[i][1], lim1[i][1])
            ax.set_xlim(lim1[0][0], lim1[0][1])
            ax.set_ylim(lim1[1][0], lim1[1][1])
            ax.set_zlim(lim1[2][0], lim1[2][1])
            fig.canvas.flush_events()



    # def optimizeMatching(self):
    #     #print 'dataterm', self.dataTerm(self.fvDef)
    #     #print 'obj fun', self.objectiveFun(), self.obj0
    #     self.coeffAff = self.coeffAff2
    #     grd = self.getGradient(self.gradCoeff)
    #     [grd2] = self.dotProduct(grd, [grd])
    #
    #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
    #     self.epsMax = 5.
    #     logging.info('Gradient lower bound: %f' %(self.gradEps))
    #     self.coeffAff = self.coeffAff1
    #     if self.param.algorithm == 'cg':
    #         cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1)
    #     elif self.param.algorithm == 'bfgs':
    #         bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
    #                   Wolfe=self.param.wolfe, memory=50)
    #     #return self.at, self.xt

