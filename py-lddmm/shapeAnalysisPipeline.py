import pandas as pd
import numpy as np
from .base import diffeo
from .base import surfaces
from .base import pointSets
from .base.affineRegistration import rigidRegistration, saveRigid
from .base import surfaceTemplate as sTemp
from .base import surfaceMatching as smatch
from .base import loggingUtils, kernelFunctions as kfun
from os import path






# Pipeline for longitudinal shape analysis
# Input: .csv file with locations of original files assumed to be either image segmentations or surfaces.
#        locations for left and right structures must be present, with fields location_left and location_right
#        Required fields:
#           subject identifier: 'id'
#           file location for left and right: 'path_left' and 'path_right'
#           landmark file location: 'path_left_lmk' and 'path_right_lmk'. Equal to 'none' is no landmark was stored
#           visit number (integer for longitudinal studies): 'visit'

# Step 1: (if needed): isosurface the image files and flip right files; save resulting surfaces in separate directory
# Step 2: Rigid alignment to the first baseline; save resulting surfaces in separate directory
# Step 3: Template estimation from baselines; save template in separate directory
# Step 4: Registration of all surfaces to template; save relevant information in vtk files in separate directory.


class Pipeline:
    def __init__(self, fileIn, dirMain, dirOutput = None, dirSurf='step1_surf', dirRigid = 'step2_rigid',
                 dirTemplate='step3_template', dirReg='step4_registration'):
        self.data = pd.read_csv(fileIn)
        self.dirMain = dirMain
        self.dirSurf = dirSurf
        self.dirRigid = dirRigid
        self.dirTemplate = dirTemplate
        self.dirReg = dirReg
        if dirOutput is None:
            self.dirOutput = dirMain + '/tmp'
        else:
            self.dirOutput = dirOutput
        self.rigTemplate = None
        if 'path_left_surf' not in self.data.columns:
            self.data['path_left_surf'] = 'none'
        if 'path_right_surf' not in self.data.columns:
            self.data['path_right_surf'] = 'none'
        if 'path_left_rigid' not in self.data.columns:
            self.data['path_left_rigid'] = 'none'
        if 'path_right_rigid' not in self.data.columns:
            self.data['path_right_rigid'] = 'none'
        if 'path_right_reg' not in self.data.columns:
            self.data['path_right_reg'] = 'none'

    def Step1_Isosurface(self, templateFile=None, zeroPad=False, axun=False,
                         withBug = False, smooth=False, targetSize=1000):
        ## Reads binary images from original directory, computes and save triangulated surfaces
        sf = surfaces.Surface()
        for record in self.data:
            for side in ('left', 'right'):
                v = diffeo.gridScalars(fileName=record['path_'+side], force_axun=axun, withBug = withBug)
                if zeroPad:
                    v.zeroPad(1)
                t = 0.5 * (v.data.max() + v.data.min())
                # print v.resol
                if smooth:
                    sf.Isosurface(v.data ,value=t ,target=targetSize ,scales=v.resol ,smooth=.75)
                else:
                    sf.Isosurface(v.data ,value=t ,target=targetSize ,scales=v.resol ,smooth=-1)

                sf.edgeRecover()
                # print sf.surfVolume()
                u = path.split(record['path_'+side])
                [nm ,ext] = path.splitext(u[1])
                record['path_'+side+'_surf'] = self.dirMain + '/' + self.dirSurf + '/' + nm + '.vtk'
                sf.saveVTK(record['path_'+side+'_surf'])

    def Step2_Rigid(self):
        tmpl = None
        tmplLmk = None
        cLmk = None
        for record in self.data:
            for side in ('left', 'right'):
                pSurf = record['path_'+side+'_surf']
                pLmk = record['path_'+side+'_lmk']
                if side == 'left':
                    flip = False
                else:
                    flip = True

                if tmpl is None:
                    tmpl = surfaces.Surface(filename = pSurf)
                    if pLmk != 'none':
                        tmplLmk, foo = pointSets.loadlmk(pLmk)
                        R0, T0 = rigidRegistration(surfaces = (tmplLmk, tmpl.vertices),
                                                   translationOnly=True, verb=False, temperature=10.,
                                                   annealing=True)
                        tmplLmk = tmplLmk + T0
                        cLmk = float(tmpl.vertices.shape[0]) / tmplLmk.shape[0]

                u = path.split(pSurf)
                [nm, ext] = path.splitext(u[1])
                sf = surfaces.Surface(filename = pSurf)
                if tmplLmk is not None and pLmk != 'none':
                    y, foo = pointSets.loadlmk(pLmk)
                    R0, T0 = rigidRegistration(surfaces = (y, sf.vertices),  translationOnly=True,
                                               verb=False, temperature=10., annealing=True)
                    y = y+T0
                    (R0, T0) = rigidRegistration(landmarks=(y, tmplLmk, 1.), flipMidPoint=False,
                                                 rotationOnly=False, verb=False,
                                                 temperature=10., annealing=True, rotWeight=1.)
                    yy = np.dot(sf.vertices, R0.T) + T0
                    yyl = np.dot(y, R0.T) + T0
                    (R, T) = rigidRegistration(surfaces = (yy, tmpl.vertices), landmarks=(yyl, tmplLmk, cLmk),
                                               flipMidPoint=flip, rotationOnly=True, verb=False,
                                               temperature=10., annealing=False, rotWeight=1.)
                    T += np.dot(T0, R.T)
                    R = np.dot(R, R0)
                    #yyl = np.dot(y, R.T) + T
                    #pointSets.savelmk(yyl, args.dirOut + '/' + nm + '_reg.lmk')
                else:
                    (R, T) = rigidRegistration(surfaces=(sf.vertices, tmpl.vertices), rotationOnly=False,
                                               flipMidPoint=flip, verb=False,
                                               temperature=10., annealing=True, rotWeight=1.)

            sf.updateVertices(np.dot(sf.vertices, R.T) + T)
            record['path_' + side + '_rigid'] = self.dirMain + '/' + self.dirRigid + '/' + nm + '.vtk'
            sf.saveVTK(record['path_' + side + '_rigid'])


    def Step3_Template(self):
        loggingUtils.setup_default_logging(self.dirOutput, fileName='info.txt', stdOutput=True)

        fv = []
        for record in self.data:
            for side in ('left', 'right'):
                pSurf = record['path_'+side+'_rigid']
                sf = surfaces.Surface(filename=pSurf)
                fv.append((sf))

        fv0 = surfaces.Surface(surf=fv[0])
        vert = np.zeros(fv0.vertices.shape)
        for sf in fv:
            vert += sf.vertices
        vert /= len(fv)
        fv0.updateVertices(vert)

        K1 = kfun.Kernel(name='laplacian', sigma=6.5)
        K2 = kfun.Kernel(name='gauss', sigma=1.0)

        sm = sTemp.SurfaceTemplateParam(timeStep=0.1, KparDiff=K1, KparDist=K2, sigmaError=1., errorType='current')
        f = sTemp.SurfaceTemplate(HyperTmpl=fv0, Targets=fv,
                                  outputDir=self.dirOutput, param=sm, testGradient=False, sgd = 10,
                                  lambdaPrior=.01, maxIter=1000, affine='euclidean', rotWeight=10.,
                                  transWeight=1., scaleWeight=10., affineWeight=100.)
        f.computeTemplate()

        f.fvTmpl.saveVTK(self.dirMain+'/'+self.dirTemplate+'/template.vtk')

    def Step3_Registration(self):
        fv0 = surfaces.Surface(filename=self.dirMain+'/'+self.dirTemplate+'/template.vtk')
        for record in self.data:
            for side in ('left', 'right'):
                pSurf = record['path_'+side+'_rigid']
                fv = surfaces.Surface(filename=pSurf)
                f = smatch.SurfaceMatching(Template=fv0, Target=fv, outputDir=self.dirOutput,
                                           testGradient=False, symmetric=False, saveTrajectories = False,
                                           internalWeight = None, maxIter=2000, affine='none', pplot=False)

                f.optimizeMatching()
                u = path.split(pSurf)
                [nm, ext] = path.splitext(u[1])
                record['path_' + side + '_reg'] = self.dirMain + '/' + self.dirReg + '/' + nm + '.hd5'
                f.saveHD5(record['path_' + side + '_reg'])
