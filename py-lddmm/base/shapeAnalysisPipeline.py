import pandas as pd
import numpy as np
import scipy.linalg as la
from . import diffeo
from . import surfaces
from . import pointSets
from .affineRegistration import rigidRegistration, rigidRegistration_multi, saveRigid
from . import surfaceTemplate as sTemp
from . import surfaceMatching as smatch
from . import loggingUtils, kernelFunctions as kfun
from os import path
import os






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
        self.dirSurf = self.dirMain + '/' + dirSurf
        self.dirRigid = self.dirMain + '/' + dirRigid
        self.dirTemplate = self.dirMain + '/' + dirTemplate
        self.dirReg = self.dirMain + '/' + dirReg
        self.fileIn = fileIn
        if dirOutput is None:
            self.dirOutput = dirMain + '/tmp'
        else:
            self.dirOutput = self.dirMain + '/' + dirOutput
        self.rigTemplate = None
        for dir in (self.dirMain, self.dirOutput, self.dirSurf, self.dirRigid, self.dirTemplate, self.dirReg):
            if not os.access(dir,os.W_OK):
                if os.access(dir,os.F_OK):
                    print('Cannot save in ' + dir)
                    return
                else:
                    os.makedirs(dir)
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
        for index, record in self.data.iterrows():
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
                #sf.updateVertices(np.dot(sf.vertices, v.affine[:3,:3].T) + v.affine[:3,3])
                #sf.updateVertices(np.dot(sf.vertices-v.affine[np.newaxis, :3,3], la.inv(v.affine[:3,:3]).T))
                # print sf.surfVolume()
                u = path.split(record['path_'+side])
                [nm ,ext] = path.splitext(u[1])
                #record[1]['path_'+side+'_surf'] = self.dirSurf + '/' + nm + '.vtk'
                self.data.at[index, 'path_'+side+'_surf'] =  self.dirSurf + '/' + nm + '.vtk'
                print(self.dirSurf + '/' + nm + '.vtk')
                sf.saveVTK(self.data.at[index, 'path_'+side+'_surf'])
        self.data.to_csv(self.fileIn)

    def Step2_Rigid_old(self):
        tmpl = None
        tmplLmk = None
        cLmk = None
        for index,record in self.data.iterrows():
            for side in ('left', 'right'):
                pSurf = record['path_'+side+'_surf']
                if 'path_'+side+'_lmk' in record.index:
                    pLmk = record['path_'+side+'_lmk']
                else:
                    pLmk = 'none'
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
                                               temperature=.1, annealing=True, rotWeight=.001)

                sf.updateVertices(np.dot(sf.vertices, R.T) + T)
                print(self.dirRigid + '/' + nm + '.vtk')
                self.data.at[index, 'path_' + side + '_rigid'] = self.dirRigid + '/' + nm + '.vtk'
                sf.saveVTK(self.data.at[index, 'path_' + side + '_rigid'])
        self.data.to_csv(self.fileIn)

    def Step2_Rigid(self):
        tmpl_left = None
        tmpl_v = None
        for index,record in self.data.iterrows():
            pSurf_left = record['path_left_surf']
            pSurf_right = record['path_right_surf']

            if tmpl_left is None:
                tmpl_left = surfaces.Surface(filename = pSurf_left)
                tmpl_right = surfaces.Surface(filename = pSurf_right)
                tmpl_v = [tmpl_left.vertices, tmpl_right.vertices]

            u = path.split(pSurf_left)
            [nm_left, ext] = path.splitext(u[1])
            u = path.split(pSurf_right)
            [nm_right, ext] = path.splitext(u[1])
            sf_left = surfaces.Surface(filename = pSurf_left)
            sf_right = surfaces.Surface(filename = pSurf_right)
            sf_v = [sf_left.vertices, sf_right.vertices]
            (R, T) = rigidRegistration_multi((sf_v, tmpl_v), rotationOnly=True,
                                       verb=True, temperature=.1, annealing=False, rotWeight=.001)

            sf_left.updateVertices(np.dot(sf_left.vertices, R.T) + T)
            sf_right.updateVertices(np.dot(sf_right.vertices, R.T) + T)
            print(self.dirRigid + '/' + nm_left + '.vtk')
            self.data.at[index, 'path_left_rigid'] = self.dirRigid + '/' + nm_left + '.vtk'
            self.data.at[index, 'path_right_rigid'] = self.dirRigid + '/' + nm_right + '.vtk'
            sf_left.saveVTK(self.data.at[index, 'path_left_rigid'])
            sf_right.saveVTK(self.data.at[index, 'path_right_rigid'])
        self.data.to_csv(self.fileIn)

    def Step3_Template(self):
        loggingUtils.setup_default_logging(self.dirOutput, fileName='info.txt', stdOutput=True)

        fv = []
        for index, record in self.data.iterrows():
            for side in ('left', 'right'):
                pSurf = record.at[index, 'path_'+side+'_rigid']
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

        f.fvTmpl.saveVTK(self.dirTemplate+'/template.vtk')

    def Step4_Registration(self):
        fv0 = surfaces.Surface(filename=self.dirTemplate+'/template.vtk')
        for index, record in self.data.iterrows():
            for side in ('left', 'right'):
                pSurf = record['path_'+side+'_rigid']
                fv = surfaces.Surface(filename=pSurf)
                f = smatch.SurfaceMatching(Template=fv0, Target=fv, outputDir=self.dirOutput,
                                           testGradient=False, symmetric=False, saveTrajectories = False,
                                           internalWeight = None, maxIter=2000, affine='none', pplot=False)

                f.optimizeMatching()
                u = path.split(pSurf)
                [nm, ext] = path.splitext(u[1])
                self.data.at[index, 'path_' + side + '_reg'] = self.dirReg + '/' + nm + '.hd5'
                f.saveHD5(self.data.at[index, 'path_' + side + '_reg'])
