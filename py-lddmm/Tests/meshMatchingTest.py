from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
    print(os.environ)
import matplotlib.pyplot as plt
import numpy as np
from base.curveExamples import Circle
from base.surfaceExamples import Sphere
from base import loggingUtils
from base.meshes import Mesh
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.meshMatching import MeshMatching, MeshMatchingParam
import pykeops
pykeops.clean_pykeops()
plt.ion()

model = 'Spheres'

def compute(model):
    loggingUtils.setup_default_logging('../Output', stdOutput = True)
    sigmaKernel = 5.
    sigmaDist = 5.
    sigmaError = 1.
    regweight = 0.1
    if model=='Circles':
        f = Circle(radius = 10.)
        fv0 = Mesh(f)
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv0.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 20, dtype=float)
        fv0.image = np.zeros((fv0.faces.shape[0], 2))
        fv0.image[:, 0] = (imagev[fv0.faces[:, 0]] + imagev[fv0.faces[:, 1]] + imagev[fv0.faces[:, 2]]) / 3
        fv0.image[:, 1] = 1 - fv0.image[:, 0]

        f = Circle(radius = 12)
        fv1 = Mesh(f)
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv1.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 10, dtype=float)
        fv1.image = np.zeros((fv1.faces.shape[0], 2))
        fv1.image[:, 0] = (imagev[fv1.faces[:, 0]] + imagev[fv1.faces[:, 1]] + imagev[fv1.faces[:, 2]]) / 3
        fv1.image[:, 1] = 1 - fv1.image[:, 0]
        ftemp = fv0
        ftarg = fv1
    elif model == 'Spheres':
        f = Sphere(radius=10.)
        fv0 = Mesh(f)
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv0.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 20, dtype=float)
        fv0.image = np.zeros((fv0.faces.shape[0], 2))
        fv0.image[:, 0] = (imagev[fv0.faces[:, 0]] + imagev[fv0.faces[:, 1]] + imagev[fv0.faces[:, 2]]
                           + imagev[fv0.faces[:, 3]]) / 4
        fv0.image[:, 1] = 1 - fv0.image[:, 0]

        f = Sphere(radius=12)
        fv1 = Mesh(f)
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv1.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 10, dtype=float)
        fv1.image = np.zeros((fv1.faces.shape[0], 2))
        fv1.image[:, 0] = (imagev[fv1.faces[:, 0]] + imagev[fv1.faces[:, 1]] + imagev[fv1.faces[:, 2]]
                           + imagev[fv1.faces[:, 3]]) / 4
        fv1.image[:, 1] = 1 - fv1.image[:, 0]
        ftemp = fv0
        ftarg = fv1
    else:
        return

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = sigmaKernel)

    sm = MeshMatchingParam(timeStep=0.1, algorithm='cg', KparDiff=K1, KparDist=('gauss', sigmaDist),
                              sigmaError=sigmaError)
    sm.KparDiff.pk_dtype = 'float64'
    sm.KparDist.pk_dtype = 'float64'
    f = MeshMatching(Template=ftemp, Target=ftarg, outputDir='../Output/meshMatchingTest/'+model,param=sm,
                        testGradient=False, regWeight = regweight, maxIter=1000,
                     affine= 'none', rotWeight=.01, transWeight = .01,
                        scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


compute(model)
