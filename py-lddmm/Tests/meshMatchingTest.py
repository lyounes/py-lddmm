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
import pygalmesh
from base.curveExamples import Circle
from base.surfaceExamples import Sphere
from base import loggingUtils
from base.meshes import Mesh
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.meshMatching import MeshMatching, MeshMatchingParam
from base.meshExamples import TwoBalls, TwoDiscs, MoGCircle
import pykeops
pykeops.clean_pykeops()
plt.ion()

model = 'GaussCenters'

def compute(model):
    loggingUtils.setup_default_logging('../Output', stdOutput = True)
    sigmaKernel = 5.
    sigmaDist = 5.
    sigmaError = 1.
    regweight = 1.
    if model=='Circles':
        f = Circle(radius = 10., targetSize=250)
        fv0 = Mesh(f, volumeRatio=5000)
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv0.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 20, dtype=float)
        fv0.image = np.zeros((fv0.faces.shape[0], 2))
        fv0.image[:, 0] = (imagev[fv0.faces[:, 0]] + imagev[fv0.faces[:, 1]] + imagev[fv0.faces[:, 2]]) / 3
        fv0.image[:, 1] = 1 - fv0.image[:, 0]

        f = Circle(radius = 12, targetSize=250)
        fv1 = Mesh(f, volumeRatio=5000)
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv1.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 10, dtype=float)
        fv1.image = np.zeros((fv1.faces.shape[0], 2))
        fv1.image[:, 0] = (imagev[fv1.faces[:, 0]] + imagev[fv1.faces[:, 1]] + imagev[fv1.faces[:, 2]]) / 3
        fv1.image[:, 1] = 1 - fv1.image[:, 0]
        ftemp = fv0
        ftarg = fv1
    elif model == 'Spheres':
        mesh = pygalmesh.generate_mesh(
            pygalmesh.Ball([0.0, 0.0, 0.0], 10.0),
            min_facet_angle=30.0,
            max_radius_surface_delaunay_ball=.75,
            max_facet_distance=0.025,
            max_circumradius_edge_ratio=2.0,
            max_cell_circumradius=.75,  # lambda x: abs(np.sqrt(np.dot(x, x)) - 0.5) / 5 + 0.025,
            verbose=False
        )
        fv0 = Mesh([np.array(mesh.cells[1].data, dtype=int), np.array(mesh.points, dtype=float)])
        print(fv0.vertices.shape)
        c0 = np.array([0,0,0])
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv0.vertices - c0[None, :]) ** 2).sum(axis=1) < 20, dtype=float)
        fv0.image = np.zeros((fv0.faces.shape[0], 2))
        fv0.image[:, 0] = (imagev[fv0.faces[:, 0]] + imagev[fv0.faces[:, 1]] + imagev[fv0.faces[:, 2]]
                           + imagev[fv0.faces[:, 3]]) / 4
        fv0.image[:, 1] = 1 - fv0.image[:, 0]

        mesh1 = pygalmesh.generate_mesh(
            pygalmesh.Ball([0.0, 0.0, 0.0], 12.0),
            min_facet_angle=30.0,
            max_radius_surface_delaunay_ball=.75,
            max_facet_distance=0.025,
            max_circumradius_edge_ratio=2.0,
            max_cell_circumradius=.75,  # lambda x: abs(np.sqrt(np.dot(x, x)) - 0.5) / 5 + 0.025,
            verbose=False
        )
        fv1 = Mesh([np.array(mesh1.cells[1].data, dtype=int), np.array(mesh1.points, dtype=float)])
        print(fv1.vertices.shape)
        # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        imagev = np.array(((fv1.vertices - c0[None, :]) ** 2).sum(axis=1) < 10, dtype=float)
        fv1.image = np.zeros((fv1.faces.shape[0], 2))
        fv1.image[:, 0] = (imagev[fv1.faces[:, 0]] + imagev[fv1.faces[:, 1]] + imagev[fv1.faces[:, 2]]
                           + imagev[fv1.faces[:, 3]]) / 4
        fv1.image[:, 1] = 1 - fv1.image[:, 0]
        ftemp = fv0
        ftarg = fv1
    elif model == 'GaussCenters':
        ftemp = MoGCircle(largeRadius=10)
        ftarg = MoGCircle(largeRadius=12, centers=1.2*ftemp.GaussCenters, typeProb=ftemp.typeProb, alpha=ftemp.alpha)
    else:
        return

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = sigmaKernel)

    sm = MeshMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, KparDist=('gauss', sigmaDist),
                              KparIm=('gauss', .1), sigmaError=sigmaError)
    sm.KparDiff.pk_dtype = 'float32'
    sm.KparDist.pk_dtype = 'float32'
    f = MeshMatching(Template=ftemp, Target=ftarg, outputDir='../Output/meshMatchingTest/'+model,param=sm,
                        testGradient=False, regWeight = regweight, maxIter=1000,
                     affine= 'none', rotWeight=.01, transWeight = .01,
                        scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


compute(model)
