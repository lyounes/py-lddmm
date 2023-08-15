from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygalmesh
from base.curveExamples import Circle
from base.surfaceExamples import Sphere
from base import loggingUtils
from base.meshes import Mesh
from base.kernelFunctions import Kernel
from base.meshMatching import MeshMatching
from base.secondOrderMeshMatching import SecondOrderMeshMatching
from base.meshExamples import TwoBalls, TwoDiscs, MoGCircle, TwoEllipses
# import pykeops
# pykeops.clean_pykeops()
plt.ion()

model = 'Spheres'
secondOrder = False
shrink = False
shrinkRatio = 0.75

if secondOrder:
    typeCost = 'LDDMM'
    order = '_SO_'
    internalCost = None
else:
    typeCost = 'LDDMM'
    internalCost = None #'divergence'
    order = ''

if shrink:
    sh = f'_shrink_{shrinkRatio:.2}_'
else:
    sh = ''

def compute(model):
    if internalCost is None:
        sigmaKernel = 1.
        sigmaDist = 5.
        sigmaError = .5
        regweight = 1.
    else:
        sigmaKernel = 1.
        sigmaDist = 5.
        sigmaError = .5
        regweight = 1.

    internalWeight = .1

    loggingUtils.setup_default_logging('../Output', stdOutput = True)
    if model=='Circles':
        fv0 = TwoDiscs(largeRadius=10, smallRadius=4.5)
        # f = Circle(radius = 10., targetSize=1000)
        # fv0 = Mesh(f, volumeRatio=1000)
        # # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        # imagev = np.array(((fv0.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 20, dtype=float)
        # fv0.image = np.zeros((fv0.faces.shape[0], 2))
        # fv0.image[:, 0] = (imagev[fv0.faces[:, 0]] + imagev[fv0.faces[:, 1]] + imagev[fv0.faces[:, 2]]) / 3
        # fv0.image[:, 1] = 1 - fv0.image[:, 0]

        fv1 = TwoDiscs(largeRadius=12, smallRadius=3)
        # f = Circle(radius = 12, targetSize=1000)
        # fv1 = Mesh(f, volumeRatio=5000)
        # # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        # imagev = np.array(((fv1.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < 10, dtype=float)
        # fv1.image = np.zeros((fv1.faces.shape[0], 2))
        # fv1.image[:, 0] = (imagev[fv1.faces[:, 0]] + imagev[fv1.faces[:, 1]] + imagev[fv1.faces[:, 2]]) / 3
        # fv1.image[:, 1] = 1 - fv1.image[:, 0]
        ftemp = fv0
        ftarg = fv1
    elif model == 'Ellipses':
        fv0 = TwoEllipses(Boundary_a=14, Boundary_b=6, smallRadius=.25)
        fv1 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.4, translation=[0.1, -0.1])
        ftemp = fv0
        ftarg = fv1
    elif model == 'Spheres':
        ftemp = TwoBalls(largeRadius=10, smallRadius=4.5)
        # mesh = pygalmesh.generate_mesh(
        #     pygalmesh.Ball([0.0, 0.0, 0.0], 10.0),
        #     min_facet_angle=30.0,
        #     max_radius_surface_delaunay_ball=.75,
        #     max_facet_distance=0.025,
        #     max_circumradius_edge_ratio=2.0,
        #     max_cell_circumradius=.75,  # lambda x: abs(np.sqrt(np.dot(x, x)) - 0.5) / 5 + 0.025,
        #     verbose=False
        # )
        # fv0 = Mesh([np.array(mesh.cells[1].data, dtype=int), np.array(mesh.points, dtype=float)])
        # print(fv0.vertices.shape)
        # c0 = np.array([0,0,0])
        # # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        # imagev = np.array(((fv0.vertices - c0[None, :]) ** 2).sum(axis=1) < 20, dtype=float)
        # fv0.image = np.zeros((fv0.faces.shape[0], 2))
        # fv0.image[:, 0] = (imagev[fv0.faces[:, 0]] + imagev[fv0.faces[:, 1]] + imagev[fv0.faces[:, 2]]
        #                    + imagev[fv0.faces[:, 3]]) / 4
        # fv0.image[:, 1] = 1 - fv0.image[:, 0]

        ftarg = TwoBalls(largeRadius=12, smallRadius=3)
        # mesh1 = pygalmesh.generate_mesh(
        #     pygalmesh.Ball([0.0, 0.0, 0.0], 12.0),
        #     min_facet_angle=30.0,
        #     max_radius_surface_delaunay_ball=.75,
        #     max_facet_distance=0.025,
        #     max_circumradius_edge_ratio=2.0,
        #     max_cell_circumradius=.75,  # lambda x: abs(np.sqrt(np.dot(x, x)) - 0.5) / 5 + 0.025,
        #     verbose=False
        # )
        # fv1 = Mesh([np.array(mesh1.cells[1].data, dtype=int), np.array(mesh1.points, dtype=float)])
        # print(fv1.vertices.shape)
        # # imagev = np.array(((mesh2.vertices - np.array([0.5, 0.5])[None, :])**2).sum(axis=1) < 0.1, dtype=float)
        # imagev = np.array(((fv1.vertices - c0[None, :]) ** 2).sum(axis=1) < 10, dtype=float)
        # fv1.image = np.zeros((fv1.faces.shape[0], 2))
        # fv1.image[:, 0] = (imagev[fv1.faces[:, 0]] + imagev[fv1.faces[:, 1]] + imagev[fv1.faces[:, 2]]
        #                    + imagev[fv1.faces[:, 3]]) / 4
        # fv1.image[:, 1] = 1 - fv1.image[:, 0]
        # ftemp = fv0
        # ftarg = fv1
    elif model == 'GaussCenters':
        sigmaKernel = 5.
        sigmaDist = 5.
        sigmaError = .5
        ftemp = MoGCircle(largeRadius=10, nregions=50)
        centers = ftemp.GaussCenters + .5 * np.random.normal(0,1,ftemp.GaussCenters.shape)
        ftarg = MoGCircle(largeRadius=12, centers=1.2*centers, typeProb=ftemp.typeProb, alpha=ftemp.alpha)
    else:
        return

    ## Transform image
    epsilon = 0.95
    alpha = np.sqrt(1 - epsilon)
    beta = (np.sqrt(1 - epsilon + ftemp.image.shape[1] * epsilon) - alpha) / np.sqrt(ftemp.image.shape[1])
    ftemp.updateImage(alpha * ftemp.image + beta * ftemp.image.sum(axis=1)[:, None])
    ftarg.updateImage(alpha * ftarg.image + beta * ftarg.image.sum(axis=1)[:, None])

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = sigmaKernel)
    options = {
        'outputDir': '../Output/meshMatchingTest/'+model+order+sh,
        'mode': 'normal',
        'maxIter': 2000,
        'affine': 'none',
        'rotWeight': 100,
        'transWeight': 10,
        'scaleWeight': 1.,
        'affineWeight': 100.,
        'KparDiff': K1,
        'KparDist': ('gauss', sigmaDist),
        'KparIm': ('gauss', .1),
        'sigmaError': sigmaError,
        'errorType': 'measure',
        'algorithm': 'bfgs',
        'internalCost': internalCost,
        'internalWeight': internalWeight,
        'regWeight': regweight,
        'pk_dtype': 'float64'
    }

    if shrink:
        ftemp = ftemp.shrinkTriangles(ratio=shrinkRatio)
        ftarg = ftarg.shrinkTriangles(ratio=shrinkRatio)

    if secondOrder:
        f = SecondOrderMeshMatching(Template=ftemp, Target=ftarg, options=options)
    else:
        f = MeshMatching(Template=ftemp, Target=ftarg, options=options)

    f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


compute(model)
