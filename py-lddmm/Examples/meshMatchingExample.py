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
from scipy import stats
import pygalmesh
from base.curveExamples import Circle
from base.surfaceExamples import Sphere
import logging
from base import loggingUtils
from base.meshes import Mesh
from base.kernelFunctions import Kernel
from base.meshMatching import MeshMatching
from base.secondOrderMeshMatching import SecondOrderMeshMatching
from base.imageMatchingLDDMM import ImageMatching
from base.meshExamples import TwoBalls, TwoDiscs, MoGCircle, TwoEllipses
# import pykeops
# pykeops.clean_pykeops()
plt.ion()

model = 'Spheres'
#model = 'Ellipses'
#model = 'EllipsesTranslation'
#model = 'Circles'
secondOrder = False
shrink = False
shrinkRatio = 0.75
eulerian = False
s=20

if secondOrder:
    typeCost = 'LDDMM'
    order = '_SO_'
    internalCost = None
else:
    typeCost = 'LDDMM'
    internalCost = 'elastic_energy'
    order = ''

if shrink:
    sh = f'_shrink_{shrinkRatio:.2}_'
else:
    sh = ''

def compute(model):
    if internalCost is None:
        sigmaKernel = 1.
        sigmaDist = 5.
        sigmaError = .1
        regweight = 1.
    else:
        sigmaKernel = .1
        sigmaDist = 5.
        sigmaError = .1
        regweight = 1.

    internalWeight = 10.

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
        sigmaDist = 1.
        sigmaKernel = 1.
    elif model == 'Ellipses':
        fv0 = TwoEllipses(Boundary_a=14, Boundary_b=6, smallRadius=0.25)
        fv1 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.4, translation=[0.25, -0.1])
        sigmaDist = 1.
        sigmaKernel = 1.
        ftemp = fv0
        ftarg = fv1
    elif model == 'EllipsesTranslation':
        fv0 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.3, translation=[0.3, 0], volumeRatio=5000)
        fv1 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.3, translation=[-0.3, 0], volumeRatio=5000)
        sigmaDist = [2., 1.]
        sigmaKernel = 1.
        sigmaError = .1
        ftemp = fv0
        ftarg = fv1
    elif model == 'Spheres':
        ftemp = TwoBalls(largeRadius=10, smallRadius=4.5)
        ftarg = TwoBalls(largeRadius=15, smallRadius=3)
        sigmaKernel = 2.
        sigmaDist = 1.
    elif model == 'SpheresSmall':
        ftemp = TwoBalls(largeRadius=10, smallRadius=4.5, facet_distance = 0.05,radius_ball=1.0, circumradius = 1.0)
        ftarg = TwoBalls(largeRadius=15, smallRadius=3, facet_distance = 0.05, radius_ball=1.0, circumradius = 1.0)
        logging.info(f'Number of faces in template: {ftemp.faces.shape[0]}') 
        sigmaKernel = 2.
        sigmaDist = 1.
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

    epsilon = 0.5
    alpha = np.sqrt(1 - epsilon)
    beta = (np.sqrt(1 - epsilon + ftemp.image.shape[1] * epsilon) - alpha) / np.sqrt(ftemp.image.shape[1])
    ftemp.updateImage(alpha * ftemp.image + beta * ftemp.image.sum(axis=1)[:, None])
    ftarg.updateImage(alpha * ftarg.image + beta * ftarg.image.sum(axis=1)[:, None])

    s0 = 1e9
    image1 = np.zeros(ftemp.image.shape)
    for k in range(ftemp.faces.shape[0]):
        image1[k, :] = stats.dirichlet.rvs(ftemp.image[k, :] * s0)
    ftemp.updateImage(image1)

    image1 = np.zeros(ftarg.image.shape)
    for k in range(ftarg.faces.shape[0]):
        image1[k, :] = stats.dirichlet.rvs(ftarg.image[k, :] * s)
    ftarg.updateImage(image1)


    sw = 10*s
    # w = np.zeros(ftemp.faces.shape[0])
    # for k in range(ftemp.faces.shape[0]):
    #     w[k] = stats.gamma.rvs(sw * ftemp.volumes[k]) / (sw * ftemp.volumes[k])
    # ftemp.updateWeights(w)
    w = np.zeros(ftarg.faces.shape[0])
    for k in range(ftarg.faces.shape[0]):
        w[k] = stats.gamma.rvs(sw * ftarg.weights[k], loc=0, scale=1/sw) #/ (sw * ftarg.weights[k])
    ftarg.updateWeights(w)

    ## Object kernel

    if eulerian:
        resolution = 0.1
        margin = 5
        imin = ftemp.vertices.min(axis=0) - margin
        imax = ftemp.vertices.max(axis=0) + margin
        imgTemp = ftemp.toImage(resolution=resolution, index=0, margin=5, bounds=[imin,imax])
        imgTarg = ftarg.toImage(resolution=resolution, index=0, margin=5, bounds=[imin,imax])
        ## True kernel size = resolution * sig * kernelSize
        sig = sigmaKernel / resolution
        sigDist = sigmaDist[0] / resolution
        logging.info(f"sigma for image matching: {sig:.2f}, {sigDist:.2f}")

        options = {
            'dim': ftarg.dim,
            'timeStep': 0.1,
            'algorithm': 'bfgs',
            'sigmaKernel': sig,
            'order': 3,
            'typeKernel': 'laplacian',
            'sigmaError': 5.,
            'rescaleFactor': 1.,
            'padWidth': 15,
            'affineAlign': None,
            'outputDir': '../Output/meshMatchingTestImageComparison/'+model+order+sh+f'_s_{s:.0f}',
            'mode': 'normal',
            'normalize':255.,
            'regWeight': 1.,
            'sigmaSmooth': sigDist,
            'maxIter': 1000
        }
        f = ImageMatching(Template=imgTemp, Target=imgTarg, options=options)

        f.restartRate = 50
    else:
        K1 = Kernel(name='laplacian', sigma=sigmaKernel)
        options = {
            'outputDir': '../Output/meshMatchingTest/' + model + order + sh+f'_s_{s:.0f}',
            'mode': 'debug',
            'maxIter': 2000,
            'affine': 'none',
            'rotWeight': 100,
            'transWeight': 10,
            'scaleWeight': 1.,
            'affineWeight': 100.,
            'KparDiff': K1,
            'KparDist': ('gauss', sigmaDist),
            'KparIm': ('euclidean', 1.),
            'sigmaError': sigmaError,
            'errorType': 'measure',
            'algorithm': 'bfgs',
            'internalCost': internalCost,
            'internalWeight': internalWeight,
            'regWeight': regweight,
            'lame_mu': 4.,
            'lame_lambda': 5.,
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
