from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')

from base.surfaceExamples import Sphere, Heart
import os
from base.surfaces import Surface
from base.surfaceSection import Hyperplane, SurfaceSection, collect_hyperplanes
from base.surfaceToSectionsMatching import SurfaceToSectionsMatching
from base import loggingUtils
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base.kernelFunctions import Kernel
#plt.ion()


loggingUtils.setup_default_logging('', stdOutput=True)
sigmaKernel = 5.
sigmaDist = 1.
sigmaError = .1
internalWeight = .5
regweight = 1.
internalCost = None #'h1'


fv1 = Heart(zoom=100)
fv0 = Heart(p=1.75, scales=(1.1, 1.5), zoom = 100)
m = fv1.vertices[:,1].min()
M = fv1.vertices[:,1].max()
h = Hyperplane()
target = ()
curves = []
hyperplanes = []
for t in range(1,10):
    ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(0, 1, 0), offset = m + 0.1*t*(M-m)))
    ss2 = ss.curve.connected_components(split=True)
    for c in ss2:
        target += (SurfaceSection(curve=c, hyperplane=ss.hyperplane),)
    # curves += [ss.curve]
    # hyperplanes += [ss.hyperplane]

m = fv1.vertices[:,2].mean()
ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(0, 0, 1), offset = m))
ss2 = ss.curve.connected_components(split=True)
for c in ss2:
    target += (SurfaceSection(curve=c, hyperplane=ss.hyperplane),)
# target += (ss,)
# curves += [ss.curve]
# hyperplanes += [ss.hyperplane]
m = fv1.vertices[:,0].mean()
ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(1, 0, 0), offset = m))
ss2 = ss.curve.connected_components(split=True)
for c in ss2:
    target += (SurfaceSection(curve=c, hyperplane=ss.hyperplane),)
# target += (ss,)
# curves += [ss.curve]
# hyperplanes += [ss.hyperplane]

K1 = Kernel(name='laplacian', sigma=sigmaKernel)

options = {
    'mode': 'debug',
    'maxIter': 2000,
    'affine': 'none',
    'regWeight': regweight,
    'rotWeight': .01,
    'transWeight': .01,
    'scaleWeight': 10.,
    'affineWeight': 100.,
    'KparDiff': K1,
    'KparDist': ('gauss', sigmaDist),
    'sigmaError': sigmaError,
    'errorType': 'measure',
    'algorithm': 'bfgs',
    'pk_dtype': 'float32',
    'internalWeight': internalWeight,
    'internalCost': internalCost,
}

options['outputDir'] = f'../Output/SurfaceToSections'
options['forceClosed'] = True
f = SurfaceToSectionsMatching(Template=fv0, Target= [target, collect_hyperplanes(target)], options=options)

f.optimizeMatching()

