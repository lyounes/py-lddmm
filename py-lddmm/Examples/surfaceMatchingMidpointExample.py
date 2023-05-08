import numpy as np
import os
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
    print(os.environ)
import matplotlib.pyplot as plt
from base import surfaces
from base import loggingUtils
from base.surfaceExamples import Sphere, Torus, Ellipse
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.surfaceMatchingMidpoint import SurfaceMatchingMidpoint
from base.surfaceMatching import SurfaceMatching
plt.ion()
loggingUtils.setup_default_logging('', stdOutput = True)
sigmaKernel = 2.5
sigmaDist = 2.
sigmaError = 1.
regweight = 1.
internalWeight = 1.
internalCost = None
## Object kernel

c = np.zeros(3)
d = np.array([11, 0, 0])
# ftemp = Sphere(c, 10)
# ftarg = Torus(c, 10, 4)
ftemp = Ellipse(c, (20, 10, 10))
ftarg = surfaces.Surface(surf= (Sphere(c-d, 10), Sphere(c+d, 10)))

K1 = Kernel(name='laplacian', sigma = sigmaKernel)
vfun = (lambda u: 1+u, lambda u: np.ones(u.shape))

options = {
    'outputDir': '../Output/surfaceMatchingMidpoint/MidpointBalls',
    'mode': 'debug',
    'maxIter': 1000,
    'affine': 'none',
    'regWeight': regweight,
    'KparDiff': K1,
    'KparDist': ('gauss', sigmaDist),
    'sigmaError': sigmaError,
    'errorType': 'varifold',
    'algorithm': 'bfgs',
    'internalWeight': internalWeight,
    'internalCost': internalCost,
    'unreducedWeight': 1.
}

f = SurfaceMatchingMidpoint(Template=ftemp, Target=ftarg, options=options)

f.optimizeMatching()

options['outputDir'] = '../Output/surfaceMatchingMidpoint/EndpointBalls'
f = SurfaceMatching(Template=ftemp, Target=ftarg, options=options)
f.optimizeMatching()
plt.ioff()
plt.show()

