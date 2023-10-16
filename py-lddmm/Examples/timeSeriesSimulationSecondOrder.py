from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.loggingUtils import setup_default_logging
from base.kernelFunctions import Kernel
from base import surfaceMatching
from base import surfaces as surfaces
from base.secondOrderSurfaceTimeMatching import SecondOrderSurfaceTimeMatching


setup_default_logging('../Output/Temp', fileName='info', stdOutput = True)

rdir = '../TestData/AtrophyLargeNoise/'
fv0 = surfaces.Surface(surf=rdir + 'baseline.vtk')
# fv1 = [fv0]
fv1 = []
for k in range(2):
    fv1 += [surfaces.Surface(surf=rdir + 'followUp' + str(2 * k + 1) + '.vtk')]
outputDir = '../Output/timeSeriesNoAtrophy'

K1 = Kernel(name='gauss', sigma = 6.5, order=4)
## typeRegression = 'geodesic' or 'spline' or 'spline2' (which is 'geodesic' and 'spline' together
options = {
    'outputDir': outputDir,
    'mode': 'normal',
    'maxIter': 2000,
    'affine': 'none',
    'regWeight': 1.,
    'Landmarks': None,
    'affineWeight': .1,
    'KparDiff': K1,
    'KparDist': ('gauss', 2.5),
    'sigmaError': 0.5,
    'errorType': 'current',
    'algorithm': 'bfgs',
    'typeRegression': 'geodesic',
    'internalWeight': 0.,
    'saveRate': 10,
    'internalCost': None
}

f = SecondOrderSurfaceTimeMatching(Template=fv0, Target=fv1, options=options)
f.optimizeMatching()



