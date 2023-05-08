from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.loggingUtils import setup_default_logging
from base.kernelFunctions import Kernel
from base import surfaces as surfaces, surfaceTimeSeries as match


setup_default_logging('../Output/Temp', fileName='info', stdOutput = True)

rdir = '../TestData/AtrophyLargeNoise/'
fv0 = surfaces.Surface(surf=rdir + 'baseline.vtk')
fv1 = [fv0]
for k in range(5):
    fv1 += [surfaces.Surface(surf=rdir + 'followUp' + str(2 * k + 1) + '.vtk')]
outputDir = '../Output/timeSeriesNoAtrophy'

K1 = Kernel(name='laplacian', sigma = 6.5, order=4)
options = {
    'outputDir': outputDir,
    'mode': 'debug',
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
    'pk_dtype': 'float64',
    'internalWeight': 0.,
    'internalCost': None
}


f = match.SurfaceTimeMatching(Template=fv0, Target=fv1, options=options)

f.optimizeMatching()



