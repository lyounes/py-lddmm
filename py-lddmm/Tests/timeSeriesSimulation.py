from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.loggingUtils import setup_default_logging
from base.kernelFunctions import Kernel
from base import surfaceMatching
from base import surfaces as surfaces, surfaceTimeSeries as match


setup_default_logging('../Output/Temp', fileName='info', stdOutput = True)

rdir = '../TestData/AtrophyLargeNoise/'
fv0 = surfaces.Surface(surf=rdir + 'baseline.vtk')
fv1 = [fv0]
for k in range(5):
    fv1 += [surfaces.Surface(surf=rdir + 'followUp' + str(2 * k + 1) + '.vtk')]
outputDir = '../Output/timeSeriesNoAtrophy'

K1 = Kernel(name='laplacian', sigma = 6.5, order=4)
sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDist=('gauss', 2.5),
                                          sigmaError=.5, errorType='current')

f = match.SurfaceTimeMatching(Template=fv0, Target=fv1, outputDir=outputDir, param=sm, regWeight=.1,
                        affine='euclidean', testGradient=True, affineWeight=.1,  maxIter=1000)

f.optimizeMatching()



