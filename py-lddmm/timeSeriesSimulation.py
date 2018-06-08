import numpy as np
import logging
import loggingUtils
import surfaces
from surfaces import *
from kernelFunctions import *
import surfaceMatching
#import secondOrderMatching as match

def compute(Atrophy=False):
    if Atrophy:
        import surfaceTimeSeriesAtrophy as match
    else:
        import surfaceTimeSeries as match 

    outputDir = '/Users/younes/Development/Results/simulationAtrophyLargeNoiseWithoutConstraint'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput = False)
    else:
        loggingUtils.setup_default_logging()

    rdir = '/Users/younes/Development/Data/sculptris/AtrophyLargeNoise/' ;
    fv0 = surfaces.Surface(filename=rdir + 'baseline.vtk')
    fv1 = [fv0]
    for k in range(5):
        fv1 += [surfaces.Surface(filename=rdir+'followUp'+str(2*k+1)+'.vtk')]

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 6.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=.5, errorType='varifold')
    if Atrophy:
        f = match.SurfaceMatching(Template=fv0, Targets=fv1, outputDir=outputDir, param=sm, regWeight=.1,
                                affine='euclidean', testGradient=True, affineWeight=.1,  maxIter_cg=100, mu=0.0001)
    else:
       f = match.SurfaceMatching(Template=fv0, Targets=fv1, outputDir=outputDir, param=sm, regWeight=.1,
                                affine='euclidean', testGradient=True, affineWeight=.1,  maxIter=1000)
 
    #, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute(True)

