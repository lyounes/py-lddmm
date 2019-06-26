from base import surfaces
from base import examples
from base.kernelFunctions import Kernel
from base import surfaceMatchingNormalExtremities as SMNE
import matplotlib
matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from base import loggingUtils

if __name__ == "__main__":
    plt.ion()
    loggingUtils.setup_default_logging('', stdOutput = True)
    fv1, fv2 = examples.split_ellipsoid(a=0.5, b=0.5, c=0.25)
    sigmaKernel = 1.
    sigmaDist = 1.
    sigmaError = .1
    internalWeight = 50.
    internalCost = 'h1'
    K1 = Kernel(name='laplacian', sigma=sigmaKernel)

    outputDir = '/Users/younes/Development/results/splitEllipsoid'

    sm = SMNE.SurfaceMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, sigmaDist=sigmaDist, internalCost = internalCost,
              sigmaError=sigmaError, errorType='varifold')


    f = SMNE.SurfaceMatching(Template=fv1, Target=fv2, outputDir=outputDir, param=sm, regWeight=1.,
            saveTrajectories=True, symmetric=False, pplot=True,
            affine='none', testGradient=False, internalWeight=internalWeight, affineWeight=1e3, maxIter_cg=1000,
            maxIter_al=5, mu=1e-4)
    f.optimizeMatching()



