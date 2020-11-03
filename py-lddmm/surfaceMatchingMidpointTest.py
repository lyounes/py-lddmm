import numpy as np
from base import surfaces
from base import loggingUtils
from base.surfaces import Surface, Sphere, Torus
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.surfaceMatchingMidpoint import SurfaceMatchingMidpoint
from base.surfaceMatching import SurfaceMatchingParam
import matplotlib.pyplot as plt
plt.ion()
def compute():

    loggingUtils.setup_default_logging('', stdOutput = True)
    sigmaKernel = 2.5
    sigmaDist = 1.
    sigmaError = 1.
    regweight = .01
    internalWeight = 1.
    internalCost = 'h1'
    ## Object kernel

    c = np.zeros(3)
    ftemp = Sphere(c, 10)
    ftarg = Torus(c, 10, 4)
    K1 = Kernel(name='laplacian', sigma = sigmaKernel)

    sm = SurfaceMatchingParam(timeStep=0.1, algorithm='cg', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                              errorType='varifold', internalCost=internalCost)
    f = SurfaceMatchingMidpoint(Template=ftemp, Target=ftarg, outputDir='/Users/younes/Development/Results/TopChange',
                                param=sm, testGradient=True, regWeight=regweight,
                                internalWeight=internalWeight, maxIter=1000, affine= 'none', rotWeight=.01,
                                transWeight = .01, scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


if __name__=="__main__":
    compute()
