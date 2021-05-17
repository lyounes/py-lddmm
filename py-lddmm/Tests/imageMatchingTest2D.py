from sys import path as sys_path
sys_path.append('..')
from base import loggingUtils
from base.imageMatchingLDDMM import ImageMatching, ImageMatchingParam
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import pyfftw

pyfftw.config.NUM_THREADS = -1

loggingUtils.setup_default_logging('', stdOutput = True)

## Object kernel
ftemp = '../testData/Images/2D/elephant01.pgm'
ftarg = '../testData/Images/2D/bird02.pgm'

sm = ImageMatchingParam(dim=2, timeStep=0.1, algorithm='cg', sigmaKernel = 2, order=3,
                        kernelSize=12, typeKernel='laplacian', sigmaError=50., rescaleFactor=1, padWidth = 15,
                        affineAlign = 'euclidean')
f = ImageMatching(Template=ftemp, Target=ftarg, outputDir='../Output/imageMatchingTest2D',param=sm,
                    testGradient=False, regWeight = 1., maxIter=1000)

f.optimizeMatching()
plt.show()

