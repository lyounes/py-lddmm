import os
from sys import path as sys_path
sys_path.append('..')
from base import loggingUtils
from base.imageMatchingLDDMM import ImageMatching
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyfftw

pyfftw.config.NUM_THREADS = -1

loggingUtils.setup_default_logging('', stdOutput = True)

## Object kernel
ftemp = '../TestData/Images/3D/2959100_2_2.hdr'
ftarg = '../TestData/Images/3D/3021014_1_2.hdr'

options = {
    'dim':3,
    'timeStep':0.1,
    'algorithm': 'bfgs',
    'sigmaKernel': 0.1,
    'order': 3,
    'kernelSize': 25,
    'typeKernel': 'laplacian',
    'sigmaError': 50.,
    'rescaleFactor':.5,
    'padWidth': 15,
    'affineAlign': 'euclidean',
    'outputDir': '../Output/imageMatchingTest3D',
    'mode': 'debug',
    'regWeight': 1.,
    'maxIter':1000
}
f = ImageMatching(Template=ftemp, Target=ftarg, options=options)

f.optimizeMatching()
plt.ioff()
