from sys import path as sys_path
sys_path.append('..')
from base import loggingUtils
from base.imageMatchingLDDMM import ImageMatching
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import pyfftw

pyfftw.config.NUM_THREADS = -1

loggingUtils.setup_default_logging('', stdOutput = True)

## Object kernel
# ftemp = '../TestData/Images/2D/l2nr011.tif'
# ftarg = '../TestData/Images/2D/l2nr013.tif'
ftemp = '../TestData/Images/2D/Hand_0000002.jpg'
ftarg = '../TestData/Images/2D/Hand_0000008.jpg'

options = {
    'dim':2,
    'timeStep':0.1,
    'algorithm': 'bfgs',
    'sigmaKernel': 5,
    'orderKernel': 3,
    'kernelSize': 25,
    'typeKernel': 'laplacian',
    'sigmaError': 50.,
    'rescaleFactor':1.,
    'padWidth': 15,
    'affineAlign': 'euclidean',
    'outputDir': '../Output/imageMatchingTest2D',
    'mode': 'debug',
    'regWeight': 1.,
    'maxIter':1000
}
f = ImageMatching(Template=ftemp, Target=ftarg, options=options)

f.restartRate = 50

f.optimizeMatching()
plt.show()

