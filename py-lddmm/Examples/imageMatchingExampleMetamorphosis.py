from sys import path as sys_path
sys_path.append('..')
from base import loggingUtils
from base.imageMatchingMetamorphosis import Metamorphosis
import matplotlib
matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
import pyfftw

pyfftw.config.NUM_THREADS = -1

loggingUtils.setup_default_logging('', stdOutput = True)

ftemp = '../testData/Images/2D/faces/s23/5.pgm'
ftarg = '../testData/Images/2D/faces/s30/10.pgm'

options = {
    'dim':2,
    'timeStep':0.05,
    'algorithm': 'bfgs',
    'sigmaKernel': 5,
    'order': 3,
    'kernelSize': 24,
    'typeKernel': 'laplacian',
    'sigmaError': 10.,
    'sigmaSmooth': 0.5,
    'rescaleFactor':1.,
    'padWidth': 10,
    'affineAlign': 'euclidean',
    'outputDir': '../Output/imageMatchingExampleMeta',
    'mode': 'normal',
    'regWeight': 1.,
    'maxIter':1000
}

f = Metamorphosis(Template=ftemp, Target=ftarg, options=options)

f.optimizeMatching()
plt.ioff()
plt.show()

