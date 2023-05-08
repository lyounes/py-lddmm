from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')

from base.surfaces import Surface
from base.kernelFunctions import Kernel
#from surfaceMatching import SurfaceMatchingParam
from base.surfaceTemplate import SurfaceTemplate
from base import loggingUtils


def main():
    fv = []
    fv0 = None
    #fv0 = Surface(surf ='../TestData/forTemplate/HyperTemplate.vtk')
    for k in range(10):
        fv.append(Surface(surf ='../TestData/forTemplate/Target'+str(k)+'.vtk'))

    loggingUtils.setup_default_logging('../Output/surfaceTemplate2', fileName='info.txt', stdOutput = True)
    K1 = Kernel(name='laplacian', sigma = 6.5)
    K2 = Kernel(name='gauss', sigma = 1.0)

    options = {
        'mode': 'debug',
        'timeStep': 0.1,
        'KparDiff': K1,
        'KparDist': K2,
        'sigmaError': 1.,
        'errorType': 'current',
        'outputDir': '../Output/surfaceTemplate',
        'testGradient':False,
        'lambdaPrior': 1.,
        'maxIter': 1000,
        'affine': 'none',
        'rotWeight': 10.,
        'sgd': 1,
        'transWeight': 1.,
        'scaleWeight': 10.,
        'affineWeight': 100.,
        'updateTemplate': True
    }
    f = SurfaceTemplate(Template=fv0, Target=fv, options=options)
    f.computeTemplate()

if __name__=="__main__":
    main()
