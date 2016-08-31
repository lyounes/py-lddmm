import numpy as np
from surfaces import Surface
from kernelFunctions import Kernel
from surfaceMatching import SurfaceMatchingParam
from surfaceTemplate import SurfaceTemplate
import loggingUtils

def main():
    [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
    y = y-1
    z = z-1
    x = x-1
    s2 = np.sqrt(2)

    I1 = .06 - ((x-0.1)**2 + 0.5*y**2 + z**2)  
    fv1 = Surface() ;
    fv1.Isosurface(I1, value = 0, target=750, scales=[1, 1, 1])

    I1 = .06 - ((x+0.1)**2 + y**2 + 0.5*z**2) 
    fv2 = Surface() ;
    fv2.Isosurface(I1, value=0, target=750, scales=[1, 1, 1])

    u = (z + y)/s2
    v = (z - y)/s2
    I1 = .095 - (x**2 + v**2 + 0.5*u**2) 
    fv3 = Surface() ;
    fv3.Isosurface(I1, value = 0, target=750, scales=[1, 1, 1])

    u = (z + y)/s2
    v = (z - y)/s2
    I1 = .095 - (x**2 + 0.5*v**2 + u**2) 
    fv4 = Surface() ;
    fv4.Isosurface(I1, value=0, target=750, scales=[1, 1, 1])

    loggingUtils.setup_default_logging('/Users/younes/Results/surfaceTemplate2', fileName='info.txt', stdOutput = True)
    K1 = Kernel(name='laplacian', sigma = 50.0)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=20., sigmaError=10., errorType='current')
    f = SurfaceTemplate(HyperTmpl=fv4, Targets=[fv1, fv2, fv3, fv4], outputDir='/Users/younes/Results/surfaceTemplate2',param=sm, testGradient=False, 
                        lambdaPrior = .1, maxIter=1000, affine='euclidean', rotWeight=10.,
                        transWeight = 1., scaleWeight=10., affineWeight=100.)
    f.computeTemplate()

if __name__=="__main__":
    main()
