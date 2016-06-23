import numpy as np
import curves
from curves import *
import loggingUtils
from kernelFunctions import *
from curveMatching import *

def compute():

    [x,y] = np.mgrid[0:200, 0:200]/100.
    y = y-1
    s2 = np.sqrt(2)

    I1 = .06 - ((x-.30)**2 + 0.5*y**2)  
    fv1 = Curve() ;
    fv1.Isocontour(I1, value = 0, target=100, scales=[1, 1])
    fv1.saveVTK('/cis/home/younes/MorphingData/curve1.vtk')
    J = np.nonzero((fv1.linel**2).sum(axis=1)<1e-10)
    for i in J[0]:
        print i
        print fv1.vertices[fv1.faces[i,0],:], " ", fv1.vertices[fv1.faces[i,1],:], " ", fv1.linel[i]

    u = (x-.7 + y)/s2
    v = (x -.7 - y)/s2
    I1 = .095 - (u**2 + 0.5*v**2) 
    fv2 = Curve() ;
    fv2.Isocontour(I1, value = 0, target=750, scales=[1, 1])
    fv2.saveVTK('/cis/home/younes/MorphingData/curve2.vtk')
    J = np.nonzero((fv2.linel**2).sum(axis=1)<1e-10)
    for i in J[0]:
        print i
        print fv2.vertices[fv2.faces[i,0],:], " ", fv2.vertices[fv2.faces[i,1],:], " ", fv2.linel[i]

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = 100.0)

    loggingUtils.setup_default_logging('/cis/home/younes/Development/Results/curveMatching', fileName='info.txt', 
                                       stdOutput = True)
    print "kernel"
    sm = CurveMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=20, sigmaError=.1, errorType='measure')
    f = CurveMatching(Template=fv1, Target=fv2, outputDir='/cis/home/younes/Development/Results/curveMatching',param=sm, testGradient=True,
                      internalWeight=1.0, maxIter=1000, affine='none', rotWeight=10., transWeight = 10., scaleWeight=100., affineWeight=100.)

    f.optimizeMatching()


    return f
    
if __name__=="__main__":
    compute()

