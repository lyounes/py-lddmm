import numpy as np
import curves
from curves import *
import loggingUtils
from kernelFunctions import *
from curveMatching import *

def compute(model='default', dirOut='/cis/home/younes'):

    if model == 'smile':
        s = 0.025
        t = np.arange(0, 10+1e-3, s)        
        x = 0.1*((t-5)**2 -25)
        f = np.zeros([t.shape[0]-1,2], dtype=int)
        f[:,0] = range(0, t.shape[0]-1)
        f[:,1] = range(1, t.shape[0])
        v = np.zeros([t.shape[0],2])
        v[:,0] = t
        v[:,1] = -x
        fv1 = Curve(FV=(f,v))
        v[:,1] = x
        fv11 = Curve(FV=(f,v))
        ftemp = (fv1, fv11)
        #x = 0.001*((t-5)**2 -25)
        #v[:,1] = x
        fv2 = Curve(FV=(f,v))
        v[:,1] =  0.09*((t-5)**2 -25)
        fv22 = Curve(FV=(f,v))
        ftarg = (fv2, fv22)
    elif model == 'manyCurves':   
        s = 0.1
        nc = 25
        ftemp = []
        rad = np.arange(7, 10, 3./nc)
        for k,r in enumerate(rad):
#            start = np.random.uniform(0, 2*np.pi)
#            end = start + np.random.uniform(0, np.pi)
            start = k*np.pi/nc
            end = start + np.pi
            #rad = np.random.uniform(7,10)
            t = np.arange(start, end+s, s)
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],2])
            v[:,0] = r*np.cos(t)
            v[:,1] = r*np.sin(t)
            ftemp.append(Curve(FV=(f,v)))
        ftarg = []
        rad = np.arange(2, 5, 3./nc)
        for k,r in enumerate(rad):
            #start = np.random.uniform(0, 2*np.pi)
            #end = start + np.random.uniform(0, np.pi)
            start = np.pi/6 + k*np.pi/nc
            end = start + np.pi
            #rad = np.random.uniform(2,5)
            t = np.arange(start, end+s, s)
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],2])
            v[:,0] = r*np.cos(t)
            v[:,1] = r*np.sin(t)
            ftarg.append(Curve(FV=(f,v)))
    elif model == 'rays':
        nrays = 10
        t = np.arange(0, 5, 0.1)
        ftemp = []
        x0 = 5
        y0 = 5
        theta = 2*np.pi*np.arange(0, nrays)/nrays
        for k in range(nrays):
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],2])
            v[:,0] = x0 + np.cos(theta[k])*t
            v[:,1] = y0 + np.sin(theta[k])*t
            ftemp.append(Curve(FV=(f,v)))
            
        ftarg = []
        x0 = 8
        y0 = 5
        theta = 2*np.pi*(np.arange(0,nrays, dtype=float)/nrays)**(0.5)
        for k in range(nrays):
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],2])
            v[:,0] = x0 + np.cos(theta[k])*t
            v[:,1] = y0 + np.sin(theta[k])*t
            ftarg.append(Curve(FV=(f,v)))
            
            
    else:   
        [x,y] = np.mgrid[0:200, 0:200]/100.
        y = y-1
        s2 = np.sqrt(2)
    
        I1 = .06 - ((x-.30)**2 + 0.5*y**2)  
        I2 = .01 - ((x-.16)**2 + y**2)  
        fv1 = Curve() ;
        fv1.Isocontour(I1, value = 0, target=200, scales=[1, 1])
        fv11 = Curve() ;
        fv11.Isocontour(I2, value = 0, target=200, scales=[1, 1])
        ftemp = (fv1, fv11)
    
        u = (x-.5 + y)/s2
        v = (x -.5 - y)/s2
        I1 = .095 - (u**2 + 0.5*v**2) 
        I2 = .01 - ((u-.20)**2 + 0.8*v**2) 
        fv2 = Curve() ;
        fv2.Isocontour(I1, value = 0, target=750, scales=[1, 1])
        fv22 = Curve() ;
        fv22.Isocontour(I2, value = 0, target=750, scales=[1, 1])
        ftarg = (fv2, fv22)

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = .5)

    loggingUtils.setup_default_logging(dirOut+'/Development/Results/curveMatching', fileName='info.txt', 
                                       stdOutput = True)    
    
    sm = CurveMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=1, sigmaError=.05, errorType='varifold', internalCost='h1Invariant')
    f = CurveMatching(Template=ftemp, Target=ftarg, outputDir=dirOut+'/Development/Results/curveMatching'+model+'2',param=sm, testGradient=False,
                      regWeight=.1, internalWeight=100.0, maxIter=10000, affine='none', rotWeight=10., transWeight = 10., scaleWeight=100., affineWeight=100.)
                      
 
    f.optimizeMatching()


    return f
    
if __name__=="__main__":
    compute(model='rays', dirOut='/Users/younes')

