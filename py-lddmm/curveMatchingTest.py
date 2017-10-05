import numpy as np
#import curves
from curves import Curve, remesh
import loggingUtils
from kernelFunctions import Kernel
from curveMatching import CurveMatchingParam, CurveMatching
import matplotlib.pyplot as plt


def compute(model='default', dirOut='/cis/home/younes'):

    sigma = 0.5
    sigmaDist = 0.5 
    sigmaError = 0.1
    internalWeight = 1000        
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
        ftemp = [fv1, fv11]
        #x = 0.001*((t-5)**2 -25)
        #v[:,1] = x
        fv2 = Curve(FV=(f,v))
        v[:,1] =  0.09*((t-5)**2 -25)
        fv22 = Curve(FV=(f,v))
        ftarg = [fv2, fv22]
        sigma = 0.2
        sigmaDist = 0.5 
        sigmaError = 0.1
        internalWeight = 1000        
    elif model == 'manyCurves':   
        s = 0.5
        nc = 10
        ftemp = []
        rad = np.arange(4, 7, 3./nc)
        for k,r in enumerate(rad):
#            start = np.random.uniform(0, 2*np.pi)
#            end = start + np.random.uniform(0, np.pi)
            start = k*np.pi/nc
            end = start + np.pi
            #rad = np.random.uniform(7,10)
            t = np.arange(start, end+s/r, s/r)
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],2])
            v[:,0] = r*np.cos(t)
            v[:,1] = r*np.sin(t)
            ftemp.append(Curve(FV=(f,v)))
        s = 0.25
        ftarg = []
        rad = np.sqrt(np.arange(2.**2, (6.)**2, ((6.)**2-(2.)**2)/nc))
        for k,r in enumerate(rad):
            #start = np.random.uniform(0, 2*np.pi)
            #end = start + np.random.uniform(0, np.pi)
            start = np.pi/6 + k*np.pi/nc
            end = start + np.pi
            #rad = np.random.uniform(2,5)
            t = np.arange(start, end+s/r, s/r)
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],2])
            v[:,0] = r*np.cos(t)
            v[:,1] = r*np.sin(t)
            ftarg.append(Curve(FV=(f,v)))
        sigma = 1.0
        sigmaDist = 2.5 
        sigmaError = 0.1
        internalWeight = 100        
    elif model == 'rays':
        nrays = 10
        t = np.arange(0.25, 5, 0.25)
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
            
        f = np.zeros([nrays,2], dtype=int)
        f[0:nrays-1,0] = range(0, nrays-1)
        f[0:nrays-1,1] = range(1, nrays)
        f[nrays-1,0] = nrays-1
        f[nrays-1,1] = 0
        v = np.zeros([nrays,2])
        for k in range(nrays):
            v[k,0] = x0 + np.cos(theta[k])*t[0]
            v[k,1] = y0 + np.sin(theta[k])*t[0]
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
        sigma = 0.2
        sigmaDist = 2.5 
        sigmaError = 0.1
        internalWeight = 250        
    elif model == '3Drays':
        nrays = 10
        t = np.arange(2, 5, 0.25)
        ftemp = []
        x0 = 5
        y0 = 5
        z0 = 5
        theta = 2*np.pi*np.arange(0, nrays)/nrays
        for k in range(nrays):
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],3])
            v[:,0] = x0 + np.cos(theta[k])*t
            v[:,1] = y0 + np.sin(theta[k])*t
            v[:,2] = z0 + 0.5*theta[k]*t
            ftemp.append(Curve(FV=(f,v)))
            
        ftarg = []
        x0 = 8
        y0 = 5
        z0 = 5
        #theta = 2*np.pi*(np.arange(0,nrays, dtype=float)/nrays)**(0.5)
        theta = 2*np.pi*(np.arange(0,nrays, dtype=float)/nrays + 0.25)
        for k in range(nrays):
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],3])
            v[:,0] = x0 + np.cos(theta[k])*t
            v[:,1] = y0 + np.sin(theta[k])*t
            v[:,2] = z0 + 0.4*theta[k]*t
            ftarg.append(Curve(FV=(f,v)))      
        sigma = 2.5
        sigmaDist = 2.5 
        sigmaError = 0.1
        internalWeight = 250        

    elif model == 'helixes':
        nrays = 5
        t = np.arange(0., 100, 2.)
        st = t/(1+t**(0.5))
        ftemp = []
        x0 = 5
        y0 = 5
        z0 = 5
        r = 0.05
        a = 0.5
        h = .5
        theta = 2*np.pi*(np.arange(0, nrays, dtype=float)/nrays)**h
        for k in range(nrays):
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],3])
            v[:,0] = x0 + r*(t)*np.cos(st+theta[k])
            v[:,1] = y0 + r*(t)*np.sin(st+theta[k])
            v[:,2] = z0 + a*st
            ftemp.append(Curve(FV=(f,v)))
            
        ftarg = []
        x0 = 5
        y0 = 5
        z0 = 6
        r = 0.05
        a = 0.5
        h = .25
        theta = 2*np.pi*(np.arange(0,nrays, dtype=float)/nrays)**h +np.pi/6
        #theta = 2*np.pi*(np.arange(0,nrays, dtype=float)/nrays + 0.25)
        for k in range(nrays):
            f = np.zeros([t.shape[0]-1,2], dtype=int)
            f[:,0] = range(0, t.shape[0]-1)
            f[:,1] = range(1, t.shape[0])
            v = np.zeros([t.shape[0],3])
            v[:,0] = x0 + r*(t)*np.cos(st+theta[k])
            v[:,1] = y0 + r*(t)*np.sin(st+theta[k])
            v[:,2] = z0 + a*st
            ftarg.append(Curve(FV=(f,v)))
        sigma = 0.1
        sigmaDist = 0.5 
        sigmaError = 0.1
        internalWeight = 500        
    elif model == 'atlasSections':
        fv01 = Curve(filename='/cis/home/younes/MorphingData/MostofskyCurves/atlas04_curve01.txt')
        fv02 = Curve(filename='/cis/home/younes/MorphingData/MostofskyCurves/atlas04_curve02.txt')
        fv03 = Curve(filename='/cis/home/younes/MorphingData/MostofskyCurves/atlas04_curve03.txt')
        fv11 = Curve(filename='/cis/home/younes/MorphingData/MostofskyCurves/atlas09_curve01.txt')
        fv12 = Curve(filename='/cis/home/younes/MorphingData/MostofskyCurves/atlas09_curve02.txt')
        fv13 = Curve(filename='/cis/home/younes/MorphingData/MostofskyCurves/atlas09_curve03.txt')
        ftemp = [fv01, fv02, fv03]
        ftarg = [fv11, fv12, fv13]
    elif model == 'ellipses':
        N = 100
        [x,y] = np.mgrid[0:2*N, 0:2*N]/float(N)
        y = y-1
        s2 = np.sqrt(2)
        scl = [10./N, 10./N]
    
        I1 = .06 - ((x-.30)**2 + 0.5*y**2)  
        I2 = .008 - ((x-.16)**2 + y**2)  
        I3 = .006 - ((x-.4)**2 + 2*y**2)  
        fv1 = Curve()
        fv1.Isocontour(I1, value = 0, target=100, scales=scl)
        fv11 = Curve()
        fv11.Isocontour(I2, value = 0, target=100, scales=scl)
        fv12 = Curve()
        fv12.Isocontour(I3, value = 0, target=100, scales=scl)
        ftemp = [fv1, fv11, fv12]
    
        u = (x-.5 + y)/s2
        v = (x -.5 - y)/s2
        I1 = .095 - (u**2 + 0.5*v**2) 
        u = (x-.35 + y)/s2
        v = (x -.35 - y)/s2
        I2 = .01 - ((u-.12)**2 + 0.5*v**2) 
        I3 = .01 - (2*(u+.10)**2 + 0.8*v**2) 
        fv2 = Curve()
        fv2.Isocontour(I1, value = 0, target=100, scales=scl)
        fv22 = Curve()
        fv22.Isocontour(I2, value = 0, target=100, scales=scl)
        fv23 = Curve()
        fv23.Isocontour(I3, value = 0, target=100, scales=scl)
        ftarg = [fv2, fv22, fv23]
        sigma = 0.2
        sigmaDist = 2.0 
        sigmaError = 0.1
        internalWeight = 500
    elif model=="cardioid":
        t = np.arange(0., 2*np.pi, 0.05)
        a = 1.5
        b = 2
        c = 0.7
        x0 = 10
        y0 = 2
        x = x0+a*np.cos(t) * (1-c*np.cos(t))
        y = y0+b*np.sin(t) * (1-c*np.cos(t))
        v = np.zeros([t.shape[0],2])
        v[:,0] = x
        v[:,1] = y
        v = remesh(v, N=100)
        N = v.shape[0]
        f = np.zeros([N,2], dtype=int)
        f[0:N-1,0] = range(0, N-1)
        f[0:N-1,1] = range(1, N)
        f[N-1,:] = [N-1,0]
        ftemp = Curve(FV=(f,v))

        a = 5
        b = 3
        c = 0.6
        #x0 = 9
        #y0 = 3
        x = x0+a*np.cos(t) * (1-c*np.cos(t))
        y = y0+b*np.sin(t) * (1-c*np.cos(t))
        v = np.zeros([t.shape[0],2])
        v[:,0] = x
        v[:,1] = y
        f = np.zeros([v.shape[0]-1,2], dtype=int)
        f[:,0] = range(0, v.shape[0]-1)
        f[:,1] = range(1, v.shape[0])
        ftarg = Curve(FV=(f,v))
        sigma = 0.5
        sigmaDist = 2.0 
        sigmaError = 0.1
        internalWeight = 250
    else:
        return

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = sigma)

    loggingUtils.setup_default_logging(dirOut+'/Development/Results/curveMatching', fileName='info.txt', 
                                       stdOutput = True)   
    
    sm = CurveMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError, errorType='varifold', internalCost='h1Invariant')
    f = CurveMatching(Template=ftemp, Target=ftarg, outputDir=dirOut+'/Development/Results/curveMatching'+model+'HLDDMM_0p2_nocomp25',param=sm, testGradient=False,
                      regWeight=1., internalWeight=internalWeight, maxIter=10000, affine='none', rotWeight=1, transWeight = 1, scaleWeight=100., affineWeight=100.)
 
    f.optimizeMatching()


    return f
    
if __name__=="__main__":
    plt.ion()
    compute(model='rays', dirOut='/Users/younes')
    plt.ioff()
    plt.show()

