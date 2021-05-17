import matplotlib
matplotlib.use("QT5Agg")
import numpy as np
from base import loggingUtils
from base import surfaces
from base.kernelFunctions import Kernel
from base.surfaceMatching import SurfaceMatching as SM, SurfaceMatchingParam as SMP
from base.surfaceMatchingNormalExtremities import SurfaceMatching as SMN, SurfaceMatchingParam as SMPN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def BuildLaminar(target, outputDir, pancakeThickness=None, runRegistration=True):
    # Step 1: Create a labeled  "pancake" approximation to the target
    h, lab, width = target.createFlatApproximation(thickness=pancakeThickness, M=75)
    #h.smooth()
    fig = plt.figure(1)
    # fig.clf()
    ax = Axes3D(fig)
    lim0 = target.addToPlot(ax, ec='k', fc='b')
    h.addToPlot(ax, ec='k', fc='r')
    ax.set_xlim(lim0[0][0], lim0[0][1])
    ax.set_ylim(lim0[1][0], lim0[1][1])
    ax.set_zlim(lim0[2][0], lim0[2][1])
    fig.canvas.flush_events()

    # Step2: Register the target to the template
    if runRegistration:
        sigmaKernel = width
        sigmaDist = width
        sigmaError = .5
        internalWeight = 50.
        internalCost = 'h1'

        K1 = Kernel(name='laplacian', sigma=sigmaKernel)

        sm = SMP(timeStep=0.1, algorithm='cg', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                 errorType='varifold', internalCost=internalCost)
        f = SM(Template=target, Target=h, outputDir=outputDir, param=sm,
               testGradient=False,
               # subsampleTargetSize = 500,
               internalWeight=internalWeight, maxIter=100, affine='none', rotWeight=.01, transWeight=.01, scaleWeight=10.,
               affineWeight=100., saveFile='firstRun')

        f.optimizeMatching()
        fvDef = f.fvDef
    else:
        fvDef = surfaces.Surface(filename=outputDir+'/firstRun10.vtk')

    # Step 3: Label the target surface based on the deformed template
    lab2 = np.zeros(fvDef.vertices.shape[0], dtype=int)
    x = fvDef.vertices
    y0 = h.vertices[h.faces[:,0], :]
    y1 = h.vertices[h.faces[:,1], :]
    y2 = h.vertices[h.faces[:,2], :]
    E = ((y1-y0)**2).sum(axis=1)
    F = ((y1-y0)*(y2-y0)).sum(axis=1)
    G = ((y2-y0)**2).sum(axis=1)
    D = E*G - F*F
    r1 = ((x[:, np.newaxis, :] - y0[np.newaxis,:,:])* (y1[np.newaxis,:,:]-y0[np.newaxis,:,:])).sum(axis=2)
    r2 = ((x[:, np.newaxis, :] - y0[np.newaxis,:,:])* (y2[np.newaxis,:,:]-y0[np.newaxis,:,:])).sum(axis=2)
    a = (r1*G[np.newaxis,:] - r2*F[np.newaxis,:])/D[np.newaxis,:]
    b = (-r1*F[np.newaxis,:] + r2*E[np.newaxis,:])/D[np.newaxis,:]
    a = np.maximum(np.minimum(a,1),0)
    b = np.maximum(np.minimum(b,1),0)
    h0 = a[:, :, np.newaxis] * (y1[np.newaxis,:,:]-y0[np.newaxis,:,:]) \
        + b[:, :, np.newaxis]*(y2[np.newaxis,:,:]-y0[np.newaxis,:,:])
    res = ((x[:, np.newaxis,:] - y0[np.newaxis,:,:] - h0)**2).sum(axis=2)
    closest = np.argmin(res, axis=1)


    for k in range(fvDef.vertices.shape[0]):
        jmin = closest[k]
        d = ((fvDef.vertices[k,:] - h.vertices[h.faces[jmin,:],:])**2).sum(axis=1)
        imin = np.argmin(d)
        lab2[k] = lab[h.faces[jmin,imin]]

    #dist = ((h.vertices[:, np.newaxis, :] - f.fvDef.vertices[np.newaxis, :, :]) ** 2).sum(axis=2)
    #closest = np.argmin(dist, axis=0)
    #lab2 = lab[closest]

    # Step 4: Extract the upper and lower surfaces from the target and save data
    h.saveVTK(outputDir + '/pancakeTemplate.vtk', scalars=lab, scal_name='Labels')
    target.saveVTK(outputDir + '/labeledTarget.vtk', scalars=lab2, scal_name='Labels')
    fv1 = target.truncate(val=0.5 - np.fabs(1 - lab2))
    fv1.saveVTK(outputDir + '/labeledTarget1.vtk')
    fv2 = target.truncate(val=0.5 - np.fabs(2 - lab2))
    fv2.saveVTK(outputDir + '/labeledTarget2.vtk')

    # Step 5: Run surface matching with normality constraints from the lower surface of the target to the upper
    sigmaKernel = 2.5
    sigmaDist = 2.5
    sigmaError = .1
    internalWeight = 50.
    internalCost = 'h1'
    K1 = Kernel(name='laplacian', sigma=sigmaKernel)


    sm = SMPN(timeStep=0.1, algorithm='bfgs', KparDiff=K1, sigmaDist=sigmaDist, internalCost = internalCost,
              sigmaError=sigmaError, errorType='varifold')

    if target.surfVolume() > 0:
        fv1.flipFaces()
    else:
        fv2.flipFaces()

    f = SMN(Template=fv1, Target=fv2, outputDir=outputDir, param=sm, regWeight=1.,
            saveTrajectories=True, symmetric=False, pplot=True,
            affine='none', testGradient=True, internalWeight=internalWeight, affineWeight=1e3, maxIter_cg=100,
            maxIter_al=5, mu=1e-5)
    f.optimizeMatching()

    return f


if __name__ == "__main__":
    plt.ion()
    loggingUtils.setup_default_logging('', stdOutput = True)

    # Read target surface file.
    #hf = './TestData/ERC.vtk'
    #hf = '/USERS/younes/Development/Data/labeledTarget.vtk'
    hf = '/USERS/younes/Development/results/testERC2/template.vtk'

    fv = surfaces.Surface(filename = hf)
    #fv = examples.ellipsoid(a=0.5, b=0.5, c=0.25, d=100)

    BuildLaminar(fv, outputDir = '/Users/younes/Development/results/testERC2Streamlines',
                 pancakeThickness = None, runRegistration=False)

    plt.ioff()
    plt.show()

