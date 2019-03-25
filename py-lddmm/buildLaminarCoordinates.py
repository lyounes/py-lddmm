import matplotlib
matplotlib.use("QT5Agg")
import numpy as np
from base import loggingUtils
from base import surfaces
from base.kernelFunctions import Kernel
from affineRegistration import rigidRegistration
from surfaceMatching import SurfaceMatching as SM, SurfaceMatchingParam as SMP
from surfaceMatchingNormalExtremities import SurfaceMatching as SMN, SurfaceMatchingParam as SMPN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def BuildLaminar(target, outputDir):
    # Create a labeled  "pancake" approximation to the target
    h, lab = target.createFlatApproximation()

    fig = plt.figure(1)
    # fig.clf()
    ax = Axes3D(fig)
    lim0 = fv.addToPlot(ax, ec='k', fc='b')
    h.addToPlot(ax, ec='k', fc='r')
    ax.set_xlim(lim0[0][0], lim0[0][1])
    ax.set_ylim(lim0[1][0], lim0[1][1])
    ax.set_zlim(lim0[2][0], lim0[2][1])
    fig.canvas.flush_events()

    # Register the pancake to the target
    sigmaKernel = 0.5
    sigmaDist = 1.
    sigmaError = 1.
    internalWeight = 10.
    internalCost = 'h1'

    K1 = Kernel(name='laplacian', sigma=sigmaKernel)

    sm = SMP(timeStep=0.1, algorithm='bfgs', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
             errorType='varifold', internalCost=internalCost)
    f = SM(Template=h, Target=target, outputDir=outputDir, param=sm,
           testGradient=False,
           # subsampleTargetSize = 500,
           internalWeight=internalWeight, maxIter=100, affine='none', rotWeight=.01, transWeight=.01, scaleWeight=10.,
           affineWeight=100.)

    f.optimizeMatching()

    # Label the target surface based on the deformed template
    dist = ((target.vertices[:, np.newaxis, :] - f.fvDef.vertices[np.newaxis, :, :]) ** 2).sum(axis=2)
    closest = np.argmin(dist, axis=1)
    lab2 = lab[closest]

    # Extract the upper and lower surfaces from the target and save data
    h.saveVTK(outputDir + '/pancakeTemplate.vtk', scalars=lab, scal_name='Labels')
    target.saveVTK(outputDir + '/labeledTarget.vtk', scalars=lab2, scal_name='Labels')
    fv1 = target.truncate(val=0.5 - np.fabs(1 - lab2))
    fv1.saveVTK(outputDir + '/labeledTarget1.vtk')
    fv2 = target.truncate(val=0.5 - np.fabs(2 - lab2))
    fv2.saveVTK(outputDir + '/labeledTarget2.vtk')

    # Run surface matching with normality constraints from the lower surface of the target to the upper

    sm = SMPN(timeStep=0.1, algorithm='bfgs', KparDiff=K1, sigmaDist=1.,
              sigmaError=.1, errorType='varifold')

    fv2.flipFaces()

    f = SMN(Template=fv1, Target=fv2, outputDir=outputDir, param=sm, regWeight=1.,
            saveTrajectories=True, symmetric=False, pplot=True,
            affine='none', testGradient=False, internalWeight=100., affineWeight=1e3, maxIter_cg=1000,
            maxIter_al=5, mu=1e-4)
    f.optimizeMatching()

    return f


if __name__ == "__main__":
    plt.ion()
    loggingUtils.setup_default_logging('', stdOutput = True)

    # Read target surface file.
    hf = './TestData/ERC.vtk'
    fv = surfaces.Surface(filename = hf)

    BuildLaminar(fv, '/Users/younes/Development/Results/pancake_hippo')

    plt.ioff()
    plt.show()

