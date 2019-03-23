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



if __name__ == "__main__":
    plt.ion()
    loggingUtils.setup_default_logging('', stdOutput = True)


    hf = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template/newTemplate.byu'
    #hf = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_06052013/erc/4_create_population_based_template/newTemplate.byu'
    fv = surfaces.Surface(filename = hf)
    h, lab = fv.createFlatApproximation()

    fig = plt.figure(1)
    # fig.clf()
    ax = Axes3D(fig)
    lim0 = fv.addToPlot(ax, ec='k', fc='b')
    # ax.set_xlim(lim0[0][0], lim0[0][1])
    # ax.set_ylim(lim0[1][0], lim0[1][1])
    # ax.set_zlim(lim0[2][0], lim0[2][1])
    # # fig = plt.figure(2)
    # ax = Axes3D(fig)
    h.addToPlot(ax, ec='k', fc='r')
    ax.set_xlim(lim0[0][0], lim0[0][1])
    ax.set_ylim(lim0[1][0], lim0[1][1])
    ax.set_zlim(lim0[2][0], lim0[2][1])
    fig.canvas.flush_events()
    # ax.auto_scale()

    sigmaKernel = 0.5
    sigmaDist = 1.
    sigmaError = 1.
    internalWeight = 10.
    internalCost = 'h1'

    K1 = Kernel(name='laplacian', sigma = sigmaKernel)

    outputDir = '/Users/younes/Development/Results/pancake_hippo'
    sm = SMP(timeStep=0.1, algorithm='bfgs', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                              errorType='varifold', internalCost=internalCost)
    f = SM(Template=h, Target=fv, outputDir=outputDir,param=sm,
                        testGradient=False,
                        #subsampleTargetSize = 500,
                         internalWeight=internalWeight, maxIter=100, affine= 'none', rotWeight=.01, transWeight = .01, scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    dist = ((fv.vertices[:, np.newaxis, :] - f.fvDef.vertices[np.newaxis,:,:])**2).sum(axis=2)
    closest = np.argmin(dist, axis=1)
    lab2 = lab[closest]

    h.saveVTK(outputDir+'/pancakeTemplate.vtk', scalars=lab, scal_name='Labels')
    fv.saveVTK(outputDir+'/labeledTarget.vtk', scalars=lab2, scal_name='Labels')
    fv1 = fv.truncate(val = 0.5 - np.fabs(1 - lab2))
    fv1.saveVTK(outputDir+'/labeledTarget1.vtk')
    fv2 = fv.truncate(val = 0.5 - np.fabs(2 - lab2))
    fv2.saveVTK(outputDir+'/labeledTarget2.vtk')

    # u = np.arange(0, fv.vertices.shape[0], dtype=int)
    # u1 = np.cumsum(lab2==1)-1
    # u2 = np.cumsum(lab2==2)-1
    # vert1 = fv.vertices[lab2==1,:]
    # faces1 = np.zeros(fv.faces.shape)
    # nf = 0
    # for k in range(fv.faces.shape[0]):
    #     if lab2[fv.faces[k,0]] == 1 and lab2[fv.faces[k,1]] == 1 and lab2[fv.faces[k,2]] == 1:
    #         faces1[nf,0] = u1[fv.faces[k,0]]
    #         faces1[nf,1] = u1[fv.faces[k,1]]
    #         faces1[nf,2] = u1[fv.faces[k,2]]
    #         nf += 1
    # faces1 = faces1[0:nf,:]
    # fv1 = surfaces.Surface(FV=(faces1,vert1))
    # fv1.removeIsolated()
    # fv1.edgeRecover()
    # fv1.saveVTK('/Users/younes/Development/Results/pancake/labeledTarget1.vtk')
    #
    # vert1 = fv.vertices[lab2==2,:]
    # faces1 = np.zeros(fv.faces.shape)
    # nf = 0
    # for k in range(fv.faces.shape[0]):
    #     if lab2[fv.faces[k,0]] == 2 and lab2[fv.faces[k,1]] == 2 and lab2[fv.faces[k,2]] == 2:
    #         faces1[nf,0] = u2[fv.faces[k,0]]
    #         faces1[nf,1] = u2[fv.faces[k,1]]
    #         faces1[nf,2] = u2[fv.faces[k,2]]
    #         nf += 1
    # faces1 = faces1[0:nf,:]
    # fv2 = surfaces.Surface(FV=(faces1,vert1))
    # fv2.removeIsolated()
    # fv2.edgeRecover()
    # fv2.saveVTK('/Users/younes/Development/Results/pancake/labeledTarget2.vtk')



    sm = SMPN(timeStep=0.1, algorithm='bfgs', KparDiff=K1, sigmaDist=1.,
                                              sigmaError=.1, errorType='varifold')

    fv2.flipFaces()

    f = SMN(Template=fv1, Target=fv2, outputDir=outputDir, param=sm, regWeight=1.,
                        saveTrajectories=True, symmetric=False, pplot=True,
                        affine='none', testGradient=True, internalWeight=100., affineWeight=1e3, maxIter_cg=1000,
                        maxIter_al=5, mu=1e-4)
    f.optimizeMatching()

    plt.ioff()
    plt.show()

