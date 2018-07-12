from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Diffeons.gaussianDiffeonsSurfaceMatching import *

def compute(createsurfaces=True):

    if createsurfaces:
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        y = y-1
        z = z-1
        s2 = np.sqrt(2)

        I1 = .06 - ((x-.50)**2 + 0.5*y**2 + z**2)  
        fv1 = Surface()
        fv1.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        #return fv1
        
        u = (z + y)/s2
        v = (z - y)/s2
        I1 = np.maximum(0.05 - (x-.6)**2 - 0.5*y**2 - z**2, 0.03 - (x-.50)**2 - 0.5*y**2 - z**2)  
        #I1 = .06 - ((x-.50)**2 + 0.75*y**2 + z**2)  
        #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 
        fv2 = Surface()
        fv2.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        fv1.saveVTK('/Users/younes/Development/Results/Diffeons/fv1alt.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Diffeons/fv2alt.vtk')
    else:
        if False:
            #path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
            path = '/Volumes/CIS/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
            #sub1 = '0186193_1_6'
            #sub2 = '1449400_1_L'
            sub2 = 'LU027_R_sumNCBC20100628'
            fv1 = surfaces.Surface(filename =path + '5_population_template_qc/newTemplate.byu')
            v1 = fv1.surfVolume()
            #f0.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
            fv2 = surfaces.Surface(filename =path + '2_qc_flipped_registered/' + sub2 + '_registered.byu')
            v2 = fv2.surfVolume()
            if (v2*v1 < 0):
                fv2.flipFaces()
        else:
            #f1.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub2+'_amyg_L.byu'))
            fv1 = Surface(filename='/Users/younes/Development/Results/Diffeons/fv1Alt.vtk')
            fv2  = Surface(filename='/Users/younes/Development/Results/Diffeons/fv2Alt.vtk')

        #return fv1, fv2

    ## Object kernel
    r0 = 50./fv1.vertices.shape[0]
    T0 = 100
    withDiffeons=True

    sm = SurfaceMatchingParam(timeStep=0.1, sigmaKernel=10., sigmaDist=5., sigmaError=1.,
                              errorType='diffeonCurrent')
        #errorType='current')

    if withDiffeons:
        gdOpt = gd.gdOptimizer(surf=fv1, sigmaDist = .5, DiffeonEpsForNet = r0, testGradient=False, maxIter=100)
        gdOpt.optimize()
        f = SurfaceMatching(Template=fv1, Target=fv2, outputDir='/Users/younes/Development/Results/Diffeons/BallsAlt50_500_d',param=sm, testGradient=False,
        Diffeons = (gdOpt.c0, gdOpt.S0, gdOpt.idx),
        subsampleTargetSize = 500,
        #DecimationTarget=100,
                            #DiffeonEpsForNet = r0,
                            #DiffeonSegmentationRatio=r0,
                            maxIter=10000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
    else:
        f = SurfaceMatching(Template=fv1, Target=fv2, outputDir='/Users/younes/Development/Results/Diffeons/Scale100_250_0',param=sm, testGradient=False,
                            subsampleTargetSize = 250,
                            zeroVar=True,
                            #DecimationTarget=T0,
                            #DiffeonEpsForNet = r0,
                            #DiffeonSegmentationRatio=r0,
                            maxIter=10000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    return f
    #f.maxIter = 200
    f.param.sigmaError=1.0
    f.setOutputDir('/Users/younes/Development/Results/Diffeons2/Scale2')
    f.restart(DecimationTarget = 2*T0)
    # f.restart(DiffeonEpsForNet = 2*r0)
    #f.restart(DiffeonSegmentationRatio = 0.025)
    #f.maxIter = 300
    f.param.sigmaError=1.
    f.setOutputDir('/Users/younes/Development/Results/Diffeons2/Scale3')
    f.restart(DecimationTarget = 3*T0)
    #f.restart(DiffeonEpsForNet = 4*r0)
    #f.restart(DiffeonSegmentationRatio = 0.05)

    return f

if __name__=="__main__":
    compute()
