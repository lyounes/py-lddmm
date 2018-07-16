import numpy as np
from common import surfaces
from common import loggingUtils
from common.surfaces import Surface
from common.kernelFunctions import Kernel
from surfaceMatching import SurfaceMatching, SurfaceMatchingParam
import matplotlib.pyplot as plt
plt.ion()
def compute(model):

    loggingUtils.setup_default_logging('', stdOutput = True)
    if model=='Balls':
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        y = y-1
        z = z-1
        s2 = np.sqrt(2)

        I1 = .06 - ((x-.50)**2 + 0.5*y**4 + z**2)  
        fv1 = Surface() ;
        fv1.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        #return fv1
        
        u = (z + y)/s2
        v = (z - y)/s2
        I1 = np.maximum(0.05 - (x-.7)**2 - 0.5*y**2 - z**2, 0.02 - (x-.50)**2 - 0.5*y**2 - z**2)  
        #I1 = .05 - np.minimum((x-.7)**2 + 0.5*y**2 + z**2, (x-.30)**2 + 0.5*y**2 + z**2)  
        #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 
        fv2 = Surface() ;
        fv2.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        fv1.saveVTK('/cis/home/younes/MorphingData/fv1.vtk')
        fv2.saveVTK('/cis/home/younes/MorphingData/fv2.vtk')
        ftemp = fv1 
        ftarg = fv2
    elif model=='Hearts':
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        ay = np.fabs(y-1)
        az = np.fabs(z-1)
        ax = np.fabs(x-0.5)
        s2 = np.sqrt(2)
        c1 = np.sqrt(0.06)
        c2 = np.sqrt(0.045)
        c3 = 0.1

        I1 = np.minimum(c1**2 - (ax**2 + 0.5*ay**2 + az**2), np.minimum((ax**2 + 0.5*ay**2 + az**2)-c2**2, 1+c3-y)) 
        fv1 = Surface() ;
        fv1.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)


        #return fv1

        #s1 = 1.375
        #s2 = 2
        s1 = 1.1
        s2 = 1.2
        p = 1.75
        I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        fv2 = Surface() ;
        fv2.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv2.vertices[:,1] += 15 - 15/s1

        s1 *= 1.1
        s2 *= 1.2
        I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        fv3 = Surface() ;
        fv3.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv3.vertices[:,1] += 15 - 15/s1

        
        fv1.saveVTK('/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Fibers/fv2.vtk')
        fv3.saveVTK('/Users/younes/Development/Results/Fibers/fv3.vtk')
        ftemp = fv1
        ftarg = fv3
    elif model=='Hippo1':
        path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
        #sub1 = '0186193_1_6'
        #sub2 = '1449400_1_L'
        sub2 = 'LU027_R_sumNCBC20100628'
        fv1 = surfaces.Surface(filename =path + '5_population_template_qc/newTemplate.byu')
        v1 = fv1.surfVolume()
        #f0.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
        fv2 = surfaces.Surface(filename =path + '2_qc_flipped_registered/' + sub2 + '_registered.byu')
        v2 = fv2.surfVolume()
        if (v2*v1 < 0):
            fv2.faces = fv2.faces[:, [0,2,1]]
        ftemp = fv1 
        ftarg = fv2
    elif model=='Hippo2':
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
            fv2.faces = fv2.faces[:, [0,2,1]]
        ftemp = fv1 
        ftarg = fv2
    elif model=='biocardMulti':
        path = '/cis/home/younes/MorphingData/biocard/'
        #path = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
        #path2 = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/amygdala/6_mappings_baseline_template_all/0_template_to_all/' ;
        #path = '/cis/home/younes/MorphingData/Biocard/'
        sub1 = '0186193_1_6'
        #sub2 = '1449400_1_L'
        sub2 = '1229175_2_4'
        ftemp = []
        ftemp.append(surfaces.Surface(filename =path + 'Atlas_hippo_L_separate.byu'))
        ftemp.append(surfaces.Surface(filename =path + 'Atlas_amyg_L_separate.byu'))
        ftemp.append(surfaces.Surface(filename =path + 'Atlas_ent_L_up_separate.byu'))
        ftarg = []
        ftarg.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_hippo_L_qc_pass1_daniel2_reg.vtk'))
        ftarg.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_amyg_L_qc_pass1_daniel2_reg.vtk'))
        ftarg.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_ec_L_qc_pass1_daniel2_reg.vtk'))
    else:
        return

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = 6.5)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='varifold', internalCost=None)#'h1')
    f = SurfaceMatching(Template=ftemp, Target=ftarg, outputDir='/Users/younes/Development/Results/'+model+'LDDMM6p5',param=sm, testGradient=False,
                        #subsampleTargetSize = 500,
                         internalWeight=10, maxIter=1000, affine= 'none', rotWeight=.01, transWeight = .01, scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


if __name__=="__main__":
    compute('biocardMulti')
