import numpy as np
import logging
import loggingUtils
import surfaces
from surfaces import *
from kernelFunctions import *
from surfaceMatching import *

def compute():

    outputDir = '/Users/younes/Development/Results/biocardSingle'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()
    #path = '/Volumes/project/biocard/data/phase_1_surface_mapping_new_structure/'
    path = '/Users/younes/Development/Data/multishape/biocard/'
    #path = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
    #path2 = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/amygdala/6_mappings_baseline_template_all/0_template_to_all/' ;
    #path = '/cis/home/younes/MorphingData/Biocard/'
    sub1 = '0186193_1_6'
    #sub2 = '1449400_1_L'
    sub2 = '1229175_2_4'
    f0 = []
    f0.append(surfaces.Surface(filename = path+'Atlas_hippo_L_separate.byu'))
    f0.append(surfaces.Surface(filename = path+'Atlas_amyg_L_separate.byu'))
    f0.append(surfaces.Surface(filename = path+'Atlas_ent_L_up_separate.byu'))
    fv0 = Surface()
    fv0.concatenate(f0)
    f1 = []
    f1.append(surfaces.Surface(filename = path+'danielData/0186193_1_6_hippo_L_qc_pass1_daniel2_reg.vtk'))
    f1.append(surfaces.Surface(filename = path+'danielData/0186193_1_6_amyg_L_qc_pass1_daniel2_reg.vtk'))
    f1.append(surfaces.Surface(filename = path+'danielData/0186193_1_6_ec_L_qc_pass1_daniel2_reg.vtk'))
    fv1 = Surface()
    fv1.concatenate(f1)


    K1 = Kernel(name='laplacian', sigma = 5.)
    
    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1, errorType='measure')
    f = (SurfaceMatching(Template=fv0, Target=fv1, outputDir=outputDir,param=sm, testGradient=False,
                         maxIter=2000, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()
