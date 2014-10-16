import numpy as np
import logging
import loggingUtils
import surfaces
from surfaces import *
from kernelFunctions import *
from surfaceMultiPhase import *

def compute():

    outputDir = '/Users/younes/Development/Results/biocardSlidingOneshape'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()
    #path = '/Volumes/project/biocard/data/phase_1_surface_mapping_new_structure/'
    path = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
    path2 = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/amygdala/6_mappings_baseline_template_all/0_template_to_all/' ;
    #path = '/cis/home/younes/MorphingData/Biocard/'
    sub1 = '0186193_1_6'
    #sub2 = '1449400_1_L'
    sub2 = '1229175_2_4'
    f0 = []
    f0.append(surfaces.Surface(filename = path+sub1+'_hippo_L_reg.byu_10_6.5_2.5.byu'))
    f0.append(surfaces.Surface(filename = path2+sub1+'_amyg_L_reg.byu_10_6.5_2.5.byu'))
    f1 = []
    f1.append(surfaces.Surface(filename = path+sub2+'_hippo_L_reg.byu_10_6.5_2.5.byu'))
    f1.append(surfaces.Surface(filename = path2+sub2+'_amyg_L_reg.byu_10_6.5_2.5.byu'))

    #f0[0].smooth()
    #f0[1].smooth()
    #f1[0].smooth()
    #f1[1].smooth()


    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 15.)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = 1.5)
    f0[0].vertices[:,1] += 2. ;
    f1[0].vertices[:,1] += 1. ;
    f0[0].vertices[:,2] += 2.0 ;
    f1[0].vertices[:,2] += 1.0 ;

    f0[0].computeCentersAreas()
    f1[0].computeCentersAreas()

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=2.5, sigmaError=1, errorType='varifold')
    f = (SurfaceMatching(Template=f0, Target=f1, outputDir=outputDir,param=sm, mu=.01,regWeightOut=1., testGradient=True,
                         typeConstraint='slidingV2', maxIter_cg=1000, maxIter_al=100, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()
