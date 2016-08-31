import glob
import loggingUtils
import surfaces
from surfaces import *
from kernelFunctions import Kernel
from surfaceTemplate import *

def main():
    files = glob.glob('/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/2_qc_flipped_registered/*L_reg.byu')
    print len(files)
    fv1 = []
    for k in range(10):
        fv1.append(surfaces.Surface(filename = files[k]))
    K1 = Kernel(name='gauss', sigma = 6.5)
    K0 = Kernel(name='gauss', sigma = 1.0)
    Kdist = Kernel(name='gauss', sigma = 2.5)

    loggingUtils.setup_default_logging('/Users/younes/Results/surfaceTemplateBiocard', fileName='info.txt', stdOutput = True)
    sm = SurfaceTemplateParam(timeStep=0.1, KparDiff=K1, KparDiff0 = K0, KparDist=Kdist, sigmaError=1., errorType='varifold')# internalCost='h1')
    f = SurfaceTemplate(HyperTmpl=fv1[0], Targets=fv1, outputDir='/Users/younes/Results/surfaceTemplateBiocard',param=sm, testGradient=True, 
                        lambdaPrior = 0.01,
                        maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
    f.computeTemplate()


if __name__=="__main__":
    main()
