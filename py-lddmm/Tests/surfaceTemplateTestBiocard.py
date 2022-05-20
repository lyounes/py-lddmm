from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from base import loggingUtils
from base.surfaces import *
from base.kernelFunctions import Kernel
from base.surfaceTemplate import *
import glob

def main():
    files = glob.glob('/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/2_qc_flipped_registered/*L_reg.byu')
    print(len(files))
    fv1 = []
    for k in range(10):
        fv1.append(surfaces.Surface(surf = files[k]))
    K1 = Kernel(name='gauss', sigma = 6.5)
    K0 = Kernel(name='gauss', sigma = 1.0)
    Kdist = Kernel(name='gauss', sigma = 2.5)

    loggingUtils.setup_default_logging('/cis/home/younes/Development/Results/surfaceTemplateBiocard', fileName='info.txt', stdOutput = True)
    sm = SurfaceTemplateParam(timeStep=0.1, KparDiff=K1, KparDiff0 = K0, KparDist=Kdist, sigmaError=1., errorType='varifold')# internalCost='h1')
    f = SurfaceTemplate(HyperTmpl=fv1[0], Targets=fv1, outputDir='/cis/home/younes/Development/Results/surfaceTemplateBiocard',param=sm, testGradient=True, 
                        lambdaPrior = 0.01,
                        maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
    f.computeTemplate()


if __name__=="__main__":
    main()
