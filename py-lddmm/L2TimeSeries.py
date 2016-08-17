import loggingUtils
import surfaces
from surfaces import *
from kernelFunctions import *
import surfaceMatching
#import secondOrderMatching as match

def compute(Atrophy=False):
    if Atrophy:
        import surfaceTimeSeriesAtrophy as match
    else:
        import surfaceTimeSeries as match

    #outputDir = '/Users/younes/Development/Results/biocardTS/spline'
    outputDir = '/Users/younes/Development/Results/L2TimeSeriesAtrophy'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput = True)
    else:
        loggingUtils.setup_default_logging()

    rdir = '/Users/younes/Development/Data/TimeseriesResults/' ;
    sub = '016_S_4584_L'
    nscan = 4
    #sub = '2729611'
    fv = []
    fv0 = surfaces.Surface(filename='/Users/younes/Development/Data/TimeseriesResults/estimatedTemplate.byu')
    for k in range(nscan):
        fv = fv + [rdir + sub + '/imageOutput_time_{0:d}_channel_0.vtk'.format(k) ]

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 2.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=.1, errorType='L2Norm')
    if Atrophy:
        f = match.SurfaceMatching(Template=fv0, fileTarg=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                affine='euclidean', testGradient=True, rotWeight=1.,  maxIter_cg=100, mu=0.0001)
    else:
       f = match.SurfaceMatching(Template=fv0, fileTarg=fv, outputDir=outputDir, param=sm, regWeight=1.,
                                affine='euclidean', testGradient=False, affineWeight=10,  maxIter=1000)
 
    #, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute(True)

