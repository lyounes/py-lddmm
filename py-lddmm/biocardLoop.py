import csv
import numpy as np
import logging
import loggingUtils
import surfaces
from surfaces import *
from kernelFunctions import *
import surfaceMatching
import threading
import Queue

def threadfun(q):
    while True:
        print str(q.qsize())+' jobs left'
        f = q.get()
        try:
            f.optimizeMatching()
            #break
        except NameError:
            print 'Exception'
        q.task_done()

def runLongitudinalSurface(minL=3, atrophy=False):
    if atrophy:
        import surfaceTimeSeriesAtrophy as match
    else:
        import surfaceTimeSeries as match

    
    with open('/cis/home/younes/MATLAB/shapeFun/CA_STUDIES/BIOCARD/filelist.txt','r') as csvf:
        rdr = list(csv.DictReader(csvf,delimiter=',',fieldnames=('lab','isleft','id','filename')))
        files = []
        lab = np.zeros(len(rdr), dtype=long) ;
        lr = np.zeros(len(rdr), dtype=long) ;
        previousLab = 0
        j = 0 
        currentFile = []
        for row in rdr:
            if int(row['lab']) == previousLab:
                if int(row['isleft']) == 1:
                    if len(row['filename']) > 0:
                        currentFile += [row['filename']]
            else:
                if len(currentFile) >= minL:
                    files +=[currentFile]
                if int(row['isleft']) == 1:
                    if len(row['filename']) > 0:
                        currentFile = [row['filename']]
                else:
                    currentFile = [] ;
                previousLab = int(row['lab'])
                
    outputDir = '/Users/younes/Development/Results/biocardTS/piecewise'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging(fileName='info')

    rdir = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
    fv0 = surfaces.Surface(filename='/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template/newTemplate.byu')
    K1 = Kernel(name='laplacian', sigma = 6.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='varifold')

    q = Queue.Queue()
    for k,s in enumerate(files):
        fv = []
        for fn in s:
                try:
                    fv += [surfaces.Surface(filename=fn+'.byu')]
                except NameError as e:
                    print e
  

        outputDir = '/cis/home/younes/Results/biocardTS/piecewise_'+str(k)

        try:
            if atrophy:
                    f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                            affine='euclidean', testGradient=False, affineWeight=.1,  maxIter_cg=1000, mu=0.0001)
            else:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                        affine='euclidean', testGradient=False, affineWeight=.1,  maxIter=1000)
        except NameError:
            print 'exception'
 
        #, affine='none', rotWeight=0.1))
        q.put(f)

    for k in range(10):
        w = threading.Thread(target=threadfun, args=(q,))
        w.setDaemon(True)
        w.start()
        #f.optimizeMatching()

    q.join()


if __name__=="__main__":
    runLongitudinalSurface(atrophy=True)
