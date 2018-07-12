#! /usr/bin/env python
import csv
import argparse
from Common import loggingUtils
from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Surfaces import surfaceMatching, surfaces
from affineRegistration import *



def runLongitudinalSurface(template, targetList, minL=3, atrophy=False, splines=False, resultDir='.'):
    if atrophy:
        pass
    elif splines:
        from surfaces import secondOrderMatching as match
    else:
        pass

    
    with open(targetList,'r') as csvf:
        rdr = list(csv.DictReader(csvf,delimiter=',',fieldnames=('lab','isleft','id','filename')))
        files = []
        ids = []
        previousLab = 0
        currentFile = []
        currentId = []
        for row in rdr:
            #print row
            if int(row['lab']) == previousLab:
                if int(row['isleft']) == 1:
                    if len(row['filename']) > 0:
                        currentFile += [row['filename']]
                        currentId = row['id']
            else:
                #print row
                if len(currentFile) >= minL:
                    files +=[currentFile]
                    ids += [currentId]
                if int(row['isleft']) == 1:
                    if len(row['filename']) > 0:
                        currentFile = [row['filename']]
                else:
                    currentFile = [] ;
                previousLab = int(row['lab'])
    #print len(files)
    #return                  
    #print files
    
    if len(template) > 0:
        fv0 = surfaces.Surface(filename=template)
        z = fv0.surfVolume()
        if z < 0:
            fv0.flipFaces()
    K1 = Kernel(name='laplacian', sigma = 2.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=1.5, sigmaError=1., errorType='varifold')

    #files = [files[1],files[5],files[8]]
    #files = [files[9]]
    #selected = (1,2,3) 
    #selected = (1,2,3,4,11) 
    selected = (2,4) 
    #selected = (4,11,22,27) 
    logset = False
    for k in selected:
        outputDir = resultDir + ids[k]
        info_outputDir = outputDir
        if __name__ == "__main__" and (not logset):
            loggingUtils.setup_default_logging(info_outputDir, fileName='info', stdOutput=False)
            logset = True
        else:
            loggingUtils.setup_default_logging(info_outputDir, fileName='info', stdOutput=False)

        s = files[k]
        logging.info(s[0])
        fv = []
        #print s[0]
        for fn in s:
                try:
                    #fv += [surfaces.Surface(filename=fn+'.byu')]
                    fv1 = surfaces.Surface(filename=fn)
                    z = fv1.surfVolume()
                    if z < 0:   
                        fv1.flipFaces()
                    fv += [fv1]
                    logging.info(fn)
                except NameError as e:
                    print e
        logging.info(outputDir)
        ## Reversing order to test bias
        #fv.reverse()
        if len(template) == 0:
            fv0 = surfaces.Surface(surf=fv[0])

        for fs in fv:        
            R0, T0 = rigidRegistration(surfaces = (fs.vertices, fv0.vertices),  verb=False, temperature=10., annealing=True, translationOnly=True)
            fs.updateVertices(np.dot(fs.vertices, R0.T) + T0)


        try:
            if atrophy:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                            affine='euclidean', testGradient=True, affineWeight=.1,  maxIter_cg=50, maxIter_al=50, mu=0.0001)
            elif splines:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                            affine='none', testGradient=False, affineWeight=.1,  maxIter=1000)                
            else:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                        affine='none', testGradient=False, affineWeight=.1,  maxIter=1000)
        except NameError:
            print 'exception'
 
        try:
            f.optimizeMatching()
        except NameError:
            print 'Exception'
 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='runs longitudinal surface matching based on an input file')
    parser.add_argument('--template', metavar='template', type = str, default = '',help='template')
    parser.add_argument('targetList', metavar='targetlist', type = str, help='file containing the list of targets')
    parser.add_argument('--results', metavar = 'resultDir', type = str, dest = 'resultDir', default = '.', help='Output directory')
    args = parser.parse_args()

    if len(args.template) == 0:
        template = '/Users/younes/Development/Data/sculptris/AtrophyLargeNoise/baseline.vtk'
        targetList = []
        for k in range(1,11):
            targetList.append('/Users/younes/Development/Data/sculptris/AtrophyLargeNoise/followUp'+str(k)+'.vtk')
        resultDir = 'Users/younes/Development/Results/AtrophySim'
    else:
        template = args.template
        targetList = args.targetList
        resultDir = args.resultDir



    #template: /cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template/newTemplate.byu'
    #targetList: '/cis/home/younes/MATLAB/shapeFun/CA_STUDIES/BIOCARD/filelist.txt'
    #Results: '/cis/home/younes/Results/biocardTS/withAtrophyRerun'
    
    
    runLongitudinalSurface(template, targetList, atrophy=False, splines=False, minL=5, resultDir=resultDir)
