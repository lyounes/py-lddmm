#! /usr/bin/env python
import os.path
import glob
import argparse
import subprocess



def createLongitudinalSurfaceScripts(minL=3):

    targetDir = '/cis/home/younes/MorphingData/TimeseriesResults/'
    source = '/cis/home/younes/github/registration/py-lddmm'
    allDir = glob.glob(targetDir+'*')
    for d in allDir:
        nscan = 0 
        while os.path.exists(d+'/imageOutput_time_{0:d}_channel_0.vtk'.format(nscan)):
            nscan += 1
        print d, nscan
        if nscan >= minL:
            shname = targetDir+'scripts/s'+ os.path.basename(d) +'.sh'
            with open(shname, 'w') as fname:
                fname.write('#!/bin/bash\n')
                fname.write('#$ -cwd\n')
                fname.write('#$ -j y\n')
                fname.write('#$ -S /bin/bash\n')
                fname.write('#$ -pe orte 8\n')
                fname.write('#$ -M laurent.younes@jhu.edu\n')
                fname.write('#$ -o /dev/null\n')
                fname.write('cd '+ source +'\n')
                #fname.write('source ~/.bashrc\n which python\n')
                #fname.write('which python\necho $PATH\necho $LD_LIBRARY_PATH\necho $SHELL\n')
                fname.write('python L2TimeSeriesSecondOrder.py ' +  os.path.basename(d) + '\n')
            cstr = "qsub  " + shname
            print cstr
            subprocess.call(cstr, shell=True)


 

if __name__=="__main__":
    createLongitudinalSurfaceScripts()
