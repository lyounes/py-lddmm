#!/opt/local/bin/python2.7
import os
from os import path
import argparse
import diffeo
import pointSets
import surfaces
import logging
import loggingUtils
from kernelFunctions import *
from affineRegistration import *


def main():


    parser = argparse.ArgumentParser(description='runs surface matching registration over directories (relative to the template)')
    parser.add_argument('template', metavar='template', nargs='+', type = str, help='template (list)')
    parser.add_argument('target', metavar='target', nargs='+', type = str, help='target (list)')
    #parser.add_argument('template', metavar='template', type = str, help='template')
    #parser.add_argument('target', metavar='target', type = str, help='target')
    parser.add_argument('--typeKernel', metavar='typeKernel', type=str, dest='typeKernel', default = 'gauss', help='kernel type') 
    parser.add_argument('--sigmaKernel', metavar='sigmaKernel', type=float, dest='sigmaKernel', default = 6.5, help='kernel width') 
    parser.add_argument('--sigmaDist', metavar='sigmaDist', type=float, dest='sigmaDist', default = 2.5, help='kernel width (error term); (default = 2.5)') 
    parser.add_argument('--sigmaError', metavar='sigmaError', type=float, dest='sigmaError', default = 1.0, help='std error; (default = 1.0)') 
    parser.add_argument('--typeError', metavar='typeError', type=str, dest='typeError', default = 'varifold', help='type error term (default: varifold)') 
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = '', help='Output directory')
    parser.add_argument('--tmpOut', metavar = 'tmpOut', type = str, dest = 'tmpOut', default = '', help='info files directory')
    parser.add_argument('--rigid', action = 'store_true', dest = 'rigid', default = False, help='Perform Rigid Registration First')
    parser.add_argument('--logFile', metavar = 'logFile', type = str, dest = 'logFile', default = 'info.txt', help='Output log file')
    parser.add_argument('--stdout', action = 'store_true', dest = 'stdOutput', default = False, help='To also print on standard output')
    parser.add_argument('--scaleFactor', metavar='scaleFactor', type=float, dest='scaleFactor',
                        default = 1, help='scale factor for all surfaces') 
    parser.add_argument('--atrophy', action = 'store_true', dest = 'atrophy', default = False, help='force atrophy')
    parser.add_argument('--atrophyVolume', action = 'store_true', dest = 'atrophyVolume', default = False, help='force atrophy')
    parser.add_argument('--symmetric', action = 'store_true', dest = 'symmetric', default = False, help='Use error term on both template and target')
    parser.add_argument('--flipTarget', action = 'store_true', dest = 'flipTarget', default = False, help='Flip target orientation')
    parser.add_argument('--mu', metavar='mu', type=float, dest='mu', default = 0.001, help='mu for augmented lagrangian') 
    args = parser.parse_args()

    if args.dirOut == '':
        args.dirOut = '.'

    if args.tmpOut == '':
        args.tmpOut = args.dirOut + '/tmp'
    if not os.path.exists(args.dirOut):
        os.makedirs(args.dirOut)
    if not os.path.exists(args.tmpOut):
        os.makedirs(args.tmpOut)
    loggingUtils.setup_default_logging(args.tmpOut, fileName=args.logFile, stdOutput = args.stdOutput)

    if args.atrophy or args.atrophyVolume:
        import surfaceMatchingAtrophy as smt
    else:
        import surfaceMatching as smt

    tmpl = surfaces.Surface(filelist=args.template)
    tmpl.vertices *= args.scaleFactor
    K1 = Kernel(name=args.typeKernel, sigma = args.sigmaKernel)
    sm = smt.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=args.sigmaDist, sigmaError=args.sigmaError, errorType=args.typeError)
    fv = surfaces.Surface(filelist=args.target)
    fv.vertices *= args.scaleFactor
    if args.flipTarget:
        logging.info('Flipping Target Orientation')
        print surfaces.currentNormDef(tmpl, fv, sm.KparDist)
        fv.flipFaces()
        print surfaces.currentNormDef(tmpl, fv, sm.KparDist)
    #print fv.vertices

    if args.rigid:
        R0, T0 = rigidRegistration(surfaces = (fv.vertices, tmpl.vertices),  verb=False, temperature=10., annealing=True)
        fv.updateVertices(np.dot(fv.vertices, R0.T) + T0)

        #print fv.vertices

    if args.atrophy or args.atrophyVolume:
        f = smt.SurfaceMatching(Template=tmpl, Target=fv, outputDir=args.tmpOut,param=sm, testGradient=False, mu = args.mu, symmetric=args.symmetric,
                            maxIter_cg=1000, affine= 'euclidean', rotWeight=.01, transWeight = .01, scaleWeight=10., affineWeight=100., volumeOnly=args.atrophyVolume)
    else:
        f = smt.SurfaceMatching(Template=tmpl, Target=fv, outputDir=args.tmpOut,param=sm, testGradient=False, symmetric=args.symmetric, saveTrajectories = True,
                            maxIter=2000, affine= 'euclidean', rotWeight=.01, transWeight = .01, scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    u = path.split(args.target)
    [nm,ext] = path.splitext(u[1])
    f.fvDef.savebyu(args.dirOut+'/'+nm+'Def.byu')
    #if args.saveTrajectories:
        

if __name__=="__main__":
    main()
