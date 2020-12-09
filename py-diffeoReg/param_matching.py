from ivector import Ivector, Domain
import numpy as np
from copy import copy, deepcopy



# /** 
#     parameter class. Reads parameter files written as follows
# 
# Start with an integer which is the dimension of the dataset
# (dim = 2,3 1 will come soon). It then consists in a sequence of keywords followed by some
# complementary input. Here is a detailed description (keywords always
# end with ':') 
# 
#   template: filename 
# 	First image file
#   target: filename 
# 	Second image file
#   tempList: nbFiles file1 file2
#              A list of template files (for multimodal matching)
#   targList: nbFiles file1 file2
#              A list of target files (for multimodal matching)
# 
#    result: fileName
#    An ascii file to store numerical output (energy, distance)
# 
#  dataSet: N filename1 ... filenameN 
#              List of images (for procrustean averaging)
# 
# dataMomenta: N filename1 ... filenameN 
#              List of initial momenta (relative to a template for simple averaging)
# 
# momentum: filename1 filename2 
#         Two files for parallel translation. The first momentum provides 
# 	the geodesic along which the second momentum is transported. 
# 	The momenta are outputs of matchShoot.
# 
# initialMomentum: fileName
#          To initialize matching procedures with a non zero momentum
# 
# outdir: the directory where the output images are saved
# 
# scale: s1 .. sdim
# 	Integers providing scaling factors for each dimension default 1 for each dimension
# 
# kernel: type shape size [degree]
# 	parameters defining the smoothing kernel for vector fields:
#             type is either gauss or laplacian
#             shape defines a normalized scale
#             size is a scaling factor (in pixels)
#            degree is required for the laplacian kernel only, and is an integer from 0 to 4
# 
# gauss_kernel: size
# 	parameters specifying a gaussian kernel spread for preliminar
# 	image smoothing default 1 
# 
# crop1: min1 max1 .... mindim maxdim
# 	specifies a multidimensional rectangle to crop the first image
# 	(default: no cropping).
# 
# crop2: min1 max1 .... mindim maxdim
# 	specifies a multidimensional rectangle to crop the second image
# 	(default: no cropping).
# 
# affine: a11 a12 ...
# 	applies the inverse of dim x dim+1 matrix A to the target
# 	image. No affine registration is run.
# 
# affineReg: affKey num_iterations
# 	computes an optimal affine registration of the target before
# 	matching. affKey is one of the following: rotation (rigid
# 	transf.), similitude (rigid+scaling), special (volume
# 	conserving) or general (the whole affine group).
# 
# space_resolution: res1 ... resdim
# 	allows for non constant resolutions across dimensions. Default
# 	1 ... 1
# 
# minVarGrad: m
# 	upper bound on the energy variation in gradient descent stopping rule.
# 	Default: 0.001
# 
# * sigma: s
# 	the coefficient before the data attchment term is 1/s^2 for
# 	matchShoot. default: 1
# 
# * lambda: s
# 	the coefficient before the data attachment term is s for
# 	metamorphosis (sorry...) default:1
# 
# * nb_iter: N
# 	maximal number of gradient descent iterations in matching
# 	programs
# 
# * nb_semi: n
#    number of iteration for semi implicit integration (default: 0)
# 
# * epsMax: eps
# 	maximum step in gradient descent. Default: 10
# 
# * time_disc: T
# 	the time discretization for metamorphoses default: 30
# 
# parallelTimeDisc: T
# 	Time discretization for parallel translation default 20
# 
# keepTarget:
# 	Work with the original target file (apply affine transformations to the template)
# 
# expand: margin1 ... marginDim value
#     Expands image with margink in kth dimension, with specified value. If value =-1, picks average value on the image boundary
# 
# expandToMaxSize:
#     If present, expand images to fit the largest one
# 
# maxShootTime:
#    Maximum number of steps for shooting procedure
# 
# scaleScalars:
#    Rescale all image values between 0 and 100
# 
# goldenSearch:
#    Use golden line search for gradient descent
# 
# binarize:
#     Work with Thresholded binary images
# 
# revertIntensities:
#      Revert image internsities
# 
# nb_threads:
#     Maximum number of threads for parallel implementation
# 
# 
# ******   Input file format
# 	Images are read in most image formats in 2D, and in Analyze format in 3D.
# */

class param_matching:
    def __init__(self, fileIn= None):
        self.paramkey = {"template:": self.readTEMPLATE,
                         "target:": self.readTARGET,
                         "tempList:":self.readTEMPLIST,
                         "targList:": self.readTARGLIST,
                         "result:": self.readRESULT,
                         "dataSet:": self.readDATASET,
                         "dataMomenta:": self.readDATAMOM,
                         "momentum:": self.readMOMENTUM,
                         "initialMomentum:": self.readINITIALMOMENTUM,
                         "miscFiles:": self.readAFILE,
                         "miscParam:": self.readAPARAM,
                         "outdir:": self.readOUTDIR,
                         "scale:": self.readSCALE,
                         "kernel:": self.readKERNEL,
                         "smoothing_kernel:": self.readGAUSS_KERNEL,
                         "crop1:": self.readCROP1,
                         "crop2:": self.readCROP2,
                         "affine:": self.readAFFINE,
                         "space_resolution:": self.readSPACERES,
                         "affineReg:": self.readAFFINEREG,
                         "projectImage:": self.readPROJECTIMAGE,
                         "applyAffineToTemplate:": self.readAPPLYAFFINETOTEMPLATE,
                         "keepTarget:": self.readKEEPTARGET,
                         "accuracy:": self.readACCURACY,
                         "sigma:": self.readSIGMA,
                         "nb_iter:": self.readNB_ITER,
                         "nb_semi:": self.readNBSEMI,
                         "nb_cg_meta:": self.readNBCGMETA,
                         "epsMax:": self.readEPSMAX,
                         "time_disc:": self.readTIMEDISC,
                         "parallelTimeDisc:": self.readPARALLELTIMEDISC,
                         "lambda:": self.readWEIGHT,
                         "gradientThreshold:": self.readGRADIENT_THRESHOLD,
                         "minVarGrad:": self.readMINVAREN,
                         "expand:": self.readEXPAND,
                         "maxShootTime:": self.readMAXSHOOTTIME,
                         "scaleScalars:": self.readSCALESCALARS,
                         "doNotModifyTemplate:": self.readDNMT,
                         "doNotModifyImages:": self.readDNMI,
                         "goldenSearch:": self.readGS,
                         "binarize:": self.readBINARIZE,
                         "flipTarget:": self.readFLIPTARGET,
                         "initMeta:": self.readINIT_META,
                         "expandToMaxSize:": self.readEXPANDTOMAXSIZE,
                         "useVectorMomentum:": self.readUSEVECTORMOMENTUM,
                         "revertIntensities:": self.readREVERTINTENSITIES,
                         "matchDensities:": self.readMATCHDENSITIES,
                         "nbThreads:": self.readNB_THREADS,
                         "continue:": self.readCONTINUE,
                         "saveMovie:": self.readSAVEMOVIE,
                         "periodic:": self.readPERIODIC,
                         "debug:": self.readDEBUG,
                         "quiet:": self.readKERNEL,
                         "normalizeKernel:": self.readNORMALIZEKERNEL
                         }

        self.affMap = ["none", "translation", "rotation", "similitude","special","general"]
        self.kernelMap = [ "gauss", "gaussLandmarks", "laplacian"]

        self.fileTemp = None
        self.fileTarg = None
        self.fileMom1 = None
        self.fileMom2 = None
        self.fileInitialMom= None
        self.fileTempList = []
        self.fileTargList = []
        self.dataSet = []
        self.dataMom = []
        self.auxFiles = []
        self.auxParam = []
        self.outDir = '.'
        self.cont = False
        self.saveMovie = False
        self.revertIntensities = False
        self.useVectorMomentum = False
        self.expandToMaxSize = False
        self.binarize = False
        self.binThreshold = 1
        self.nb_threads = 1
        self.gs = False
        self.keepTarget = False
        self.applyAffineToTemplate = False
        self.initTimeData = False
        self.readBinaryTemplate = False
        self.doNotModifyTemplate = False
        self.doNotModifyImages = False
        self.foundTemplate = False
        self.foundTarget = False
        self.foundResult = False
        self.foundAuxiliaryFile = False
        self.foundAuxiliaryParam = False
        self.foundInitialMomentum = False
        self.flipTarget = False
        self.initMeta = False
        self.initMetaNIter = 50
        self.flipDim = 0
        self.saveProjectedMomentum = True
        self.matchDensities = False
        self.foundScale = False
        self.gradInZ0 = True
        self.weight = 1
        self.time_disc =30
        self.nb_iter=1000
        self.nbCGMeta = 3
        self.sigma = 1
        self.epsMax = 10
        self.accuracy = 1.1
        self.sigmaGauss = -1
        self.gradientThreshold = -1
        self.epsilonTangentProjection = 0.01
        self.sizeGauss = 50
        self.kernel_type = 'gauss'
        self.sigmaKernel = .1
        self.orderKernel = 0
        self.inverseKernelWeight = 0.0001
        self.sizeKernel = 100
        self.nb_iterAff = 1000
        self.affine_time_disc = 20
        self.tolGrad = 0
        self.verb=1
        self.printFiles = 1
        self.nb_semi = 3
        self.minVarEn = 0.001
        self.type_group = 'none'
        self.scaleScalars = False
        self.scaleThreshold = 100
        self.crop1 = False
        self.crop2 = False
        self.affine = False
        self.spRes = False
        self.projectImage = False
        self.expand_value = -1
        self.Tmax = 10
        self.parallelTimeDisc = 20
        self.kernelNormalization = 1
        self.normalizeKernel = False
        self.doDefor = True
        self.periodic = 0


    def readstr(self, k, input):
        return input[k+1], k+1

    def readFile(self, fileName):
        input = []
        readDim = False
        with open(fileName,'r') as ifs:
            line = ifs.readline()
            while len(line) > 0:
                if line[0] != '\n' and line[0] != '#':
                    print(line)
                    content = line.split()
                    if not readDim:
                        if len(content) < 2 or content[0] != 'dim:':
                            return 'Error: file should start with dimension'
                        else:
                            self.ndim = int(content[1])
                    else:
                        input += content
        self.readInput(input)

    def readInput(self, input):
        k = 0
        newkey = True
        while k < len(input):
            content = input[k].split()
            kwd = content[0]
            if kwd in self.paramkey.keys():
                fct = self.paramkey[kwd]
                k = fct(k, input) + 1

        if not self.foundScale:
            self.dim = np.ones(self.ndim, dtype=int)

        if not self.spRes:
            self.spaceRes = np.ones(self.ndim, dtype=int)

        if self.scaleScalars and self.gradientThreshold < 0:
            self.gradientThreshold = self.scaleThreshold / 10


    def readOUTDIR(self, k, input):
        self.fileTemp = input[k+1]
        return k+1

    def readTEMPLATE(self, k, input):
        self.fileTemp = input[k+1]
        self.foundTemplate = True
        return k+1

    def readRESULT(self, k, input):
        self.fileResult = input[k+1]
        self.foundResult = True
        return k+1

    def readTARGET(self, k, input):
        self.fileTarg = input[k+1]
        self.foundTarget = True
        return k+1

    def readDATASET(self, k, input):
        l = k
        nb = int(input[l])
        self.dataSet = []
        for i in range(nb):
            self.dataSet.append(input[l+1])
            l += 1
        self.foundTarget = True
        return l


    def readTEMPLIST(self, k, input):
        l = k
        nb = int(input[l])
        self.fileTempList = []
        for i in range(nb):
            self.fileTempList.append(input[l + 1])
            l += 1
        self.foundTemplate = True
        return l

    def readTARGLIST(self, k, input):
        l = k
        nb = int(input[l])
        self.fileTargList = []
        for i in range(nb):
            self.fileTargList.append(input[l + 1])
            l += 1
        self.foundTarget = True
        return l

    def readDATAMOM(self, k, input):
        l = k
        nb = int(input[l])
        self.dataMom = []
        for i in range(nb):
            self.dataMom.append(input[l + 1])
            l += 1
        return l

    def readAFILE(self, k, input):
        l = k
        nb = int(input[l])
        self.auxFiles = []
        for i in range(nb):
            self.auxFiles.append(input[l + 1])
            l += 1
        self.foundAuxiliaryFile = True
        return l

    def readAPARAM(self, k, input):
        l = k
        nb = int(input[l])
        self.auxParam = np.zeros(nb)
        for i in range(nb):
            self.auxParam[i] = float(input[l + 1])
            l += 1
        self.foundAuxiliaryParam = True
        return l

    def readINITIALMOMENTUM(self, k, input):
        self.fileInitialMomentum = input[k+1]
        self.foundInitialMomentum = True
        return k+1

    def readMOMENTUM(self, k, input):
        self.fileMom1 = input[k+1]
        self.fileMom1 = input[k+2]
        return k+2

    def readUSEVECTORMOMENTUM(self, k, input):
        self.useVectorMomentum = True
        return k

    def readREVERTINTENSITIES(self, k, input):
        self.revertIntensities = True
        return k

    def readPERIODIC(self, k, input):
        self.periodic = True
        return k

    def readDEBUG(self, k, input):
        self.verb = 2
        return k

    def readQUIET(self, k, input):
        self.verb = 0
        return k

    def readCONTINUE(self, k, input):
        self.cont = True
        return k

    def readMATCHDENSITIES(self, k, input):
        self.matchDensities = True
        return k

    def readNB_THREADS(self, k, input):
        self.nb_threads = int(input[k+1])
        return k + 1

    def readSCALE(self, k, input):
        l = k
        self.dim = np.zeros(self.ndim)
        for i in range(self.ndim):
            self.dim[i] = float(input[l])
            l += 1
        return l

    def readBINARIZE(self, k, input):
        self.binarize = True
        return k

    def readFLIPTARGET(self, k, input):
        self.flipTarget = True
        return k

    def readINIT_META(self, k, input):
        self.initMeta = True
        self.initMetaNIter = int(input[k+1])
        return k+1

    def readEXPANDTOMAXSIZE(self, k, input):
        self.expandToMaxSize = True
        return k

    def readKERNEL(self, k, input):
        self.kernel_type = input[k+1]
        l = k+1
        if self.kernel_type == 'gauss_lmk':
            self.sigmaKernel = float(input[l+1])
            l += 1
        elif self.kernel_type == 'gauss':
            self.sigmaKernel = float(input[l + 1])
            self.sizeKernel = 4 * ((2 * int(input[l + 2])) // 4)
            l += 2
        elif self.kernel_type == 'laplacian':
            self.sigmaKernel = float(input[l+1])
            self.sizeKernel = 4 * ((2*int(input[l+2])) // 4)
            self.orderKernel = int(input[l+3])
            l += 3

        return l

    def readGAUSS_KERNEL(self, k, input):
        self.sigmaGauss = 0.05
        self.sizeGauss = 2 * int(input[k+1])
        return k+1

    def readMINVAREN(self, k, input):
        self.minVarEn = float(input[k+1])
        return k+1

    def readGRADIENT_THRESHOLD(self, k, input):
        self.gradientThreshold = float(input[k+1])
        return k+1

    def readACCURACY(self, k, input):
        self.accuracy = float(input[k+1])
        return k+1

    def readEXPAND(self, k, input):
        l = k
        self.expand_margin = np.zeros(self.ndim)
        for i in range(self.ndim):
            self.expand_margin[i] = int(input[l+1])
            l += 1
        self.expand_margin = float(input[l+1])
        return l+1

    def readCROP1(self, k, input):
        im = Ivector(self.ndim)
        iM = Ivector(self.ndim)
        l = k
        for i in range(self.ndim):
            im[i] = int(input[l+1])
            iM[i] = int(input[l+2])
            l += 2
        self.crop1 = True
        self.cropD1 = Domain(im, iM)
        return l


    def readCROP2(self, k, input):
        im = Ivector(self.ndim)
        iM = Ivector(self.ndim)
        l = k
        for i in range(self.ndim):
            im[i] = int(input[l + 1])
            iM[i] = int(input[l + 2])
            l += 2
        self.crop2 = True
        self.cropD2 = Domain(im, iM)
        return l

    def readAFFINE(self, k, input):
        l = k
        self.affMat = np.zeros((self.ndim, self.ndim+1))
        for i in range(self.ndim):
            for j in range(self.ndim+1):
                self.affMat[i][j] = float(input[l+1])
                l += 1
        self.affine = True

    def readMAXSHOOTTIME(self, k, input):
        self.Tmax = int(input[k+1])
        return k+1

    def readNBCGMETA(self, k, input):
        self.nbCGMeta = int(input[k + 1])
        return k + 1

    def readSPACERES(self, k, input):
        self.spaceRes = np.zeros(self.ndim)
        l = k
        for i in range(self.ndim):
            self.spaceRes[i] = float(input[l+1])
            l += 1
        self.spRes = True
        return l

    def readPROJECTIMAGE(self, k, input):
        self.projectImage = True
        return k

    def readGS(self, k, input):
        self.gs = True
        return k

    def readAFFINEREG(self, k, input):
        self.type_group = input[k+1]
        nb_it = int(input[k+2])
        if self.type_group not in self.affMap:
            print("Unknown type of affine group")
            return
        if nb_it > 0:
            self.nb_iterAff = nb_it
        return k+2

    def readKEEPTARGET(self, k, input):
        self.keepTarget = True
        return k

    def readAPPLYAFFINETOTEMPLATE(self, k, input):
        self.applyAffineToTemplate = True
        return k

    def readSCALESCALARS(self, k, input):
        self.scaleScalars = True
        self.scaleThreshold = float(input[k+1])
        return k+1

    def readPARALLELTIMEDISC(self, k, input):
        self.parallelTimeDisc = float(input[k+1])
        return k+1

    def readTIMEDISC(self, k, input):
        self.time_disc = float(input[k+1])
        return k+1

    def readWEIGHT(self, k, input):
        self.weight = float(input[k+1])
        self.sigma = 1 /np.sqrt(self.weight)
        return k+1

    def readNB_ITER(self, k, input):
        self.nb_iter = int(input[k+1])
        return k+1

    def readNBSEMI(self, k, input):
        self.nb_semi = int(input[k + 1])
        return k + 1

    def readSIGMA(self, k, input):
        self.sigma = float(input[k+1])
        self.weight = 1/(self.sigma**2)
        return k+1

    def readEPSMAX(self, k, input):
        self.epsMax = float(input[k+1])
        return k+1

    def readDNMT(self, k, input):
        self.doNotModifyTemplate = True
        return k

    def readDNMI(self, k, input):
        self.doNotModifyImages = True
        return k

    def readSAVEMOVIE(self, k, input):
        self.saveMovie = True
        return k

    def readNORMALIZEKERNEL(self, k, input):
        self.normalizeKernel = True
        return k

    def defaultArray(self, dm):
        self.ndim = dm
        self.dim = np.ones(self.ndim, dtype=int)
        self.spaceRes = np.ones(self.ndim)

    def copy(self):
        return copy(self)

    # def copy_from(self, self):
    #     self = copy(self)
    #     self.ndim = self.ndim
    #     self.dim = self.dim.copy()
    #     self.cropD1 = self.cropD1.copy()
    #     self.cropD2 = self.cropD2.copy()
    #     self.fileTemp = self.fileTemp
    #     self.fileTarg = self.fileTarg
    #     self.fileMom1 = self.fileMom1
    #     self.fileMom2 = self.fileMom2
    #     self.fileInitialMom = self.fileInitialMom
    #     self.outDir = self.outDir
    #     self.fileTempList = self.fileTempList.copy()
    #     self.fileTargList = self.fileTargList.copy()
    #     self.dataSet = self.dataSet.copy()
    #     self.dataMom = self.dataMom.copy()
    #     self.auxFiles = self.auxFiles.copy()
    #     self.auxParam = self.auxParam.copy()
    #     self.spaceRes = self.spaceRes.copy()
    #     self.affMat = self.affMat.copy()
    #
    #     self.nb_threads = self.nb_threads
    #     self.kernelType = self.kernelType
    #     self.sigmaKernel =self.sigmaKernel
    #     self.sizeKernel = self.sizeKernel
    #     self.orderKernel = self.orderKernel
    #     self.inverseKernelWeight = self.inverseKernelWeight
    #     self.sigmaGauss = self.sigmaGauss
    #     self.Tmax = self.Tmax
    #     self.sizeGauss = self.sizeGauss
    #     self.type_group = self.type_group
    #     self.kernel_type = self.kernel_type
    #     self.verb = self.verb
    #     self.printFiles = self.printFiles
    #     self.gs = self.gs
    #     self.gradInZ0 = self.gradInZ0
    #     self.foundTemplate = self.foundTemplate
    #     self.foundResult = self.foundResult
    #     self.foundTarget = self.foundTarget
    #     self.foundScale = self.foundScale
    #     self.flipTarget = self.flipTarget
    #     self.flipDim = self.flipDim
    #     self.initMeta = self.initMeta
    #     self.crop1 =self.crop1
    #     self.crop2 = self.crop2
    #     self.affine = self.affine
    #       spRes = self.spRes
    #       projectImage = self.projectImage
    #       scaleScalars = self.scaleScalars
    #       binarize = self.binarize
    #       expandToMaxSize = self.expandToMaxSize
    #       matchDensities = self.matchDensities
    #       revertIntensities = self.revertIntensities
    #       binThreshold =self.binThreshold
    #       affine_time_disc = self.affine_time_disc
    #       nb_iterAff =  self.nb_iterAff
    #       nb_semi =  self.nb_semi
    #       tolGrad = self.tolGrad
    #       accuracy = self.accuracy
    #       minVarEn  = self.minVarEn
    #       expand_value = self.expand_value
    #       expand_margin.resize(self.expand_margin.size())
    #       for (unsigned int i=0 i<expand_margin.size() i++)
    #     expand_margin[i] = self.expand_margin[i]
    #       sigma = self.sigma
    #       epsMax = self.epsMax
    #       gradientThreshold = self.gradientThreshold
    #       epsilonTangentProjection = self.epsilonTangentProjection
    #       nb_iter = self.nb_iter
    #       cont = self.cont
    #       applyAffineToTemplate = self.applyAffineToTemplate
    #       kernelNormalization = self.kernelNormalization
    #       normalizeKernel = self.normalizeKernel
    #
    #       parallelTimeDisc = self.parallelTimeDisc
    #       time_disc = self.time_disc
    #       useVectorMomentum = self.useVectorMomentum
    #       doDefor = self.doDefor
    #       doNotModifyTemplate = self.doNotModifyTemplate
    #       doNotModifyImages = self.doNotModifyImages
    #       readBinaryTemplate = self.readBinaryTemplate
    #       keepFFTPlans = self.keepFFTPlans
    #       initTimeData = self.initTimeData
    #       foundInitialMomentum = self.foundInitialMomentum
    #       keepTarget = self.keepTarget
    #       saveProjectedMomentum = self.saveProjectedMomentum
    #       lambda = self.lambda
    #       scaleThreshold = self.scaleThreshold
    #       saveMovie = self.saveMovie
    #       periodic = self.periodic
    # }
    
    def printDefaults(self):
        print(param_matching())
        
    def __repr__(self):
        print("kernelType: ", self.kernel_type)
        print(f"kernel shape parameter: {self.sigmaKernel: .4f}")
        print(f"kernel size parameter: {self.sizeKernel: .4f}")
        print(f"kernel order parameter: {self.orderKernel: .4f} \n")

        print(f"match densities instead of images(flag): {self.matchDensities: b} \n")

        print(f"initial smoothing parameter: {self.sizeGauss: .4f}")
        print(f"flip target (flag): {self.flipTarget: b}")
        print(f"scale scalars (flag): {self.scaleScalars:b}")
        print(f"revert image intensity (flag): {self.revertIntensities: b}")
        print(f"expand images to maximum size (flag): {self.expandToMaxSize: b}\n")

        s = "Spatial resolution:  "
        for sp in self.spaceRes:
            s += f"{sp: 0.2f}"
        print(s)
        print(f"threshold for image gradients (relative): {self.gradientThreshold: .4f}")
        print(f"data term penalty (sigma): {self.sigma: 0.4f}\n")
        print(f"number of iterations for semi-Lagrangian integration: {self.nb_semi:d}")
        print(f"tolerance parameter for gradient descent: {self.tolGrad: 0.4f}")
        print(f"Minimal relative variation of objective function for optimization: {self.minVarEn: .4f}")
        print(f"max step in gradient descent (epsMax): {self.epsMax: .4f}")
        print(f"maximal number of iterations (nb_iter): {self.nb_iter:d}")
        print(f"line search flag: {self.gs: b}")
        print(f"gradient in Z0 (flag): {self.gradInZ0: b}\n")

        print(f"maximal shooting time: {self.Tmax: d}")
        print(f"time discretization for parallel transport (images): {self.parallelTimeDisc:d}")
        print(f"time discretization for velocity matching and metamorphosis: {self.time_disc:d}\n")

        print(f"use vector momentum (flag for covMatrix): {self.useVectorMomentum: b}")

