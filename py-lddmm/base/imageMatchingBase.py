import os
import numpy as np
import logging
import glob
from copy import deepcopy
from .basicMatching import BasicMatching 
from .gridscalars import GridScalars, saveImage
from .diffeo import DiffeoParam, Diffeomorphism, Kernel
from skimage.transform import resize as imresize, AffineTransform, EuclideanTransform, warp
from .affineBasis import AffineBasis
from scipy.ndimage.interpolation import affine_transform
from scipy.optimize import minimize
from scipy.linalg import expm

from functools import partial
import matplotlib
#matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class ImageMatchingParam():
#     def __init__(self, ):
#         self.timeStep = timeStep
#         self.sigmaKernel = sigmaKernel
#         self.typeKernel = typeKernel
#         self.kernelNormalization = 1.
#         self.maskMargin = 2
#         self.dim = dim
#         if np.isscalar(resol):
#             self.resol = (resol,)*self.dim
#         else:
#             self.resol = resol
#         self.epsMax = 100
#         if KparDiff is None:
#             self.KparDiff = Kernel(name = self.typeKernel, sigma = self.sigmaKernel, order=order, size=kernelSize,
#                                    dim=dim)
#         else:
#             self.KparDiff = KparDiff
#         # self.diffeoPar = DiffeoParam(dim, timeStep, KparDiff, sigmaKernel, order, kernelSize, typeKernel, resol)
#         self.sigmaError = sigmaError
#         self.algorithm = algorithm
#         self.wolfe = Wolfe
#         if sigmaSmooth > 0:
#             self.smoothKernel = Kernel(name = 'gauss', sigma = sigmaSmooth, size=25, dim=dim)
#             self.smoothKernel.K /= self.smoothKernel.K.sum()
#         else:
#             self.smoothKernel = None
#         self.rescaleFactor = rescaleFactor
#         self.padWidth = padWidth
#         self.metaDirection = metaDirection
#         self.affineAlign = affineAlign


class ImageMatchingBase(BasicMatching, Diffeomorphism):
    def __init__(self, Template=None, Target=None, options = None):
        if options is None:
            options = dict()
        if 'dim' not in options.keys():
            logging.error('Options must include a dimension; pursuing with dim=2 -- no guarantees')
            options['dim'] = 2
        self.dim = options['dim']
        super().__init__(Template, Target, options)
        # super(Diffeomorphism, self).__init__(self.im0.data.shape, options)
        # super().__init__()
        #          regWeight = 1.0, verb=True,
        #          testGradient=True, saveFile = 'evolution',
        #          outputDir = '.',pplot=True):

        # self.set_template_and_target(Template, Target, misc = {'affineAlign': self.options['affineAlign']})#, subsampleTargetSize)
        if np.isscalar(self.options['resol']):
            self.options['resol'] = (self.options['resol'],) * self.dim

        self.init(self.im0.data.shape, self.options)
        self.shape = self.im0.data.shape
        self.saveMovie = self.options['saveMovie']
        self.pplot = self.options['pplot']
        if self.pplot:
            self.initial_plot()

        self.initial_save()

        # self.rescaleFactor = rescaleFactor
        # self.padWidth = padWidth
        # self.metaDirection = metaDirection
        # self.affineAlign = affineAlign

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['dim'] = 3
        options['order'] = -1,
        options['kernelSize'] = 50
        options['sigmaKernel'] = 1.
        options['orderKernel'] = 10
        options['typeKernel'] = 'gauss'
        options['resol'] = 1.
        options['rescaleFactor'] = 1.
        options['padWidth'] = 0
        options['metaDirection'] = 'FWD'
        options['sigmaSmooth'] = 0.01
        options['affineAlign'] = None
        options['saveMovie'] = True
        options['epsMax'] = 100
        options['normalize'] = None
        return options


    def setDotProduct(self, unreduced=False):
        if self.options['algorithm'] == 'cg':
             self.euclideanGradient = False
             self.dotProduct = self.dotProduct_Riemannian
        else:
            self.euclideanGradient = True
            self.dotProduct = self.dotProduct_euclidean

    def set_parameters(self):
        super().set_parameters()
        # self.saveRate = 10
        # self.gradEps = -1
        # self.randomInit = False
        # self.iter = 0
        # self.kernelNormalization = 1.
        # self.maskMargin = 2
        # self.epsMax = 100
        # if self.options['KparDiff'] is None:
        #     self.KparDiff = Kernel(name = self.options['typeKernel'],
        #                            sigma = self.options['sigmaKernel'],
        #                            order=self.options['orderKernel'],
        #                            size=self.options['kernelSize'],
        #                            dim=self.dim)
        # else:
        #     self.KparDiff = self.options['KparDiff']
        #     # self.diffeoPar = DiffeoParam(dim, timeStep, KparDiff, sigmaKernel, order, kernelSize, typeKernel, resol)
        #
        # self.resol = self.options['resol']

        self.originalTemplate = deepcopy(self.im0)
        self.originalTarget = deepcopy(self.im1)
        if self.options['sigmaSmooth'] > 0:
            self.smoothKernel = Kernel(name = 'gauss', sigma = self.options['sigmaSmooth'], size=25, dim=self.options['dim'])
            self.smoothKernel.K /= self.smoothKernel.K.sum()
            self.im0.data = self.smoothKernel.ApplyToImage(self.im0.data)
            self.im1.data = self.smoothKernel.ApplyToImage(self.im1.data)
        else:
            self.smoothKernel = None

    def AffineRegister(self, affineAlign = None, tolerance = None):
        if tolerance is None:
            tolerance = self.options['padWidth']
        Afb = AffineBasis(dim=self.options['dim'], affine=affineAlign)
        #padWidth = max(np.max(self.im0.data.shape), np.max(self.im1.data.shape))//2
        # start = (padWidth,) * self.dim
        # end = tuple(np.array(start, dtype=int) + np.array(self.im1.data.shape, dtype=int))
        # slices = tuple(map(slice, start, end))

        #im0 = np.pad(self.im0.data, padWidth, mode='constant', constant_values=1e10)
        im1 = self.im1.data
        # im1 = np.pad(self.im1.data, padWidth, mode='constant', constant_values=1e10)
        bounds = [[0], [self.im0.data.shape[0]-1]]
        for i in range(1, self.dim):
            newbounds = []
            for b in bounds:
                newbounds += [b + [0]]
                newbounds += [b + [self.im0.data.shape[i]-1]]
            bounds = newbounds
        bounds = np.array(bounds).T

        def enerAff(gamma):
            U = np.zeros((self.dim + 1, self.dim + 1))
            AB = Afb.basis.dot(gamma)
            U[:self.dim, :self.dim] = AB[:self.dim ** 2].reshape((self.dim, self.dim))
            U[:self.dim, self.dim] = AB[self.dim ** 2:]
            U = expm(U)
            A = U[:self.dim, :self.dim]
            b = U[:self.dim, self.dim]
            newbounds = np.dot(A, bounds) + b[:, None] - bounds
            if np.sqrt(np.sum(newbounds**2, axis=1).max()) > tolerance:
                return 1e50

            AI1 = affine_transform(im1, A, b, mode='nearest', order=1)
            res = ((self.im0.data - AI1) ** 2).sum()
            return res

        gamma = np.zeros(Afb.affineDim)
        opt = minimize(enerAff, gamma, method='Powell')
        gamma = opt.x
        U = np.zeros((self.dim + 1, self.dim + 1))
        AB = Afb.basis.dot(gamma)
        U[:self.dim, :self.dim] = AB[:self.dim ** 2].reshape((self.dim, self.dim))
        U[:self.dim, self.dim] = AB[self.dim ** 2:]
        U = expm(U)
        A = U[:self.dim, :self.dim]
        b = U[:self.dim, self.dim]
        self.im1.data =  affine_transform(im1, A, b, order=1, mode='nearest')
        # self.im1.data = im1[slices]

    def set_template_and_target(self, Template, Target, misc = None): #subsampleTargetSize=-1, affineAlign=None):
        affineAlign = self.options['affineAlign']

        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.im0 = GridScalars(grid=Template, dim=self.options['dim'])
            if self.options['padWidth'] > 0:
                self.im0.data = np.pad(self.im0.data, self.options['padWidth'], mode='edge')
            self.im0.data = imresize(self.im0.data,
                                     np.floor(np.array(self.im0.data.shape)*self.options['rescaleFactor']).astype(int))

        if Target is None:
            logging.error('Please provide a target surface')
            return
        else:
            self.im1 = GridScalars(grid=Target, dim=self.options['dim'])
            if self.options['padWidth'] > 0:
                self.im1.data = np.pad(self.im1.data, self.options['padWidth'], mode='edge')
            self.im1.data = imresize(self.im1.data, self.im0.data.shape)
            if affineAlign is not None:
                self.AffineRegister(affineAlign=affineAlign)


        if self.options['normalize'] is not None:
            m = min(self.im0.data.min(), self.im1.data.min())
            M = max(max(self.im0.data.max(), self.im1.data.max())-m, 1e-10)
            self.im0.data = self.options['normalize'] * (self.im0.data -m) /(M)
            self.im1.data = self.options['normalize'] * (self.im1.data -m) /(M)

                # tform3 = AffineTransform()
                # hog0 = hogFeature(self.im0.data)
                # hog1 = hogFeature(self.im1.data)
                # src = corner_peaks(corner_harris(self.im0.data), threshold_rel=0.001)
                # dest = corner_peaks(corner_harris(self.im1.data), threshold_rel=0.001)
                # plt.figure(1)
                # plt.imshow(self.im0.data, cmap='gray')
                # plt.scatter(src[:,1], src[:,0])
                # plt.figure(2)
                # plt.imshow(self.im1.data, cmap='gray')
                # plt.scatter(dest[:,1], dest[:,0])
                # plt.gcf().canvas.flush_events()
                # plt.show(block=False)
                # plt.show(block=False)
                # tform3.estimate(src, dest)
                # self.im1.data = warp(self.im1.data, tform3, output_shape=self.im0.data.shape)


        # if (subsampleTargetSize > 0):
        #     self.fvInit.Simplify(subsampleTargetSize)
        #     logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))
        # self.im0.saveVTK(self.outputDir+'/Template.vtk')
        # self.im1.saveVTK(self.outputDir+'/Target.vtk')


#    def initialize_variables(self):


    def initial_plot(self):
        pass
        # fig = plt.figure(3)
        # ax = Axes3D(fig)
        # lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r')
        # if self.fv1:
        #     lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
        # else:
        #     lim0 = lim1
        # ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
        # ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
        # ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
        # fig.canvas.flush_events()

    def initial_save(self):
        if len(self.im0.data.shape) == 3:
            ext = '.vtk'
        else:
            ext = ''
        saveImage(self.originalTemplate.data, self.outputDir + '/OriginalTemplate'+ ext)
        saveImage(self.originalTarget.data, self.outputDir + '/OriginalTarget'+ ext)
        saveImage(self.im0.data, self.outputDir + '/Template'+ ext)
        saveImage(self.im1.data, self.outputDir + '/Target' + ext)
        saveImage(self.KparDiff.K, self.outputDir + '/Kernel' + ext, normalize=True)
        if self.smoothKernel:
            saveImage(self.smoothKernel.K, self.outputDir + '/smoothKernel' + ext, normalize=True)
        saveImage(self.mask.min(axis=0), self.outputDir + '/Mask' + ext, normalize=True)




