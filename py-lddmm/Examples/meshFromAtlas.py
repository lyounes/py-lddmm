from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from qpsolvers import solve_qp
import pygalmesh
from base.curveExamples import Circle
from base.surfaceExamples import Sphere
import logging
from base import loggingUtils
from base.meshes import Mesh
from base.kernelFunctions import Kernel
from base.meshMatching import MeshMatching
from base.secondOrderMeshMatching import SecondOrderMeshMatching
from base.imageMatchingLDDMM import ImageMatching
from base.meshExamples import TwoBalls, TwoDiscs, MoGCircle, TwoEllipses
# import pykeops
# pykeops.clean_pykeops()
plt.ion()


def qp_atlas(f0, f1, kernel, solver='cvxopt', truth = None):
    nlabel = f0.image.shape[1]
    ntype = f1.image.shape[1]
    alpha = np.mean(f1.weights)
    print(alpha)
    alpha = 1
    #for i0 in range(nlabel):
    a0 = (f0.weights * f0.volumes)[:, None] * f0.image
        #for i1 in range(i0, nlabel):
    # a1 = f0.volumes[:, None] * f0.image
    P = a0.T @ kernel.applyK(f0.centers, a0)
    # for i0 in range(nlabel):
    #     for i1 in range(i0):
    #         P[i0, i1] = P[i1, i0]
    P = np.kron(P, np.eye(ntype))

    A = np.zeros((nlabel, nlabel*ntype))
    # for i0 in range(nlabel):
    #     a0 = alpha * f0.volumes * f0.image[:,i0]
    #     for f in range(ntypes):
    a1 = (f1.weights * f1.volumes)[:, None] * f1.image
    q = -np.ravel(a0.T @ kernel.applyK(f1.centers, a1, firstVar=f0.centers))
    if truth is not None:
        r = P@truth + q
        print('error', np.fabs(r).max())
    for i0 in range(nlabel):
        for f in range(ntype):
            A[i0, i0*ntype + f] = 1

    b = np.ones(nlabel)
    lb = np.zeros(nlabel*ntype)

    res = solve_qp(P, q, A=A, b=b, lb=lb, solver=solver)

    return res


loggingUtils.setup_default_logging('../Output', stdOutput = True)


fv0 = TwoEllipses(Boundary_a=14, Boundary_b=6, smallRadius=0.33)
fv1 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.4, translation=[0.25, -0.1])
sigmaDist = 1.
sigmaKernel = 1.
nlabel = fv0.image.shape[1]
ntypes = 5
s = 100000

transProb = np.random.uniform(0,1, size=(nlabel, ntypes))
transProb *= s / transProb.sum(axis=1)[:, None]
image1 = np.zeros((fv1.faces.shape[0], ntypes))
w = np.zeros(fv1.faces.shape[0])
for k in range(fv1.faces.shape[0]):
    image1[k, :] = stats.dirichlet.rvs(fv1.image[k, :] @ transProb)
    w[k] = stats.gamma.rvs(s*fv1.volumes[k]) / (s*fv1.volumes[k])

f1 = Mesh(mesh=fv1)
f1.updateImage(image1)
f1.updateWeights(w)

f1.saveVTK('../Output/randomMesh.vtk')

K1 = Kernel(name='laplacian', sigma=sigmaKernel)
options = {
    'outputDir': '../Output/meshMatchingTest/Atlas',
    'mode': 'normal',
    'maxIter': 100,
    'affine': 'none',
    'rotWeight': 100,
    'transWeight': 10,
    'scaleWeight': 1.,
    'affineWeight': 100.,
    'KparDiff': K1,
    'KparDist': ('gauss', sigmaDist),
    'KparIm': ('gauss', .1),
    'sigmaError': 0.5,
    'errorType': 'measure',
    'algorithm': 'bfgs',
    'internalCost': None,
    'internalWeight': 0,
    'regWeight': 1.,
    'pk_dtype': 'float64'
}

f11 = Mesh(mesh=fv1)
f11.updateImage(fv1.image @ (transProb/s))
res = qp_atlas(fv1, f11, K1, truth = np.ravel(transProb/s))
res = np.reshape(res, (nlabel, ntypes))

print('estimated', res)
print('true', transProb/s)
fv = Mesh(mesh=fv1)
image0 = fv.image @ res
fv.updateImage(image0)
fv.saveVTK('../Output/reconstructedMesh.vtk')

alpha = np.mean(f1.weights)
print(f'alpha= {alpha:.4f}')
fv = Mesh(mesh=fv0)
for iter in range(10):
    res = qp_atlas(fv, f1, K1)
    res = np.reshape(res, (nlabel, ntypes))
    print(res)
    #res = transProb / s
    f0 = Mesh(mesh=fv)
    image0 = fv.image @ res
    f0.updateImage(image0)
    f0.updateWeights(f0.weights*alpha)
    print(f0.vertices.shape, f0.faces.shape, f0.image.shape)
    mm = MeshMatching(Template=f0, Target=f1, options=options)
    mm.optimizeMatching()
    fv.updateVertices(f0.vertices)
