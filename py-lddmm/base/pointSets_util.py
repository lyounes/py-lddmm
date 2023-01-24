import numpy as np
from numba import jit, int64

@jit(nopython=True)
def det2D(x1, x2):
    return x1[:,0]*x2[:,1] - x2[:,0]*x1[:,1]

@jit(nopython=True)
def rot90(x):
    y = np.zeros(x.shape)
    y[:,0] = -x[:,1]
    y[:, 1] = x[:,0]
    return y

@jit(nopython=True)
def det3D(x1, x2, x3):
    return (x1 * np.cross(x2, x3)).sum(axis = 1)

