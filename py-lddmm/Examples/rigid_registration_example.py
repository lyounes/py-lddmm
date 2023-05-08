import numpy as np
import scipy as sp
from base.affineRegistration import rigidRegistration_varifold
import matplotlib.pyplot as plt
from base import loggingUtils

loggingUtils.setup_default_logging('../Output', fileName='info', stdOutput=True)

dim = 3
N = 250
exact = False
S = np.eye(dim)
S[0,0] = -1
b = np.zeros(dim)
b[0] = 2
# S = np.array([[-1, 0], [0,1]])
# b = np.array([2,0])

# S = np.eye(2)
# b = np.zeros(2)
theta = np.pi/4
if dim == 2:
    V = np.array([[0.25, -0.25], [-0.25,1]])
    m1 = np.array([-2,0])
    m2 = np.array([-3,-5])
    T0 = np.array([1.5, -.5])
    R0 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
else:
    V = np.array([[0.25, -0.25, 0], [-0.25, 1, 0], [0,0,1]])
    m1 = np.array([-2, 0, 0])
    m2 = np.array([-1, -1, 0])
    T0 = np.array([1.5, -.5, 1])
    R0 = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],
                   [0,0,1]])
SVS = S@V@S
x1 = sp.stats.multivariate_normal.rvs(m1, V, size = N)
x2 = sp.stats.multivariate_normal.rvs(m2, V, size = N)
x = np.concatenate((x1, x2), axis = 0)
if exact:
    ys = x@S.T + b
    y = np.concatenate((x, ys), axis=0)
else:
    y1 = sp.stats.multivariate_normal.rvs(m1, V, size = N)
    y2 = sp.stats.multivariate_normal.rvs(m2, V, size = N)
    ys1 = sp.stats.multivariate_normal.rvs(S@m1 + b, SVS, size=N)
    ys2 = sp.stats.multivariate_normal.rvs(S@m2+b, SVS, size=N)
    y = np.concatenate((y1, y2, ys1, ys2), axis = 0)

x = x @ R0.T + T0

R, T = rigidRegistration_varifold(x, y, symmetry=[S, b], ninit=8)

if dim == 2:
    plt.figure(1)
    plt.scatter(x[:,0], x[:,1])
    plt.scatter(y[:,0], y[:,1])
    Rx = x@R.T + T
    plt.figure(2)
    plt.scatter(Rx[:,0], Rx[:,1])
    plt.scatter(y[:,0], y[:,1])
    plt.show()
