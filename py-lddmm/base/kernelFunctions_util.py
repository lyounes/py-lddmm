from numba import jit, prange, int64
import numpy as np
from math import pi
from pykeops.numpy import Genred
import pykeops

KP = -1

c_ = np.array([[1,0,0,0,0],
               [1,1,0,0,0],
               [1,1,1/3,0,0],
               [1,1,0.4,1/15,0],
               [1,1,3/7,2/21,1/105]])

c1_ = np.array([[0,0,0,0],
                [1,0,0,0],
                [1/3,1/3,0,0],
                [1/5, 1/5, 1/15, 0],
                [1/7,1/7,2/35,1/105]])

c2_ = np.array([[0,0,0],
               [0,0,0],
               [1/3,0,0],
               [1/15,1/15,0],
               [1/35,1/35,1/105]])


# Polynomial factor for Laplacian kernel
@jit(nopython=True)
def lapPol(u, ord):
    u2 = u*u
    return c_[ord,0] + c_[ord,1] * u + c_[ord,2] * u2 + c_[ord,3] * u*u2 + c_[ord,4] * u2*u2

@jit(nopython=True)
def lapPolDiff(u, ord):
    u2 = u*u
    return c1_[ord,0] + c1_[ord,1] * u + c1_[ord,2] * u2 + c1_[ord,3] * u*u2

@jit(nopython=True)
def lapPolDiff2(u, ord):
    return c2_[ord,0] + c2_[ord,1] * u + c2_[ord,2] * u*u

@jit(nopython=True)
def atanK(u):
    return np.arctan(u) + pi/2

@jit(nopython=True)
def atanKDiff(u):
    return 1/(1 + u**2)

@jit(nopython=True)
def logcoshK(u):
    v = np.fabs(u)
    return u + v + np.log1p(np.exp(-2*v))

@jit(nopython=True)
def logcoshKDiff(u):
    return 1 + np.tanh(u)

@jit(nopython=True)
def ReLUK(u):
    return (1 + np.sign(u))*u/2

@jit(nopython=True)
def ReLUKDiff(u):
    return heaviside(u)

@jit(nopython=True)
def heaviside(u):
    return (np.sign(u - 1e-8) + np.sign(u + 1e-8) + 2) / 4



@jit(nopython=True, parallel=True)
def kernelmatrix(y, x, name, scale, ord):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    f = np.zeros((num_nodes_y, num_nodes))

    wsig = 0
    for s in scale:
        wsig +=  s**KP

    if 'gauss' in name:
        for k in prange(num_nodes_y):
            for l in prange(num_nodes):
                ut0 = np.sum((y[k,:] - x[l,:])**2)
                Kh = 0
                for s in scale:
                    ut = ut0 / s*s
                    if ut < 1e-8:
                        Kh += s**KP
                    else:
                        Kh += np.exp(-0.5*ut) * s**KP
                Kh /= wsig
                f[k,l] = Kh
    elif 'lap' in name:
        for k in prange(num_nodes_y):
            for l in prange(num_nodes):
                ut0 = np.sqrt(np.sum((y[k, :] - x[l, :]) ** 2))
                Kh = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh += s ** KP
                    else:
                        lpt = lapPol(ut, ord)
                        Kh += lpt * np.exp(-ut) * s ** KP
                Kh /= wsig
                f[k,l] = Kh
    return f

def applyK(y, x, a, name, scale, order, cpu=False):
    if not cpu and pykeops.config.gpu_available:
        return applyK_pykeops(y, x, a, name, scale, order)
    else:
        return applyK_numba(y, x, a, name, scale, order)


@jit(nopython=True, parallel=True)
def applyK_numba(y, x, a, name, scale, order):
    res = np.zeros((y.shape[0], a.shape[1]))
    ns = len(scale)
    sKP = scale**KP
    wsig = sKP.sum()
    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if name == 'min':
            for k in prange(y.shape[0]):
                for l in range(x.shape[0]):
                    u = np.minimum(ys[k, :], xs[l, :])
                    #res[k,:] += logcoshK(u)*a[l,:]
                    res[k,:] += ReLUK(u)*a[l,:] *sKP[s]
        elif 'gauss' in name:
            for k in prange(y.shape[0]):
                resk = np.zeros(a.shape[1])
                for l in range(x.shape[0]):
                    u = ((ys[k, :] - xs[l, :]) ** 2).sum() / 2
                    resk += np.exp(- u) * a[l,:] * sKP[s]
                res[k,:] += resk
        elif 'lap' in name:
            for k in prange(y.shape[0]):
                resk = np.zeros(a.shape[1])
                for l in range(x.shape[0]):
                    u = np.sqrt(((ys[k,:] - xs[l,:]) ** 2).sum())
                    u1 = lapPol(u, order)
                    u1 *= np.exp(-u)
                    u1 *= sKP[s]
                    resk += u1 *a[l,:]
                res[k,:] += resk
                    #print('a', a[l,:])
    res /= wsig
    return res

def applyK_pykeops(y, x, a, name, scale, order):
    res = np.zeros(y.shape)
    ns = len(scale)
    sKP = scale**KP
    wsig = sKP.sum()
    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if name == 'min':
            dtype = "float64"
            D = xs.shape[1]
            Dv = a.shape[1]
            sKPs = np.array([sKP[s]])
            formula_min = "ReLU(Min(Concat(ys,xs))) * a * sKPs"
            variables_min = ["ys = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
                             "xs = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
                             "a = Vj(" + str(Dv) + ")",  # Third arg:  j-variable of size Dv
                             "sKPs = Pm(1)"
                             ]  # Fourth arg: scalar parameter
            my_routine_min = Genred(formula_min, variables_min, reduction_op="Sum", dtype=dtype, dtype_acc=dtype, axis=1)
            res = my_routine_min(ys.astype(dtype), xs.astype(dtype), a.astype(dtype), sKPs.astype(dtype))
        elif 'gauss' in name:
            dtype = "float64"
            D = xs.shape[1]
            Dv = a.shape[1]
            g = np.array([0.5])  # Parameter of the Gaussian RBF kernel
            sKPs = np.array([sKP[s]])
            formula_gauss = "Exp(-g * SqDist(ys,xs)) * a * sKPs"
            variables_gauss = ["g = Pm(1)",   # First arg: scalar parameter
                               "ys = Vi(" + str(D) + ")",  # Second arg:  i-variable of size D
                               "xs = Vj(" + str(D) + ")",  # Third arg: j-variable of size D
                               "a = Vj(" + str(Dv) + ")",  # Fourth arg:  j-variable of size Dv
                               "sKPs = Pm(1)"
                               ]  # Fifth arg: scalar parameter
            my_routine_gauss = Genred(formula_gauss, variables_gauss, reduction_op="Sum",dtype=dtype,dtype_acc=dtype,axis=1)
            res = my_routine_gauss(g.astype(dtype), ys.astype(dtype), xs.astype(dtype), a.astype(dtype), sKPs.astype(dtype))
        elif 'lap' in name:
            dtype = "float64"
            D = xs.shape[1]
            Dv = a.shape[1]
            sKPs = np.array([sKP[s]])
            formula_lap = "(c_0 + c_1 * Norm2(ys-xs) + c_2 * Square(Norm2(ys-xs)) + c_3 * Norm2(ys-xs)*Square(Norm2(ys-xs)) + c_4 * Square(Norm2(ys-xs))*Square(Norm2(ys-xs))) * Exp(-Norm2(ys-xs)) * a * sKPs"
            variables_lap = ["c_0 = Pm(1)",  # First arg: scalar parameter
                             "c_1 = Pm(1)",  # Second arg: scalar parameter
                             "c_2 = Pm(1)",  # Third arg: scalar parameter
                             "c_3 = Pm(1)",  # Fourth arg: scalar parameter
                             "c_4 = Pm(1)",  # Fifth arg: scalar parameter
                             "ys = Vi(" + str(D) + ")",  # Sixth arg:  i-variable of size D
                             "xs = Vj(" + str(D) + ")",  # Seventh arg: j-variable of size D
                             "a = Vj(" + str(Dv) + ")",  # Eighth arg:  j-variable of size Dv
                             "sKPs = Pm(1)"
                             ]  # Ninth arg: scalar parameter
            my_routine_lap = Genred(formula_lap, variables_lap, reduction_op="Sum", dtype=dtype, dtype_acc=dtype, axis=1)
            res = my_routine_lap(np.array([c_[order, 0]]).astype(dtype), np.array([c_[order, 1]]).astype(dtype),
                                 np.array([c_[order, 2]]).astype(dtype), np.array([c_[order, 3]]).astype(dtype),
                                 np.array([c_[order, 4]]).astype(dtype),
                                 ys.astype(dtype), xs.astype(dtype), a.astype(dtype), sKPs.astype(dtype))
    res /= wsig
    return res

def applyDiffKT(y, x, p, a, name, scale, order, regweight=1., lddmm=False, cpu=False):
    if not cpu and pykeops.config.gpu_available:
        return applyDiffKT_pykeops(y, x, p, a, name, scale, order, regweight=regweight, lddmm=lddmm)
    else:
        return applyDiffKT_numba(y, x, p, a, name, scale, order, regweight=regweight, lddmm=lddmm)

@jit(nopython=True, parallel=True)
def applyDiffKT_numba(y, x, p, a, name, scale, order, regweight=1., lddmm=False):
    res = np.zeros(y.shape)
    ns = len(scale)
    sKP1 = scale**(KP-1)
    sKP = scale**(KP)
    wsig = sKP.sum()
    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if name == 'min':
            for k in prange(y.shape[0]):
                for l in range(x.shape[0]):
                    if lddmm:
                        akl = p[k, :] * a[l, :] + a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :]
                    else:
                        akl = p[k, :] * a[l, :]
                    u = np.minimum(ys[k,:],xs[l,:])
                    #res[k, :] += (heaviside(x[k,:]-y[l,:])*a1[k,:]*a2[l,:])*logcoshKDiff(u)/s
                    res[k, :] += (heaviside(xs[l,:]-ys[k,:])*akl)*ReLUKDiff(u)*sKP1[s]
        elif 'gauss' in name:
            for k in prange(y.shape[0]):
                for l in range(x.shape[0]):
                    if lddmm:
                        akl = p[k, :] * a[l, :] + a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :]
                    else:
                        akl = p[k, :] * a[l, :]
                    u = ((ys[k,:]-xs[l,:])**2).sum()/2
                    res[k, :] += (ys[k,:]-xs[l,:]) * (-np.exp(- u) * akl.sum())*sKP1[s]
        elif 'lap' in name:
            for k in prange(y.shape[0]):
                for l in range(x.shape[0]):
                    if lddmm:
                        akl = p[k, :] * a[l, :] + a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :]
                    else:
                        akl = p[k, :] * a[l, :]
                    u = np.sqrt(((ys[k,:] - xs[l,:]) ** 2).sum())
                    res[k, :] += (ys[k,:]-xs[l,:]) * (-lapPolDiff(u, order) * np.exp(- u) *
                                                    akl.sum())*sKP1[s]
    res /= wsig
    return res

def applyDiffKT_pykeops(y, x, p, a, name, scale, order, regweight=1., lddmm=False):
    res = np.zeros(y.shape)
    ns = len(scale)
    sKP1 = scale**(KP-1)
    sKP = scale**(KP)
    wsig = sKP.sum()
    D = x.shape[1]
    Da = a.shape[1]
    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if name == 'min':
            dtype = "float64"
            sKP1s = np.array([sKP1[s]])
            if lddmm:
                h = np.array([2. * regweight])
                formula2_min = "(Step(xs-ys) * Sum(p_i * a_j + a_i * p_j - h * a_i * a_j)) * Step(Min(Concat(ys,xs))) * sKP1s"
                variables2_min = ["ys = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
                                  "xs = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
                                  "a_j = Vj(" + str(Da) + ")",  # Third arg:  j-variable of size D
                                  "a_i = Vi(" + str(Da) + ")",  # Fourth arg:  i-variable of size D
                                  "p_j = Vj(" + str(Da) + ")",  # Fifth arg: j-variable of size D
                                  "p_i = Vi(" + str(Da) + ")",  # Sixth arg: i-variable of size D
                                  "h = Pm(1)",
                                  "sKP1s = Pm(1)",
                                  ]  # Seventh and eighth args: scalar parameters
                my_routine2_min = Genred(formula2_min, variables2_min, reduction_op="Sum", dtype=dtype, dtype_acc=dtype, axis=1)
                res = my_routine2_min(ys.astype(dtype), xs.astype(dtype), a.astype(dtype), a.astype(dtype),
                                      p.astype(dtype), p.astype(dtype), h.astype(dtype), sKP1s.astype(dtype))
            else:
                formula2_min = "(Step(xs-ys) * Sum(p_i * a_j)) * Step(Min(Concat(ys,xs))) * sKP1s"
                variables2_min = ["ys = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
                                  "xs = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
                                  "a_j = Vj(" + str(Da) + ")",  # Third arg:  j-variable of size D
                                  "p_i = Vi(" + str(Da) + ")",  # Fourth arg: i-variable of size D
                                  "sKP1s = Pm(1)",
                                  ]  # Fifth arg: scalar parameter
                my_routine2_min = Genred(formula2_min, variables2_min, reduction_op="Sum", dtype=dtype, dtype_acc=dtype, axis=1)
                res = my_routine2_min(ys.astype(dtype), xs.astype(dtype), a.astype(dtype), p.astype(dtype), sKP1s.astype(dtype))
        elif 'gauss' in name:
            dtype = "float64"
            g = np.array([0.5])   # Parameter of the Gaussian RBF kernel
            sKP1s = np.array([sKP1[s]])
            if lddmm:
                h = np.array([2. * regweight])
                formula2_gauss = "(ys-xs) * (-Exp(-g * SqDist(ys,xs)) * Sum(p_i * a_j + a_i * p_j - h * a_i * a_j)) * sKP1s"
                # formula2_gauss = "(ys-xs) * (-Exp(-g * SqDist(ys,xs)) * ((p_i | a_j)+(a_i | p_j) - h * (a_i | a_j))) * sKP1s"
                variables2_gauss = ["ys = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
                                    "xs = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
                                    "a_j = Vj(" + str(Da) + ")",  # Third arg:  j-variable of size D
                                    "a_i = Vi(" + str(Da) + ")",  # Fourth arg:  i-variable of size D
                                    "p_j = Vj(" + str(Da) + ")",  # Fifth arg: j-variable of size D
                                    "p_i = Vi(" + str(Da) + ")",  # Sixth arg: i-variable of size D
                                    "g = Pm(1)",
                                    "h = Pm(1)",
                                    "sKP1s = Pm(1)",
                                    ]  # Seventh, eighth, and ninth args: scalar parameters
                my_routine2_gauss = Genred(formula2_gauss, variables2_gauss, reduction_op="Sum",dtype=dtype,dtype_acc=dtype,axis=1)
                res = my_routine2_gauss(ys.astype(dtype), xs.astype(dtype), a.astype(dtype), a.astype(dtype),
                                        p.astype(dtype), p.astype(dtype), g.astype(dtype), h.astype(dtype),
                                        sKP1s.astype(dtype))
            else:
                formula2_gauss = "(ys-xs) * (-Exp(-g * SqDist(ys,xs)) * Sum(p_i * a_j)) * sKP1s"
                variables2_gauss = ["ys = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
                                    "xs = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
                                    "a_j = Vj(" + str(Da) + ")",  # Third arg:  j-variable of size D
                                    "p_i = Vi(" + str(Da) + ")",  # Fourth arg: i-variable of size D
                                    "g = Pm(1)",
                                    "sKP1s = Pm(1)",
                                    ]  # Fifth and sixth args: scalar parameters
                my_routine2_gauss = Genred(formula2_gauss, variables2_gauss, reduction_op="Sum", dtype=dtype, dtype_acc=dtype, axis=1)
                res = my_routine2_gauss(ys.astype(dtype), xs.astype(dtype), a.astype(dtype),
                                        p.astype(dtype), g.astype(dtype), sKP1s.astype(dtype))
        elif 'lap' in name:
            dtype = "float64"
            sKP1s = np.array([sKP1[s]])
            if lddmm:
                h = np.array([2. * regweight])
                formula2_lap = "(ys-xs) * (-(c_0 + c_1 * Norm2(ys-xs) + c_2 * Square(Norm2(ys-xs)) + c_3 * Norm2(ys-xs)*Square(Norm2(ys-xs))) * Exp(-Norm2(ys-xs)) * Sum(p_i * a_j + a_i * p_j - h * a_i * a_j)) * sKP1s"
                variables2_lap = ["c_0 = Pm(1)",  # First arg: scalar parameter
                                  "c_1 = Pm(1)",  # Second arg: scalar parameter
                                  "c_2 = Pm(1)",  # Third arg: scalar parameter
                                  "c_3 = Pm(1)",  # Fourth arg: scalar parameter
                                  "ys = Vi(" + str(D) + ")",  # Fifth arg:  i-variable of size D
                                  "xs = Vj(" + str(D) + ")",  # Sixth arg: j-variable of size D
                                  "a_j = Vj(" + str(Da) + ")",  # Seventh arg:  j-variable of size D
                                  "a_i = Vi(" + str(Da) + ")",  # Eighth arg:  i-variable of size D
                                  "p_j = Vj(" + str(Da) + ")",  # Ninth arg: j-variable of size D
                                  "p_i = Vi(" + str(Da) + ")",  # Tenth arg: i-variable of size D
                                  "h = Pm(1)",
                                  "sKP1s = Pm(1)",
                                  ]  # Eleventh and twelfth args: scalar parameters
                my_routine2_lap = Genred(formula2_lap, variables2_lap, reduction_op="Sum", dtype=dtype, dtype_acc=dtype, axis=1)
                res = my_routine2_lap(np.array([c1_[order, 0]]).astype(dtype), np.array([c1_[order, 1]]).astype(dtype),
                                      np.array([c1_[order, 2]]).astype(dtype), np.array([c1_[order, 3]]).astype(dtype),
                                      ys.astype(dtype), xs.astype(dtype), a.astype(dtype), a.astype(dtype),
                                      p.astype(dtype), p.astype(dtype), h.astype(dtype), sKP1s.astype(dtype))
            else:
                formula2_lap = "(ys-xs) * (-(c_0 + c_1 * Norm2(ys-xs) + c_2 * Square(Norm2(ys-xs)) + c_3 * Norm2(ys-xs)*Square(Norm2(ys-xs))) * Exp(-Norm2(ys-xs)) * Sum(p_i * a_j)) * sKP1s"
                variables2_lap = ["c_0 = Pm(1)",  # First arg: scalar parameter
                                  "c_1 = Pm(1)",  # Second arg: scalar parameter
                                  "c_2 = Pm(1)",  # Third arg: scalar parameter
                                  "c_3 = Pm(1)",  # Fourth arg: scalar parameter
                                  "ys = Vi(" + str(D) + ")",  # Fifth arg:  i-variable of size D
                                  "xs = Vj(" + str(D) + ")",  # Sixth arg: j-variable of size D
                                  "a_j = Vj(" + str(Da) + ")",  # Seventh arg:  j-variable of size D
                                  "p_i = Vi(" + str(Da) + ")",  # Eighth arg: i-variable of size D
                                  "sKP1s = Pm(1)",
                                  ]  # Ninth arg: scalar parameter
                my_routine2_lap = Genred(formula2_lap, variables2_lap, reduction_op="Sum", dtype=dtype, dtype_acc=dtype, axis=1)
                res = my_routine2_lap(np.array([c1_[order, 0]]).astype(dtype), np.array([c1_[order, 1]]).astype(dtype),
                                      np.array([c1_[order, 2]]).astype(dtype), np.array([c1_[order, 3]]).astype(dtype),
                                      ys.astype(dtype), xs.astype(dtype), a.astype(dtype), p.astype(dtype), sKP1s.astype(dtype))
    res /= wsig
    return res

@jit(nopython=True, parallel=True)
def applyDiv(y, x, a, name, scale, order):
    res = np.zeros((y.shape[0], 1))
    if name == 'min':
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                for s in scale:
                    u = np.minimum(y[k,:],x[l,:]) / s
                    #res[k, :] += (heaviside(x[k,:]-y[l,:])*a[l,:]*logcoshKDiff(u)/s).sum()
                    res[k, :] += (heaviside(x[k,:]-y[l,:])*a[l,:]*ReLUKDiff(u)/s).sum()
    elif 'gauss' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                for s in scale:
                    res[k, :] += ((y[k,:]-x[l,:])*a[l,:]).sum() * \
                                 (-np.exp(- ((y[k,:]-x[l,:])**2).sum()/(2*s**2)))/(s**2)
    elif 'lap' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                for s in scale:
                    u = np.sqrt(((y[k,:] - x[l,:]) ** 2).sum()) / s
                    res[k, :] += ((y[k,:]-x[l,:])*a[l,:]).sum() * (-lapPolDiff(u, order) * np.exp(- u) /(s**2))
    res /= len(scale)
    return res


def applylocalk(y, x, a, name, scale, order, neighbors, num_neighbors):
    if 'lap' in name:
        return applylocalk_lap(y,x,a,scale,order,neighbors,num_neighbors)
    elif 'gauss' in name:
        return applylocalk_gauss(y,x,a,scale,neighbors,num_neighbors)

@jit(nopython=True, parallel=True)
def applylocalk_gauss(y, x, a, scale, neighbors, num_neighbors):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP


    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1


    for k in prange(num_nodes_y):
        sqdist = np.zeros(dim)
        for l in range(num_nodes):
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk],endSet[kk]):
                    ut0 += (y[k,neighbors[jj]]-x[l,neighbors[jj]])**2
                Kv = 0
                for s in scale:
                    ut = ut0/s**2
                    if ut < 1e-8:
                        Kv += s**KP
                    else:
                        Kv += np.exp(-0.5*ut) * s**KP
                sqdist[kk] = Kv/wsig
            for j in range(dim):
                f[k,j] += sqdist[num_neighbors[j]] * a[l,j]
    return f

@jit(nopython=True, parallel=True)
def applylocalk_lap(y, x, a, scale, order, neighbors, num_neighbors):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP


    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1


    for k in prange(num_nodes_y):
        sqdist = np.zeros(dim)
        for l in range(num_nodes):
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk], endSet[kk]):
                    ut0 += (y[k, neighbors[jj]] - x[l, neighbors[jj]]) ** 2
                ut0 = np.sqrt(ut0)
                Kv = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kv += s ** KP
                    else:
                        lpt = lapPol(ut, order)
                        Kv += lpt * np.exp(-ut) * s ** KP
                sqdist[kk] = Kv / wsig
            for j in range(dim):
                f[k, j] += sqdist[num_neighbors[j]] * a[l, j]
    return f

@jit(nopython=True, parallel=True)
def applylocalk_naive(y, x, a, name, scale, order):
    f = np.zeros(y.shape)
    wsig = 0
    for s in scale:
        wsig += s**KP

    s = scale[0]
    ys = y / s
    xs = x / s
    if 'lap' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                ut = np.fabs(ys[k, :] - xs[l, :])
                f[k, :] += lapPol(ut, order) * np.exp(-ut) * a[l, :]
    elif 'gauss' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                ut = np.maximum(np.minimum(np.fabs(ys[k, :] - xs[l, :]), 10),-10)
                f[k,:] += np.exp(-ut*ut/2) * a[l,:]
    return f


def applylocalkdifft(y, x, p, a, name, scale, order, neighbors, num_neighbors, regweight=1., lddmm=False):
    if 'lap' in name:
        return applylocalkdifft_lap(y, x, p, a, scale, order, neighbors, num_neighbors, regweight, lddmm)
    elif 'gauss' in name:
        return applylocalkdifft_gauss(y, x, p, a, scale, neighbors, num_neighbors, regweight, lddmm)

@jit(nopython=True, parallel=True)
def applylocalkdifft_gauss(y, x, p, a, scale, neighbors, num_neighbors, regweight=1., lddmm=False):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP

    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1

    for k in prange(num_nodes_y):
        for l in range(num_nodes):
            sqdist = np.zeros(dim)
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk],endSet[kk]):
                    ut0 += (y[k,neighbors[jj]]-x[l,neighbors[jj]])**2
                Kv_diff = 0
                for s in scale:
                    ut = ut0/s**2
                    if ut < 1e-8:
                        Kv_diff -= s**(KP-2)
                    else:
                        Kv_diff -= np.exp(-0.5*ut)*s**(KP-2)
                sqdist[kk] = Kv_diff/wsig
            for j in range(dim):
                kk = num_neighbors[j]
                if lddmm:
                    akl = p[k, j] * a[l, j] + p[l, j] * a[k, j] - 2 * regweight * a[k, j] * a[l, j]
                else:
                    akl = p[k, j] * a[l,j]
                for jj in range(startSet[kk], endSet[kk]):
                    ii = neighbors[jj]
                    f[k,ii] += sqdist[kk] * (y[k,ii]-x[l,ii])* akl
    return f

@jit(nopython=True, parallel=True)
def applylocalkdifft_lap(y, x, p, a, scale, order, neighbors, num_neighbors, regweight=1., lddmm=False):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP

    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1

    for k in prange(num_nodes_y):
        sqdist = np.zeros(dim)
        for l in range(num_nodes):
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk],endSet[kk]):
                    ut0 += (y[k,neighbors[jj]]-x[l,neighbors[jj]])**2
                ut0 = np.sqrt(ut0)
                Kv_diff = 0
                for s in scale:
                    ut = ut0/s**2
                    if ut < 1e-8:
                        Kv_diff -= s**(KP-2)
                    else:
                        Kv_diff -= lapPolDiff(ut, order)*np.exp(-ut)*s**(KP-2)
                sqdist[kk] = Kv_diff/wsig
            for j in range(dim):
                kk = num_neighbors[j]
                if lddmm:
                    akl = p[k, j] * a[l, j] + p[l, j] * a[k, j] - 2 * regweight * a[k, j] * a[l, j]
                else:
                    akl = p[k, j] * a[l,j]
                for jj in range(startSet[kk], endSet[kk]):
                    ii = neighbors[jj]
                    f[k,ii] += sqdist[kk] * (y[k,ii]-x[l,ii])* akl
    return f

@jit(nopython=True, parallel=True)
def applylocalkdiv(x, y, a, name, scale, order, neighbors, num_neighbors):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes,1))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP

    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = 0
            sqdist = np.zeros(dim)
            for l in range(num_nodes_y):
                for kk in range(nSets):
                    ut0 = 0
                    for jj in range(startSet[kk],endSet[kk]):
                        ut0 += (x[k,neighbors[jj]]-y[l,neighbors[jj]])**2
                    Kv_diff = 0
                    for s in scale:
                        ut = ut0/s**2
                        if ut < 1e-8:
                            Kv_diff -= 0.5*s**(KP-2)
                        else:
                            Kv_diff -= 0.5*np.exp(-0.5*ut)*s**(KP-2)
                    sqdist[kk] = Kv_diff/wsig
                for j in range(dim):
                    kk = num_neighbors[j]
                    for jj in range(startSet[kk], endSet[kk]):
                        ii = neighbors[jj]
                        df += sqdist[kk] * 2*(x[k,ii]-y[l,ii])* a[l,ii]
            f[k] = df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = 0
            sqdist = np.zeros(dim)
            for l in range(num_nodes_y):
                for kk in range(nSets):
                    ut0 = 0
                    for jj in range(startSet[kk],endSet[kk]):
                        ut0 += (x[k,neighbors[jj]]-y[l,neighbors[jj]])**2
                    ut0 = np.sqrt(ut0)
                    Kv_diff = 0
                    for s in scale:
                        ut = ut0/s**2
                        if ut < 1e-8:
                            Kv_diff -= 0.5*s**(KP-2)
                        else:
                            Kv_diff -= 0.5*lapPolDiff(ut, order)+np.exp(-ut)*s**(KP-2)
                    sqdist[kk] = Kv_diff/wsig
                for j in range(dim):
                    kk = num_neighbors[j]
                    for jj in range(startSet[kk], endSet[kk]):
                        ii = neighbors[jj]
                        df += sqdist[kk] * 2*(x[k,ii]-y[l,ii])* a[l,ii]
            f[k] = df
    return f

@jit(nopython=True, parallel=True)
def applylocalk_naivedifft(y, x, p, a, name, scale, order, regweight=1., lddmm=False):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros(y.shape)

    wsig = 0
    for s in scale:
        wsig += s ** KP
    if lddmm:
        lddmm_ = 1.0
    else:
        lddmm_ = 0

    if 'lap' in name:
        for k in prange(num_nodes_y):
            for l in range(num_nodes):
                xy = y[k,:] - x[l,:]
                ut0 = np.fabs(xy)
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0 / s
                    Kv_diff -= lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
                sqdist = Kv_diff/wsig
                akl = p[k, :] * a[l, :] + lddmm_ * (a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :])
                f[k, :] += sqdist * xy * akl
    elif 'gauss' in name:
        for k in prange(num_nodes_y):
            for l in range(num_nodes):
                xy = y[k,:] - x[l,:]
                ut0 = np.fabs(xy)
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0/s
                    Kv_diff -= np.exp(-0.5*ut*ut)*s**(KP-2)
                sqdist = Kv_diff/wsig
                akl = p[k, :] * a[l, :] + lddmm_ * (a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :])
                f[k,:] +=  sqdist * xy* akl
    return f

@jit(nopython=True, parallel=True)
def applylocalk_naivediv(x, y, a, name, scale, order):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes,1))

    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = 0
            for l in range(num_nodes_y):
                ut0 = np.fabs(x[k,:] - y[l,:])
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0/s
                    Kv_diff -= 0.5*np.exp(-0.5*ut*ut)*s**(KP-2)
                sqdist = Kv_diff/wsig
                for j in range(dim):
                    df -=  sqdist[j] * 2*(x[k,j]-y[l,j])* a[l,j]
            f[k] = df

    elif 'lap' in name:
        for k in prange(num_nodes):
            df = 0
            for l in range(num_nodes_y):
                ut0 = np.fabs(x[k, :] - y[l, :])
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0 / s
                    Kv_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
                sqdist = Kv_diff/wsig
                for j in range(dim):
                    df -= sqdist[j] * 2*(x[k,j]-y[l,j])* a[l,j]
            f[k] = df
    return f

@jit(nopython=True, parallel=True)
def applykdiff1(x, a1, a2, name, scale, order):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
             ut0 = ((x[k,:] - x[l,:])**2).sum()
             Kh_diff = 0
             for s in scale:
                ut = ut0 / s**2
                if ut < 1e-8:
                    Kh_diff -= 0.5*s**(KP-2)
                else:
                    Kh_diff -= 0.5*np.exp(-0.5*ut)*s**(KP-2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * ((x[k, :] - x[l, :]) * a1[k, :]).sum() * a2[l, :]
            f[k, :] += df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5*lapPolDiff(0,order)*s**(KP-2)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut,order) * np.exp(-1.0*ut)*s**(KP-2)
                Kh_diff /= wsig
                df += Kh_diff * 2*((x[k,:]-x[l,:])*a1[k,:]).sum() * a2[l,:]
            f[k, :] += df
    return f

@jit(nopython=True, parallel=True)
def applykdiff2(x, a1, a2, name, scale, order):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s ** 2
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
                    Kh_diff /= wsig
                    df += Kh_diff * 2 * ((x[k, :] - x[l, :]) * a1[l, :]).sum() * a2[l, :]
            f[k, :] += df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-1.0 * ut) * s ** (KP - 2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * ((x[k, :] - x[l, :]) * a1[l, :]).sum() * a2[l, :]
            f[k, :] += df
    return f

@jit(nopython=True, parallel=True)
def applykdiff1and2(x, a1, a2, name, scale, order):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                dx = x[k,:] - x[l,:]
                ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s ** 2
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * (dx*(a1[k, :] - a1[l, :])).sum() * a2[l, :]
            f[k, :] += df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                dx = x[k,:] - x[l,:]
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-1.0 * ut) * s ** (KP - 2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * (dx*(a1[k, :] - a1[l, :])).sum() * a2[l, :]
            f[k, :] += df
    return f

@jit(nopython=True, parallel=True)
def applykdiff11(x, a1, a2, p, name, scale, order):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = ((x[k,:] - x[l,:])**2).sum()
                Kh_diff = 0
                Kh_diff2 = 0
                for s in scale:
                    ut = ut0/s**2
                    if ut < 1e-8:
                        Kh_diff -= 0.5*s**(KP-2)
                        Kh_diff2 += 0.25 * s**(KP-4)
                    else:
                        Kh_diff -= 0.5*np.exp(-0.5*ut)*s**(KP-2)
                        Kh_diff2 += 0.25*np.exp(-0.5*ut)*s**(KP-4)
                Kh_diff /= wsig
                Kh_diff2 /= wsig
                df += Kh_diff2 * 4 * (a1[k, :] * a2[l, :]).sum() \
                      * ((x[k, :] - x[l, :]) * p[k, :]).sum() * (x[k, :] - x[l, :]) \
                      + Kh_diff * 2 * (a1[k, :] * a2[l, :]).sum() * p[k, :]
            f[k, :] += df
    if 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                Kh_diff2 = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5*lapPolDiff(0, order)*s**(KP-2)
                        Kh_diff2 += 0.25*lapPolDiff2(0, order)*s**(KP-4)
                    else:
                        Kh_diff -= 0.5*lapPolDiff(ut, order) * np.exp(-ut)*s**(KP-2)
                        Kh_diff2 += 0.25*lapPolDiff2(ut, order) * np.exp(-ut)*s**(KP-4)
                Kh_diff /= wsig
                Kh_diff2 /= wsig
                df += Kh_diff2 * 4*(a1[k,:] * a2[l,:]).sum() \
                      * ((x[k,:]-x[l,:])*p[k,:]).sum() *(x[k,:]-x[l,:]) \
                      + Kh_diff * 2 * (a1[k,:] * a2[l,:]).sum() * p[k,:]
            f[k, :] += df

    return  f


@jit(nopython=True, parallel=True)
def applykdiff12(x, a1, a2, p, name, scale, order):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
                Kh_diff = 0
                Kh_diff2 = 0
                for s in scale:
                    ut = ut0 / s ** 2
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * s ** (KP - 2)
                        Kh_diff2 += 0.25 * s ** (KP - 4)
                    else:
                        Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
                        Kh_diff2 += 0.25 * np.exp(-0.5 * ut) * s ** (KP - 4)
                Kh_diff /= wsig
                Kh_diff2 /= wsig
                df += -Kh_diff2 * 4 * (a1[k, :] * a2[l, :]).sum() \
                      * ((x[k, :] - x[l, :]) * p[l, :]).sum() * (x[k, :] - x[l, :]) \
                      - Kh_diff * 2 * (a1[k, :] * a2[l, :]).sum() * p[l, :]
            f[k, :] += df
    if 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                Kh_diff2 = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
                        Kh_diff2 += 0.25 * lapPolDiff2(0, order) * s ** (KP - 4)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
                        Kh_diff2 += 0.25 * lapPolDiff2(ut, order) * np.exp(-ut) * s ** (KP - 4)
                Kh_diff /= wsig
                Kh_diff2 /= wsig
                df += -Kh_diff2 * 4 * (a1[k, :] * a2[l, :]).sum() \
                      * ((x[k, :] - x[l, :]) * p[l, :]).sum() * (x[k, :] - x[l, :]) \
                      - Kh_diff * 2 * (a1[k, :] * a2[l, :]).sum() * p[l, :]
            f[k, :] += df

    return f


@jit(nopython=True, parallel=True)
def applykdiff11and12(x, a1, a2, p, name, scale, order):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                dx = x[k, :] - x[l, :]
                dp = p[k, :] - p[l, :]
                ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
                Kh_diff = 0
                Kh_diff2 = 0
                for s in scale:
                    ut = ut0 / s ** 2
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * s ** (KP - 2)
                        Kh_diff2 += 0.25 * s ** (KP - 4)
                    else:
                        Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
                        Kh_diff2 += 0.25 * np.exp(-0.5 * ut) * s ** (KP - 4)
                Kh_diff /= wsig
                Kh_diff2 /= wsig
                df += 2 * (a1[k,:]*a2[l,:]).sum() *  (2 * Kh_diff2 *(dx*dp).sum() *dx + Kh_diff * dp)
            f[k, :] += df
    if 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                dx = x[k, :] - x[l, :]
                dp = p[k, :] - p[l, :]
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                Kh_diff2 = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
                        Kh_diff2 += 0.25 * lapPolDiff2(0, order) * s ** (KP - 4)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
                        Kh_diff2 += 0.25 * lapPolDiff2(ut, order) * np.exp(-ut) * s ** (KP - 4)
                Kh_diff /= wsig
                Kh_diff2 /= wsig
                df += 2 * (a1[k,:]*a2[l,:]).sum() *  (2 * Kh_diff2 *(dx*dp).sum() *dx + Kh_diff * dp)
            f[k, :] += df

    return f

@jit(nopython=True, parallel=True)
def applykmat(y, x, beta, name, scale, order):
    dim = x.shape[1]
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dimb = beta.shape[2]
    f = np.zeros((num_nodes_y, dimb))

    sKP = scale**KP
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if 'gauss' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dimb)
                for l in range(num_nodes):
                    u = ((ys[k,:] - xs[l,:])**2).sum()/2
                    fk += np.exp(-u)*beta[k,l,:]*sKP[s]
                f[k,:] += fk
        elif 'lap' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dimb)
                for l in range(num_nodes):
                    u = np.sqrt(((ys[k, :] - xs[l, :]) ** 2).sum())
                    u1 = lapPol(u, order) * np.exp(-u) * sKP[s]
                    fk += u1*beta[k,l,:]
                f[k, :] += fk
    f /= wsig
    return f



@jit(nopython=True, parallel=True)
def applykdiffmat(y, x, beta, name, scale, order):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))

    sKP = scale**KP
    sKP1 = scale**(KP-1)
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if 'gauss' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dim)
                for l in range(num_nodes):
                    u = ((ys[k,:] - xs[l,:])**2).sum()/2
                    fk -= sKP1[s] * np.exp(-u) * (ys[k, :] - xs[l, :]) * beta[k,l]
                f[k,:] += fk
        elif 'lap' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dim)
                for l in range(num_nodes):
                    u = np.sqrt(((ys[k, :] - xs[l, :]) ** 2).sum())
                    u1 = lapPolDiff(u, order) * np.exp(-u) * sKP1[s]
                    fk -= u1 * (ys[k, :] - xs[l, :]) * beta[k,l]
                f[k, :] += fk
    f/=wsig
    return f


