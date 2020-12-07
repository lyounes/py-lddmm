import numpy as np
from ivector import Ivector, Domain
import copy
import struct


def absMax(a, b):
    return ((a + b + np.fabs(a - b)) / 2)


def absMin(a, b):
    return ((a + b - np.fabs(a - b)) / 2)


class _Vector(np.ndarray):
    def __new__(subtype, dtype=float, domain=None, M=None, m=None):
        if domain is not None:
            d_ = domain
        elif m is not None:
            d_ = Domain(m=m, M=M)
        elif M is not None:
            d_ = Domain(M= M)
        else:
            d_ = None
        if d_ is not None:
            obj = super(_Vector, subtype).__new__(subtype, shape=d_.length, dtype=dtype)
            obj.d = d_.copy()
            obj[:] = 0
        else:
            obj = super(_Vector, subtype).__new__(subtype, shape=0, dtype=dtype)
            obj.d = Domain()
        return obj

    def domain(self):
        return self.d

    def resize(self, shape = None, domain=None, MINMAX=None):
        if domain is None:
            if MINMAX is None:
                if shape is None:
                    return
                else:
                    I = Ivector(len(shape))
                    J = Ivector(len(shape))
                    J[:] = shape
                    self.d = Domain(I,J)
            else:
                self.d = Domain(MINMAX[0], MINMAX[1])
        else:
            self.d = domain.copy()
        super(_Vector, self).resize(self.d.length)

    def zeros(self, shape = None, domain=None, MINMAX=None):
        if domain is None:
            if MINMAX is None:
                if shape is None:
                    return
                else:
                    I = Ivector(len(shape))
                    J = Ivector(len(shape))
                    J[:] = shape
                    self.d = Domain(I,J)
            else:
                self.d = Domain(MINMAX[0], MINMAX[1])
        else:
            self.d = domain.copy()
        super(_Vector, self).resize(self.d.shape)
        self[...] = 0

    def ones(self, shape = None, domain=None, MINMAX=None):
        if domain is None:
            if MINMAX is None:
                if shape is None:
                    return
                else:
                    I = Ivector(len(shape))
                    J = Ivector(len(shape))
                    J[:] = shape
                    self.d = Domain(I,J)
            else:
                self.d = Domain(MINMAX[0], MINMAX[1])
        else:
            self.d = domain.copy()
        super(_Vector, self).resize(self.d.shape)
        self[...] = 1

    def copy_from(self, x):
        self.resize(x.d)
        self[...] = x[...]

    def __copy__(self):
        return copy.deepcopy(self)

    def length(self):
        return self.d.length

    def pos(self):
        return self.d.positive()

    def getValue(self, I):
        return self[I - self.d.m]

    def shift(self, step):
        dest = self.copy()
        for c in range(self.length()):
          dest[c] = self[self.d.rPos(c, dir, step)]
        return dest

    def crop(self, D):
        im = Ivector(D.n)
        iM = D.M - D.m
        DD = Domain(im, iM)
        dest = _Vector(D)
        le = dest.size
        I = D.M
        dest[:] = self[:]
        for i in range(le):
            dest[i] = self.getValue(I)
            I.inc(D.m, D.M)
        print("cropped", D)
        return dest


    def subCopy(self, x, MINMAX = None):
        if MINMAX is None:
            MIN = x.d.m
            MAX = x.d.M
            m = self.d.m
            M = self.d.M
            if self.d.n == 2:
              c = 0
              for k0 in range(MIN[0], MAX[0]):
                for k1 in range(MIN[1], MAX[1]):
                  iy = ((k0-m[0]) * (M[1]-m[1]) + (k1-m[1]))
                  self[iy] = x[c]
                  c += 1
            elif self.d.n == 3:
                c=0
                for k0 in range(MIN[0], MAX[0]):
                    for k1 in range(MIN[1], MAX[1]):
                        for k2 in range(MIN[2], MAX[2]):
                            iy = ((k0 - m[0]) * (M[1] - m[1]) + (k1 - m[1])) * (M[2] - m[2]) + k2 - m[2]
                            self[iy] = x[c]
                            c += 1

            else:
                I = MIN.copy()
                iy = self.d.position(I)
                c = 0
                while c < x.length():
                    self[iy] = x[c]
                    c += 1
                    I.inc(MIN, MAX)
                    iy = self.d.position(I)
        else:
            c = 0
            I = MINMAX[0]
            iy = self.d.position(I)

            while (c < x.d.length):
                self[iy] = x[c]
                c += 1
                I.inc(MINMAX[0], MINMAX[1])
                iy = self.d.position(I)


    def symCopy(self, x):
        MIN0 = x.d.m
        MAX0 = x.d.M
        MIN = self.d.m
        MAX = self.d.M
        I0 = MIN0.copy()
        c=0
        while c < x.d.length:
            I = I0.copy()
            for i in range(self.d.n):
                if I[i] < 0:
                    I[i] = MAX[i] + I[i]
                else:
                    I[i] += MIN[i]
            iy = self.d.position(I)
            self[iy] = x[c]
            c += 1
            I0.inc(MIN0, MAX0)

    def flip(self,dm):
        res = _Vector(domain = self.d)
        c = 0
        MIN = self.d.m
        MAX = self.d.M
        I0 = MIN.copy()
        while c < self.d.length:
            I = I0.copy()
            I[dm] = MAX[dm] - I[dm] + MIN[dm] -1
            iy = self.d.position(I)
            res[iy] = self[c]
            c += 1
        return res

    def extract(self, x, MINMAX = None):
        c = 0
        if MINMAX is None:
            MIN = x.d.m
            MAX = x.d.M
        else:
            MIN = MINMAX[0]
            MAX = MINMAX[1]
        I = MIN.copy()
        iy = self.d.position(I)
        while c < x.length():
            x[c] = self[iy]
            c += 1
            I.inc(MIN, MAX)
            iy = self.d.position(I)


    def expandBoundaryCentral(self, margin, value=None):
        if value is None:
            value = self.avgBoundary()
        tmp = self.copy()
        I = self.d.m - margin
        J = self.d.M + margin
        self.resize(MINMAX = [I,J])
        self[:] = value

        I += margin
        J -= margin
        self.subCopy(tmp, MINMAX= [I, J])


    def expandBoundary(self, margin, value=None):
        if value is None:
            value = self.avgBoundary()
        if type(margin) in (int, np.ndarray, Ivector):
            tmp = self.copy()
            I = self.d.m
            J = self.d.M + 2*margin
            self.resize(MINMAX = [I,J])
            self[:] = value
            I += margin
            J -= margin
            self.subCopy(tmp, MINMAX=[I, J])
        elif type(margin) is Domain:
            tmp = self.copy()
            diff1 = Ivector(self.d.n)
            diff2 = Ivector(self.d.n)

            for k in range(self.d.n):
                diff1[k] = (margin.M[k] - margin.m[k] - (self.d.M[k] - self.d.m[k])) // 2
                diff2[k] = margin.M[k] - margin.m[k] - (self.d.M[k] - self.d.m[k]) - diff1[k]

            self.resize(margin)
            self[:] = value
            I = margin.m + diff1
            J = margin.M - diff2
            self.subCopy(tmp, MINMAX=[I, J])
        else:
            return


    def write(self, path):
        f = 'i'*(1+2*self.ndim) + 'f'*self.length()
        bts = struct.pack(f, self.ndim, *self.domain().m, *self.domain().M, *self[:])
        with open(path, 'w') as ofs:
            ofs.write(bts)

    def read(self, path):
        with open(path, 'r') as ifs:
            bts = ifs.read()
        self.ndim = struct.unpack_from('i', bts)
        f = 'i'*(1+2*self.ndim)
        tmp = struct.unpack_from(f, bts)
        I = tmp[1:4]
        J = tmp[4:]
        self.resize(MINMAX=[I,J])
        f = 'i'*(1+2*self.ndim) + 'f'*self.length()
        tmp = struct.unpack_from(f, bts)
        self[:] = bts[(1+2*self.ndim):]

    def zero(self):
        self[:] = 0


    def binarize(self, binT, coeff):
        J = self > binT
        self[:] = 0
        self[J] = coeff


    def rescale(self, D):
        kmin = np.zeros(D.n, dtype=int)
        kmax = np.zeros(D.n, dtype=int)
        rkmin = np.zeros(D.n)
        rkmax = np.zeros(D.n)

        ratio = (self.d.M - self.d.m) / (D.M - D.m)
        offset = self.d.m - ratio *D.m
        I = D.m.copy()
        res = _Vector(D)
        c=0

        while c < D.length:
            for k in range(D.n):
                kmin[k] = np.floor(offset[k] + ratio[k] * I[k])
                kmax[k] = np.ceil(offset[k] + ratio[k] * (I[k] + 1))
                if kmax[k] >= self.d.M[k]:
                    kmax[k] = self.d.M[k]
                if kmax[k] == kmin[k]:
                    rkmin[k] = ratio[k]
                else:
                    rkmin[k] = kmin[k] + 1 - ratio[k] * I[k]
                    rkmax[k] = ratio[k] * (I[k] + 1) - kmax[k] + 1

                Dloc = Domain(kmin, kmax)
                J = kmin.copy()
                c2 = 0
                u=0
                ww=0
                while c2 < Dloc.length():
                    zz = 1
                    for kloc in range(D.n):
                        if J[kloc] == kmin[kloc]:
                            zz *= rkmin[kloc]
                        elif J[kloc] == kmax[kloc]:
                            zz *= rkmax[kloc]
                    ww += zz
                    u = zz * self.getvalue[J]
                    Dloc.inc(J)
                    c2 += 1
                u /= ww
                res[c] = u
                D.inc(I)
                c+= 1
        return res

    def __add__(self, src):
        res = _Vector(domain = self.d)
        res[:] = self[:] + src
        return res


    def __sub__(self, src):
        res = _Vector(domain=self.d)
        res[:] = self[:] - src
        return res

    def sqr(self):
        self[:] *= self[:]

    def __iadd__(self, x):
        self[:] += x

    def __isub__(self, x):
        self[:] -= x

    def __imul__(self, x):
        self[:] *= x

    def __idiv__(self, x):
        self[:] /= x

    def sumProd(self, y):
        return (self[:]*y).sum()

    def norm2(self):
        return (self[:]**2).sum()

    def dist2(self, w):
        return ((self[:] - w)**2).sum()

    def execFunction(self, fun):
        res = _Vector(domain=self.d)
        for i in range(self.length()):
            res[i] = fun(i)

    def avgBoundary(self):
        c = 0
        mm = 0
        nn = 0
        I = self.d.m.copy()
        while c < self.length():
            for j in range(self.d.n):
                if I[j] == self.d.m[j] or I[j] == self.d.M[j]:
                    mm += self[c]
                    nn +=1
                    break
            c += 1
            self.d.inc(I)

        return mm/nn

    def stdBoundary(self):
        c = 0
        vv = 0
        mm = 0
        nn = 0
        I = self.d.m.copy()
        while c < self.length():
            for j in range(self.d.n):
                if I[j] == self.d.m[j] or I[j] == self.d.M[j]:
                    vv += self[c]**2
                    mm += self[c]
                    nn +=1
                    break
            c += 1
            self.d.inc(I)

        return np.sqrt(vv /nn - (mm/nn)**2)


    def medianBoundary(self):
        c = 0
        nn = 0
        tmp = np.zeros(self.length())
        I = self.d.m.copy()
        while c < self.length():
            for j in range(self.d.n):
                if I[j] == self.d.m[j] or I[j] == self.d.M[j]:
                    tmp[nn] = self[c]
                    nn +=1
                    break
            c += 1
        return np.median(tmp[:nn])

    def maxAbs(self):
        return np.fabs(self).max()


    def censor(self, q, mask = None):
        if mask is None:
            b = np.quantile(self, [q,1-q])
        else:
            b = np.quantile(self[mask], [q,1-q])

        self[:] = np.maximum(np.minimum(self, b[1]), b[0])


    def laplacianSmoothing(self,  nIter, w):
        tmp = _Vector(domain = self.d)
        for it in range(nIter):
            I = np.copy(self.d.m)
            for c in range(self.length()) :
                S = 0
                nb = 0
                for k in range(self.d.n):
                    if I[k] < self.d.M[k]-1:
                        S += self.d.rPos(c, k, 1)
                        nb += 1
                    if I[k] > self.d.m[k]:
                        S += self.d.rPos(c, k, -1)
                        nb += 1
                S *= w / nb
                tmp[c] = self[c]
                tmp[c] *= 1 - w
                tmp[c] += S
                self.d.inc(I)
        self[:] = tmp

    def erosion(self, nIter, Mask0):
        tmp = _Vector(domain=self.d, dtype=bool)
        Mask = Mask0.copy()
        for it in range(nIter):
            I = self.d.m.copy()
            for c in range(self.length()):
                res = 1
                for k in range(self.d.n):
                    if I[k] < self.d.M[k]-1:
                        if not Mask[self.d.rPos(c, k, 1)]:
                            res = 0
                    if I[k] > self.d.m[k]:
                        if not Mask[self.d.rPos(c, k, -1)]:
                            res = 0
                tmp[c] = res
                self.d.inc(I)
            Mask = tmp.copy()
        return Mask
