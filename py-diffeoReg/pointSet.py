import numpy as np
import copy
import struct

class Point(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                 strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(Point, subtype).__new__(subtype, shape, dtype,
                                            buffer, offset, strides,
                                            order)
        obj[:] = 0
        return obj




class PointSet(np.ndarray):
    def __new__(subtype, nelem, dim=3, buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(PointSet, subtype).__new__(subtype, (nelem, dim), dtype = float,
                                            buffer = buffer, offset = offset, strides = strides,
                                            order = order)
        obj[:,:] = 0
        obj.dim = dim
        return obj

    def zeros(self, s,d):
        tmp = PointSet(s,d)
        return tmp

    def copy_from(self, src):
        self.resize(src.shape)
        self[:] = src.copy()
        self.dim = src.dim

    def al(self, n, d):
        self.resize(n,d)
        self[:,:] = 0
        self.dim = d

    def __copy__(self):
        return copy.deepcopy(self)

    def get_points(self, file, dm):
        with open(file, 'r') as ifs:
            N = 0
            line = ifs.readline()
            line = ifs.readline()
            while len(line) > 0 and line[0] not in ['\n', '#']:
                line = ifs.readline()
            content = line.split()
            N = int(content[0])
            self.al(N, dm)
            for k in range(N):
                line = ifs.readline()
                while len(line) > 0 and line[0] not in ['\n', '#']:
                    line = ifs.readline()
                content = line.split()
                for l in range(dm):
                    self[k,l] = float(content[l])

    def read(self, filein):
        with open(filein, 'r') as ifs:
            bfr = ifs.read()

        frm = 'ii'
        Ndm = struct.unpack_from(bfr, frm)
        frm += 'f'*(Ndm[0]*Ndm[1])
        self.resize(Ndm[0], Ndm[1])
        Ndm = struct.unpack_from(bfr, frm)
        self[:] = Ndm[2:]

    def write_points(self, fileout):
        with open(fileout, 'w') as ofs:
            N=self.shape[0]
            ofs.write("Landmarks\n")
            ofs.write(str(N)+'\n')
            for k in range(N):
                ofs.write(str(k)+'\n')
                for j in range(self.shape[1]):
                    ofs.write(f'{self[k,j]:.4f} ')
                ofs.write('\n')

    def write(self, fileout):
        with open(fileout, 'w') as ofs:
            frm = 'ii'+ 'f'*self.size()
            bts = struct.pack(frm, self.shape, self[...])
            ofs.write(bts)

    def affineInterp(self, mat):
        d = self.shape[1]
        res = PointSet(self.shape[0], self.shape[1])
        res[...] = np.dot(self[...], mat[:,:d].T) + mat[:,d]
        return res

    def norm2(self):
        res = (self[...]**2).sum()

    def __repr__(self):
        res = ''
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res += str(self[i,j]) + " "
            res += '\n'


class MatSet:
    def __init__(self, s=0, d=0):
        self._s = s
        self._d = d
        self._t = np.zeros((s,d,s,d))

    def size(self):
        return self._s
    def dim(self):
        return self._d
    def copy_from(self, src):
        self._s = src._s
        self._d = src._d
        self._t = src._t.copy()

    def zeros(self, s, d):
        self._s = s
        self._d = d
        self._t = np.zeros((s, d, s, d))

    def eye(self, s, d):
        self._s = s
        self._d = d
        self._t = np.zeros((s, d, s, d))
        for k in range(s):
            for kk in range(d):
                self._t[k,kk,k,kk] = 1

def PointSetScp(x,y):
    return (x*y).sum()


def affineInterp(src, mat):
    d = src.shape[1]
    res = PointSet(src.shape[0], src.shape[1])
    res[...] = np.dot(src[...], mat[:, :d].T) + mat[:, d]
    return res
