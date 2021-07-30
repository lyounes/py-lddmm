import numpy as np
import copy


# class InfoArray(np.ndarray):
#     def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
#                 strides=None, order=None, info=None):
#         obj = super(InfoArray, subtype).__new__(subtype, shape, dtype,
#                                                 buffer, offset, strides,
#                                                 order)
#         # set the new 'info' attribute to the value passed
#         obj.info = info
#         # Finally, we must return the newly created object:
#         return obj


class Ivector(np.ndarray):
    def __new__(subtype, shape, dtype=int, buffer=None, offset=0,
                 strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(Ivector, subtype).__new__(subtype, shape, dtype,
                                            buffer, offset, strides,
                                            order)
        obj[:] = 0
        return obj


    # def __array_finalize__(self, obj):
    #   if obj is None:
    #     return
    #   self[:] = 0

    # def __new__(subtype, s):
    #   print(subtype)
    #   obj = super(Ivector, subtype).__new__(subtype, s, dtype=int)

    def ok(self):
        return self.size > 0

    def pos(self):
        if self.size <= 0:
            return False
        for i in range(self.size):
            if self[i] < 0:
                return False
        return True

    def __eq__(self, other):
        if other.size != self.size:
            return False
        for i in range(self.size):
            if other[i] != self[i]:
                return False
        return True

    def __ne__(self, other):
        if other.size != self.size:
            return True
        for i in range(self.size):
            if other[i] != self[i]:
                return True
        return False

    def __le__(self, other):
        if other.size != self.size:
            return False
        for i in range(self.size):
            if other[i] < self[i]:
                return False
        return True

    def __lt__(self, other):
        eq = True
        if other.size != self.size:
            return False
        for i in range(self.size):
            if other[i] < self[i]:
                return False
            if other[i] > self[i]:
                eq = False
        return not eq

    def inc(self, min_=None, max_=None):
        if max_ is None:
            return -1
        if min_ is None:
            min_ = Ivector(self.size)

        if self == (max_-1):
            return -1
        i = max_.size- 1
        while i >= 0:
            if self[i] < max_[i]-1:
                self[i] += 1
                for j in range(i + 1, max_.size):
                    self[j] = min_[j]
                break
            i -= 1
        return i

    def zero(self):
        for i in range(self.size):
            self[i] = 0


class Domain:
    def __init__(self, m=None, M=None):
        self.n = 0
        self.length = 0
        if M is not None:
            self.M = M.copy()
            self.n = M.size
            if m is not None:
                self.m = m.copy()
            else:
                self.m = Ivector(self.n)

        self.cum = np.zeros(self.n, dtype=int)
        self.cumMin = 0
        self.calc_cum()

    def __eq__(self, other):
        return other.m == self.m and other.M == self.m

    def __ne__(self, other):
        return (other.m != self.m or other.M != self.M)

    def shape(self):
        return self.M - self.m

    def dimension(self):
        return self.n

    def minWidth(self):
        return (self.M - self.m).min()

    def maxWidth(self):
        return (self.M - self.m).max()

    def getCum(self, i):
        return self.cum[i]

    def getCumMin(self):
        return self.cumMin

    def move(self, step, direction):
        return step * self.cum[direction]

    # def putm(Ivector &I) const {I.resize(n); for(unsigned int i=0; i<n;i++) I[i] = m[i];}

    def shiftPlus(self, S):
        self.m += S
        self.M += S

    def shiftMinus(self, S):
        self.m -= S
        self.M -= S

    def getMax(self):
        return self.M.copy() - 1


    # void putM(Ivector &I) const {I.resize(n); for(unsigned int i=0; i<n;i++) I[i] = M[i];}

    # /**
    #     sets value of ith index of  lower bound
    # */
    # void setm(int i, int k) {m[i] = k ;}
    # /**
    #    sets value of ith index of  upper bound
    # */
    # void setM(int i, int k) {M[i] = k;}
    # /**
    #    returns value of ith index of  lower bound
    # */
    # int getm(const int i) const {return m[i];}
    # /**
    #    returns value of ith index of  upper bound
    # */
    # int getM(const int i) const {return M[i];}
    #
    # /**
    #    increments index I within the domain
    # */
    def inc(self, I):
        return I.inc(self.m, self.M)

    # relative position
    def rPos(self, i0, dim, step):
        return i0 + step * self.cum[dim]

    # _real rPos(std::vector<_real>::iterator i0, const int dim, const int step) const { return *(i0 + step * cum[dim]) ; }

    def copy(self):
        s = Domain(self.m, self.M)
        return s

    def calc_cum2(self):
        if self.n > 0:
            self.cum[0] = 1
            for i in range(1, self.n):
                self.cum[i + 1] = (self.M[i] - self.m[i]) * self.cum[i]
            self.length = self.cum[self.n - 1] * (self.M[self.n - 1] - self.m[self.n - 1])
            self.cumMin = (self.cum * self.m).sum()
            return self.length
        else:
            return -1

    def calc_cum(self):
        if self.n > 0:
            self.cum[self.n - 1] = 1
            for i in range(self.n - 2, -1, -1):
                self.cum[i] = (self.M[i + 1] - self.m[i + 1]) * self.cum[i + 1]
            self.length = self.cum[0] * (self.M[0] - self.m[0])
            self.cumMin = (self.cum * self.m).sum()
            return self.length
        else:
            return -1

    def positive(self):
        return self.length > 0

    def position(self, u):
        j = - self.cumMin
        for i in range(self.n):
            j += u[i] * self.cum[i]
        return j

    def fromPosition(self, u, i):
        u = Ivector(self.n)
        ii = i
        for j in range(self.n):
            u[j] = ii / self.cum[j] + self.m[j]
            ii = ii % self.cum[j]

    def __repr__(self):
        st = ""
        for i in range(self.n):
            st += str(self.m[i]) + " "
        st += '\n'
        for i in range(self.n):
            st += str(self.M[i]) + " " + '\n'
        st += '\n'
        return st
