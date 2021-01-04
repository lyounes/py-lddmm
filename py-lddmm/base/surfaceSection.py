import numpy as np
from scipy import linalg as LA
from .curves import Curve
from .surfaces import Surface
from numba import jit, int64
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@jit(nopython=True)
def ComputeIntersection_(VS, SF, ES, FES, u, offset):
    h = np.dot(VS, u) - offset
    #h = (VS* u[None,:]).sum(axis=1) - offset
    tol = 0
    vertices = np.zeros((ES.shape[0], 3))
    intersect = -np.ones(ES.shape[0], dtype=int64)

    iv = 0
    for i in range(ES.shape[0]):
        he0 = h[ES[i,0]]
        he1 = h[ES[i,1]]
        if (he0 > tol and he1 < -tol) or (he0 < -tol and he1 > tol):
            r = -he0/(he1-he0)
            vertices[iv, :] = (1-r) * VS[ES[i,0],:] + r * VS[ES[i,1],:]
            intersect[i] = iv
            iv += 1
    vertices = vertices[:iv, :]

    edges = np.zeros((FES.shape[0], 2),dtype=int64)
    ie = 0
    for i in range(FES.shape[0]):
#        I = FS[i,:]
        J = FES[i, :]
        for k in (0,1,2):
            if intersect[J[k]]>=0 and intersect[J[(k+1)%3]]>=0:
                i1 = intersect[J[k]]
                i2 = intersect[J[(k+1)%3]]
                v = vertices[i2, :] - vertices[i1,:]
                if np.sum(np.cross(u, v)*SF[i,:]) > 0:
                    edges[ie, :] = [i1, i2]
                else:
                    edges[ie, :] = [i2, i1]
                ie += 1
    edges = edges[:ie, :]
    return edges, vertices

@jit(nopython=True)
def CurveGrad2Surf(curveGrad, VS, ES, u, offset):
        h = np.dot(VS, u) - offset
        tol = 0
        grad = np.zeros(VS.shape)

        iv = 0
        for i in range(ES.shape[0]):
            he0 = h[ES[i, 0]]
            he1 = h[ES[i, 1]]
            if (he0 > tol and he1 < -tol) or (he0 < -tol and he1 > tol):
                d = he1 - he0
                r = -he0 / d
                dp = VS[ES[i, 1], :] - VS[ES[i, 0], :]
                g0 = curveGrad[iv, :] - ((dp*curveGrad[iv, :]).sum()/d) * u
                grad[ES[i,0], :] += (1-r) * g0
                grad[ES[i,1], :] += r * g0
                iv += 1
        return grad

        # if hfM[i] > -tol and hfm[i].min() < tol:
        #     hi = h[i,:]
        #     if shf[i].sum() > 1-1e-10:
        #         i0 = np.argmin(hi)
        #     else:
        #         i0 = np.argmax(hi)
        #     i1 = (i0+1) % 3
        #     i2 = (i0+2) % 3
        #     h0 = hi[i0]
        #     h1 = hi[i1]
        #     h2 = hi[i2]
        #     p0 = VS[FS[i,i0], :]
        #     p1 = VS[FS[i, i1], :]
        #     p2 = VS[FS[i, i2], :]
        #     q0 = (h0*p1 - h1*p0)/(h0-h1)
        #     q1 = (h0*p2 - h2*p0)/(h0-h2)
        #     vertices[iv, :] = q0
        #     vertices[iv+1,:] = q1
        #     edges[ie, :] = [iv, iv+1]
        #     iv += 2
        #     ie += 1
    #return edges, vertices





class Hyperplane:
    def __init__(self, hyperplane=None, u = (0,0,1), offset = 0):
        if hyperplane is None:
            self.u = np.array(u, dtype=float)
            self.offset = offset
        elif type(hyperplane) is Hyperplane:
            self.u = hyperplane.u
            self.offset = hyperplane.offset
        else:
            self.u = hyperplane[0]
            self.offset = hyperplane[1]

class SurfaceSection:
    def __init__(self, curve=None, hyperplane=None, surf = None, plot = None):
        self.hyperplane = Hyperplane(hyperplane)
        if surf is None:
            self.curve = Curve(curve)
            self.ComputeHyperplane(curve)
        else:
            self.ComputeIntersection(surf, hyperplane, plot=plot)
        #self.normals = self.curve.computeUnitVertexNormals()
        self.normals = np.cross(self.hyperplane.u, self.curve.linel)
        self.normals /= np.maximum(np.sqrt((self.normals**2).sum(axis=1)), 1e-10)[:, None]
        self.area = -1
        self.outer = False
        self.hypLabel = -1

    def ComputeIntersection(self, surf, hyperplane, plot = None):
        if surf.edges is None:
            surf.getEdges()
        F, V = ComputeIntersection_(surf.vertices, surf.surfel, surf.edges, surf.faceEdges,
                                    hyperplane.u, hyperplane.offset)
        self.curve = Curve([F,V])
        if plot is not None:
            fig = plt.figure(plot)
            fig.clf()
            ax = Axes3D(fig)
            lim1 = self.curve.addToPlot(ax, ec='k', fc='b', lw=1)
            ax.set_xlim(lim1[0][0], lim1[0][1])
            ax.set_ylim(lim1[1][0], lim1[1][1])
            ax.set_zlim(lim1[2][0], lim1[2][1])
            fig.canvas.flush_events()

        # self.curve.removeDuplicates()
        # self.curve.orientEdges()
        self.hyperplane.u = hyperplane.u
        self.hyperplane.offset = hyperplane.offset

    def ComputeHyperplane(self, c):
        m = c.vertices.mean(axis=0)
        vm = c.vertices-m[None, :]
        S = np.dot(vm.T, vm)
        l,v = LA.eigh(S, subset_by_index = [0,0])
        self.hyperplane = Hyperplane(u=v[:,0], offset=(m[:,None]*v).sum())


def Surf2SecDist(surf, s, curveDist, curveDist0 = None, plot = None):
    s0 = SurfaceSection(surf=surf, hyperplane=s.hyperplane, plot = plot)
    obj = curveDist(s0.curve, s.curve)
    if curveDist0 is not None:
        obj2 = obj + curveDist0(s.curve)
    return obj

def Surf2SecGrad(surf, s, curveDistGrad):
    s0 = SurfaceSection(surf=surf, hyperplane=s.hyperplane)
    if s0.curve.vertices.size > 0:
        cgrad = curveDistGrad(s0.curve, s.curve)
        grad = CurveGrad2Surf(cgrad, surf.vertices, surf.edges, s.hyperplane.u, s.hyperplane.offset)
    else:
        grad = np.zeros(surf.vertices.shape)
    return grad


def readTargetFromTXT(filename):
    fv1 = ()
    with open(filename, 'r') as f:
        s = f.readline()
        nc = int(s)
        for i in range(nc):
            s = f.readline()
            npt = int(s)
            pts = np.zeros((npt,3))
            for j in range(npt):
                s = f.readline()
                pts[j,:] = s.split()
            c = Curve(pts)
            fv1 += (SurfaceSection(curve=c),)
    uo = np.zeros((nc, 4))
    nh = 0
    tol = 1e-5
    hk = np.zeros(4)
    for k,f in enumerate(fv1):
        f.area = f.curve.enclosedArea()
        found = False
        hk[:3] = f.hyperplane.u
        hk[3] = f.hyperplane.offset
        if k > 0:
            dst = np.sqrt(((hk - uo[:nh, :])**2).sum(axis=1))
            dst2 = np.sqrt(((hk + uo[:nh, :])**2).sum(axis=1))
            if dst.min() < tol:
                i = np.argmin(dst)
                f.hypLabel = i
                found = True
            elif dst2.min() < tol:
                i = np.argmin(dst)
                f.hypLabel = i
                found = True
        if not found:
            uo[nh, :] = hk
            f.hypLabel = nh
            nh += 1
    hyperplanes = uo[:nh, :]
    J = -np.ones(nh, dtype=int)
    maxArea = np.zeros(nh)
    for k,f in enumerate(fv1):
        f.hyperplane.u = uo[f.hypLabel, :3]
        f.hyperplane.offset = uo[f.hypLabel, 3]
        f.outer = False
        if f.area > maxArea[f.hypLabel]:
            maxArea[f.hypLabel] = f.area
            J[f.hypLabel] = k



    for k in range(nh):
        fv1[J[k]].outer = True

    eps = 1e-4
    for k,f in enumerate(fv1):
        c = Curve(curve=f.curve)
        n = np.zeros(c.vertices.shape)
        for i in range(c.faces.shape[0]):
            n[c.faces[i, 0], :] += f.normals[i]/2
            n[c.faces[i, 1], :] += f.normals[i] / 2
        c.updateVertices(c.vertices + eps*n)
        a = np.fabs(c.enclosedArea())
        if (a > f.area and not f.outer) or \
                (a < f.area and f.outer):
            f.curve.flipFaces()
            f.normals *= -1

    return fv1, hyperplanes