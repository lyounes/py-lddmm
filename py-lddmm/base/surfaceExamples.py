import numpy as np
from scipy.spatial import Delaunay
from .surfaces import Surface

class Rectangle(Surface):
    def __init__(self, corners = None, npt = 25):
        super().__init__()
        if corners is None:
            self.corners = ([0,0,0], [1,0,0], [0,1,0])
        else:
            self.corners = corners

        ul = np.array(self.corners[0])
        ur = np.array(self.corners[1])
        ll = np.array(self.corners[2])
        lr = ur + ll - ul

        t = np.zeros((npt**2,2))
        v0 = ur - ul
        nv0 = np.sqrt((v0**2).sum())
        t0 = np.linspace(0, 1, npt)

        v1 = ll - ul
        nv1 = np.sqrt((v1**2).sum())
        t1 = np.linspace(0, 1, npt)
        x,y = np.meshgrid(t0,t1)
        x = np.ravel(x)
        y = np.ravel(y)
        self.vertices = ul[None, :] +  x[:, None] * v0[None, :] + y[:, None] * v1[None, :]
        pts = np.zeros((npt**2, 2))

        nf = 2*(npt-1)**2
        self.faces = np.zeros((nf,3), dtype=int)
        jf = 0
        for k1 in range(npt -1):
            for k2 in range(npt-1):
                c1 = k1 * npt + k2
                self.faces[jf, 0] = c1
                self.faces[jf, 1] = c1 + npt
                self.faces[jf, 2] = c1 + 1
                self.faces[jf+1, 0] = c1 + 1
                self.faces[jf+1, 1] = c1 + npt
                self.faces[jf+1, 2] = c1 + npt + 1
                jf += 2
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        self.weights = np.ones(self.vertices.shape[0], dtype=int)
        self.face_weights = np.ones(self.faces.shape[0], dtype=int)
        self.computeCentersAreas()









class Sphere(Surface):
    def __init__(self, center=(0,0,0), radius=1, resolution = 100, targetSize = 1000):
        super().__init__()
        self.center = center
        self.radius = radius
        M = resolution
        [x, y, z] = np.mgrid[0:2 * M+1, 0:2 * M+1, 0:2 * M+1] / M
        x = x - 1
        y = y - 1
        z = z - 1
        s2 = np.sqrt(2)
        I1 = .5 - (x**2 + y**2 + z**2)
        self.Isosurface(I1, value = 0, target = targetSize, scales=[1, 1, 1], smooth=0.01)
        v = self.center + (self.vertices -M) * self.radius * s2/M
        self.updateVertices(v)

class Ellipse(Surface):
    def __init__(self, center=(0,0,0), radius=(1,1,1), rotation=None, resolution = 100, targetSize = 1000):
        super().__init__()
        if rotation is None:
            rotation = np.eye(3)
        self.center = center
        self.radius = np.array(radius)
        M = resolution
        [x, y, z] = np.mgrid[0:2 * M+1, 0:2 * M+1, 0:2 * M+1] / M
        x = x - 1
        y = y - 1
        z = z - 1
        s2 = np.sqrt(2)
        I1 = .5 - (x**2 + y**2 + z**2)
        self.Isosurface(I1, value = 0, target = targetSize, scales=[1, 1, 1], smooth=0.01)
        v = self.center + np.dot((self.vertices -M) * self.radius[None,:] * s2/M,  rotation.T)
        self.updateVertices(v)

class Torus(Surface):
    def __init__(self, center, radius1, radius2, resolution = 100, targetSize = 1000):
        super().__init__()
        self.center = center
        self.radius1 = radius1
        self.radius2 = radius2
        M = resolution
        [x, y, z] = np.mgrid[0:2 * M, 0:2 * M, 0:2 * M] / M - 1
        s2 = np.sqrt(2)
        r = radius2/(radius1*s2)
        I1 = r**2 - 0.5 - (x**2 + y**2 + z**2) + s2 * np.sqrt(x**2+y**2)
        self.Isosurface(I1, value = 0, target = targetSize, scales=[1, 1, 1], smooth=0.01)
        v = self.center + (self.vertices -M) * self.radius1 * s2/M
        self.updateVertices(v)


class Heart(Surface):
    def __init__(self, resolution = 100, targetSize = 1000, p=2., parameters = (0.25, 0.20, 0.1), scales=(1., 1.),
                 zoom = 1.):
        super().__init__()
        M = resolution
        [x, y, z] = np.mgrid[0:2 * M, 0:2 * M, 0:2 * M] / M
        ay = np.fabs(y - 1)
        az = np.fabs(z - 1)
        ax = np.fabs(x - 0.5)
        c_out = parameters[0]
        c_in = parameters[1]
        c_up = parameters[2]
        s1 = scales[0]
        s2 = scales[1]

        I1 = np.minimum(c_out ** p / s1 - ((ax ** p + 0.5 * ay ** p + az ** p)),
                        np.minimum((s2 * ax ** p + s2 * 0.5 * ay ** p + s2 * az ** p) - c_in**p / s1, 1 + c_up/s1 - y))

        self.Isosurface(I1, value=0, target=targetSize, scales=[1, 1, 1], smooth=0.01)
        self.vertices[:,1] += 15 - 15/s1
        v = zoom*(self.vertices -M) * np.sqrt(2)/M
        self.updateVertices(v)
