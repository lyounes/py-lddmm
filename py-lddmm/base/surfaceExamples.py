import numpy as np
from .surfaces import Surface


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
