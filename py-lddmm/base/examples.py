import numpy as np
from base.surfaces import Surface

def flowers(n=6):
    [x, y, z] = np.mgrid[0:200, 0:200, 0:200] / 100.
    ay = np.fabs(y - 1)
    az = np.fabs(z - 1)
    ax = np.fabs(x - 0.5)
    s2 = np.sqrt(2)
    c1 = np.sqrt(0.06)
    c2 = np.sqrt(0.03)
    c3 = 0.1

    th = np.arctan2(ax, az)
    I1 = c1 ** 2 - (ax ** 2 + 0.5 * ay ** 2 + az ** 2) * (1 + 0.25 * np.cos(n * th))
    # I2 = -(ax ** 2 + 0.5 * ay ** 2 + az ** 2) * (1+0.5*np.cos(6*th))  + c2 ** 2
    I2 = -(ax ** 2 + 0.5 * ay ** 2 + az ** 2) + c2 ** 2
    fvTop = Surface()
    fvTop.Isosurface(I1, value=0, target=3000, scales=[1, 1, 1], smooth=-0.01)
    fvTop = fvTop.truncate((np.array([0, 1, 0, 95]), np.array([0, -1, 0, -105])))

    fvBottom = Surface()
    fvBottom.Isosurface(I2, value=0, target=3000, scales=[1, 1, 1], smooth=-0.01)
    fvBottom = fvBottom.truncate((np.array([0, 1, 0, 95]), np.array([0, -1, 0, -105])))
    return fvBottom,fvTop

def waves(w0=(0,0), w1=(1,0.05), r=(1,0.5), delta=10, d=25):
    [x, y, z] = np.mgrid[0:2*d, 0:2*d, 0:2*d] / d - 1
    az = z - w0[1]*np.sin(w0[0]*np.pi*x)
    fvBottom = Surface()
    fvBottom.Isosurface(az, value=0, target=-1, scales=[1, 1, 1], smooth=-0.01)
    fvBottom = fvBottom.truncate((np.array([0, 1, 0, d - r[1]*d]), np.array([0, -1, 0, -d - r[1]*d]),
                                  np.array([1, 0, 0, d - r[0]*d]), np.array([-1, 0, 0, -d - r[0]*d])))

    az = z - w1[1]*np.sin(w1[0]*np.pi*x)
    fvTop = Surface()
    fvTop.Isosurface(az, value=0,  target=-1, scales=[1, 1, 1], smooth=-0.01)
    fvTop = fvTop.truncate((np.array([0, 1, 0, d - r[1]*d]), np.array([0, -1, 0, -d - r[1]*d]),
                            np.array([1, 0, 0, d - r[0]*d]), np.array([-1, 0, 0, -d - r[0]*d])))
    fvTop.updateVertices(fvTop.vertices + np.array([0,0,d*(w0[1]+w1[1])+delta]))


    return fvBottom,fvTop
