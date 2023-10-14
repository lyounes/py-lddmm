import matplotlib
import os

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, int64
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import scipy as scp
import scipy.linalg as LA
from scipy.sparse import csr_matrix, lil_matrix
from .curves import Curve
from .surfaces import Surface
from .vtk_fields import vtkFields
from meshpy.geometry import GeometryBuilder
import meshpy.tet as tet
import meshpy.triangle as tri
from scipy.sparse.csgraph import connected_components
from skimage import measure
from copy import deepcopy
try:
    from vtk import vtkCellArray, vtkPoints, vtkPolyData, vtkVersion,\
        vtkLinearSubdivisionFilter, vtkQuadricDecimation,\
        vtkWindowedSincPolyDataFilter, vtkImageData, VTK_FLOAT,\
        vtkDoubleArray, vtkContourFilter, vtkPolyDataConnectivityFilter,\
        vtkCleanPolyData, vtkPolyDataReader, vtkUnstructuredGridReader, vtkOBJReader, vtkSTLReader,\
        vtkDecimatePro, VTK_UNSIGNED_CHAR, vtkPolyDataToImageStencil,\
        vtkImageStencil
    from vtk.util.numpy_support import vtk_to_numpy
    gotVTK = True
except ImportError:
    v2n = None
    print('could not import VTK functions')
    gotVTK = False

# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.colors import LightSource
# from matplotlib import colors
# from matplotlib import pyplot as plt
#
# from . import diffeo
# import scipy.linalg as spLA
# from scipy.sparse import coo_matrix
# import glob
import logging
from .pointSets_util import det2D, det3D

# @jit(nopython=True)
# def det2D(x1, x2):
#     return x1[:,0]*x2[:,1] - x2[:,0]*x1[:,1]
#
# @jit(nopython=True)
# def det3D(x1, x2, x3):
#     return (x1 * np.cross(x2, x3)).sum(axis = 1)


def twelve_vertexes(dimension=3):
    if dimension == 2:
        t = np.linspace(0, 2*np.pi, 12)
        ico = np.zeros((12,2))
        ico[:,0] = np.cos(t)
        ico[:,1] = np.sin(t)
    else:
        phi = (1+np.sqrt(5))/2

        ico = np.array([
            [phi, 1, 0],
            [phi,-1, 0],
            [-phi, -1, 0],
            [-phi, 1, 0],
            [1, 0, phi],
            [-1, 0, phi],
            [-1, 0, -phi],
            [1, 0,-phi],
            [0, phi, 1],
            [0, phi,-1],
            [0,-phi, -1],
            [0,-phi, 1]]
        )

    return ico

#@jit(nopython=True)
def computeCentersVolumesNormals__(faces, vertices, weights, checkOrientation= False):
    dim = vertices.shape[1]
    if dim == 2:
        xDef1 = vertices[faces[:, 0], :]
        xDef2 = vertices[faces[:, 1], :]
        xDef3 = vertices[faces[:, 2], :]
        centers = (xDef1 + xDef2 + xDef3) / 3 ##
        x12 = xDef2-xDef1
        x13 = xDef3-xDef1
        volumes =  det2D(x12, x13)/2
        if checkOrientation:
            if volumes.min() < -1e-12:
                if volumes.max() > 1e-12:
                    print('Warning: mesh has inconsistent orientation', (volumes<0).sum(), (volumes>0).sum())
                else:
                    f_ = np.copy(faces[:,1])
                    faces[:, 1] = np.copy(faces[:,2])
                    faces[:,2] = f_
                    xDef2 = vertices[faces[:, 1], :]
                    xDef3 = vertices[faces[:, 2], :]
                    x12 = xDef2-xDef1
                    x13 = xDef3-xDef1
                    volumes =  (x12[:,0] * x13[:,1] - x12[:,1]*x13[:,0])/2
        J = np.array([[0., -1.], [1.,0.]])
        normals = np.zeros((3, faces.shape[0], 2))
        normals[0, :, :] = (xDef3 - xDef2) @ J.T
        normals[1, :, :] = (xDef1 - xDef3) @ J.T
        normals[2, :, :] = (xDef2 - xDef1) @ J.T
    elif dim == 3:
        xDef1 = vertices[faces[:, 0], :]
        xDef2 = vertices[faces[:, 1], :]
        xDef3 = vertices[faces[:, 2], :]
        xDef4 = vertices[faces[:, 3], :]
        centers = (xDef1 + xDef2 + xDef3 + xDef4) / 4
        x12 = xDef2-xDef1
        x13 = xDef3-xDef1
        x14 = xDef4-xDef1
        volumes =  det3D(x12, x13, x14)/6
        if checkOrientation:
            if volumes.min() < -1e-12:
                if volumes.max() > 1e-12:
                    print('Warning: mesh has inconsistent orientation', (volumes<0).sum(), (volumes>0).sum())
                else:
                    f_ = np.copy(faces[:,2])
                    faces[:, 2] = np.copy(faces[:,3])
                    faces[:,3] = f_
                    #faces = faces[:, [0,1,3,2]]
                    xDef3 = vertices[faces[:, 2], :]
                    xDef4 = vertices[faces[:, 3], :]
                    x13 = xDef3 - xDef1
                    x14 = xDef4 - xDef1
                    volumes = det3D(x12, x13, x14) / 6
        normals = np.zeros((4, faces.shape[0], 3))
        normals[0, :, :] = np.cross(xDef4 - xDef2, xDef3 - xDef2)
        normals[1, :, :] = np.cross(xDef2 - xDef1, xDef4 - xDef1)
        normals[2, :, :] = np.cross(xDef4 - xDef1, xDef2 - xDef1)
        normals[3, :, :] = np.cross(xDef2 - xDef1, xDef3 - xDef1)
    # else:
    #     return np.zeros(0), np.zeros(0),np.zeros(0),np.zeros(0)

    vertex_weights = np.zeros(vertices.shape[0])
    face_per_vertex = np.zeros(vertices.shape[0])
    for k in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            vertex_weights[faces[k, j]] += weights[k] * volumes[k]
            face_per_vertex[faces[k, j]] += volumes[k]
    mv = volumes.sum()/volumes.shape[0]
    for k in range(face_per_vertex.shape[0]):
        if np.fabs(face_per_vertex[k]) > 1e-10*mv:
            vertex_weights[k] /= face_per_vertex[k]
        # else:
        #     if checkOrientation:
        #         print('Isolated vertex', k, face_per_vertex[k])

    return centers, volumes, normals, vertex_weights


@jit(nopython=True)
def get_edges_(faces, parallel=True):
    int64 = "int64"
    nv = faces.max() + 1
    dim = faces.shape[1] - 1
    nf = faces.shape[0]

    #edg0 = csr_matrix(shape=shp, dtype=int64)
    edgi = dict()
    ne = 0
    #edg0 = np.zeros(shp, dtype=int64)
    edges = np.zeros(((dim+1)*nf, dim), dtype=int64)
    for k in range(faces.shape[0]):
        for j in range(dim+1):
            indx = list(faces[k, :])
            indx.remove(faces[k,j])
            indx.sort()
            inds = ''
            for s in indx:
                inds += str(s) + '-'
            #indx = str(indx)
            if inds not in edgi:
                edgi[inds] = ne
                edges[ne, :] = indx
                ne += 1
            # if dim==2:
            #     edg0 |= {(indx[0], indx[1])}
            #     #edg0[indx[0], indx[1]] = 1
            # else:
            #     edg0 |= {(indx[0], indx[1], indx[2])}
                #edg0[indx[0], indx[1], indx[2]] = 1

    #J = np.nonzero(edg0)
    # ne = J[0].shape[0]
    # ne = len(edg0)
    edges = edges[:ne,:]
    # edges = np.zeros((ne, dim), dtype=int64)
    # for k,I in enumerate(edgi):
    #     #for j in range(dim):
    #     edges[k, :] = I

    #print(edges.shape, edges.max(), faces.shape, faces.max())
    # edgi = dict()
    # ne = 0
    # # edgi = csr_matrix(shp, dtype=int64)
    # #edgi = np.zeros((nv,)*dim, dtype=int64)
    # for k in range(ne):
    #     if dim == 2:
    #         edgi[edges[k, 0], edges[k, 1]] = k
    #     else:
    #         edgi[edges[k, 0], edges[k, 1], edges[k,2]] = k

    edgesOfFaces = np.zeros(faces.shape, dtype=int64)
    for k in range(faces.shape[0]):
        for j in range(dim+1):
            indx = list(faces[k, :])
            indx.remove(faces[k,j])
            indx.sort()
            inds = ''
            for s in indx:
                inds += str(s) + '-'
            # indx = str(indx)
            #if dim==2:
            edgesOfFaces[k,j] = edgi[inds] #edg0[indx[0], indx[1]]
            #else:
            #    edgesOfFaces[k,j] = edg0[indx[0], indx[1], indx[2]]

    facesOfEdges = - np.ones((ne, 2), dtype=int64)
    if dim == 2:
        for k in range(faces.shape[0]):
            i0 = faces[k, 0]
            i1 = faces[k, 1]
            i2 = faces[k, 2]
            for f in ([i0, i1], [i1, i2], [i2, i0]):
                kk = edgi[str(min(f[0], f[1]))+'-'+str(max(f[0],f[1])) +'-']
                #kk = edgi[str([min(f[0], f[1]), max(f[0],f[1])])]
                if facesOfEdges[kk, 0] >= 0:
                    facesOfEdges[kk, 1] = k
                else:
                    facesOfEdges[kk, 0] = k
    else:
        for k in range(faces.shape[0]):
            i0 = faces[k, 0]
            i1 = faces[k, 1]
            i2 = faces[k, 2]
            i3 = faces[k, 3]
            for f in ([i0, i1, i2], [i2, i1, i3], [i0, i2, i3], [i1, i0, i3]):
                f.sort()
                kk = edgi[str(f[0])+'-'+str(f[1])+'-'+str(f[2])+'-']
                if facesOfEdges[kk, 0] >= 0:
                    facesOfEdges[kk, 1] = k
                else:
                    facesOfEdges[kk, 0] = k

    bdry = np.zeros(edges.shape[0], dtype=int64)
    for k in range(edges.shape[0]):
        if facesOfEdges[k, 1] < 0:
            bdry[k] = 1
    return edges, facesOfEdges, edgesOfFaces, bdry


class Mesh:
    def __init__(self, mesh=None, weights=None, image=None, imNames=None, volumeRatio = 1000.):
        if type(mesh) in (list, tuple):
            if isinstance(mesh[0], Mesh):
                self.concatenate(mesh)
            elif type(mesh[0]) is str:
                fvl = []
                for name in mesh:
                    fvl.append(Mesh(mesh=name))
                self.concatenate(fvl)
            else:
                self.vertices = np.copy(mesh[1])
                self.faces = np.int_(np.copy(mesh[0]))
                self.dim = self.vertices.shape[1]
                self.component = np.zeros(self.faces.shape[0], dtype=int)
                if weights is None:
                    self.weights = np.ones(self.faces.shape[0], dtype=int)
                    #self.vertex_weights = np.ones(self.vertices.shape[0], dtype=int)
                elif np.isscalar(weights):
                    #self.vertex_weights = weights*np.ones(self.vertices.shape[0], dtype=int)
                    self.weights = weights*np.ones(self.faces.shape[0], dtype=int)
                else:
                    self.weights = np.copy(weights)
                if image is None:
                    self.image = np.ones((self.faces.shape[0], 1))
                    self.imNames = ['0']
                    self.imageDim = 1
                else:
                    self.image = np.copy(image)
                    self.imageDim = self.image.shape[1]
                    self.imNames = []
                    if imNames is None:
                        for k in range(self.imageDim):
                            self.imNames.append(str(k))
                    else:
                        self.imNames = imNames
                self.computeCentersVolumesNormals(checkOrientation=True)
        elif type(mesh) is str:
            self.read(mesh)
        elif issubclass(type(mesh), Mesh):
            self.vertices = np.copy(mesh.vertices)
            self.volumes = np.copy(mesh.volumes)
            self.normals = np.copy(mesh.normals)
            self.faces = np.copy(mesh.faces)
            self.centers = np.copy(mesh.centers)
            self.component = np.copy(mesh.component)
            if weights is None:
                self.weights = np.copy(mesh.weights)
                self.vertex_weights = np.copy(mesh.vertex_weights)
            else:
                self.updateWeights(weights)
            self.dim = mesh.dim
            self.image = np.copy(mesh.image)
            self.imageDim = mesh.imageDim
            self.imNames = mesh.imNames
        elif issubclass(type(mesh), Curve):
            g = GeometryBuilder()
            g.add_geometry(mesh.vertices, mesh.faces)
            mesh_info = tri.MeshInfo()
            g.set(mesh_info)
            vol = mesh.enclosedArea()
            f = tri.build(mesh_info, verbose=False, max_volume=vol/volumeRatio)
            self.vertices = np.array(f.points)
            self.faces = np.array(f.elements, dtype=int)
            self.dim = 2
            self.component = np.zeros(self.faces.shape[0], dtype=int)
            if weights is None:
                w=1
            else:
                w = weights
            self.vertex_weights = w * np.ones(self.vertices.shape[0], dtype=int)
            self.weights = w * np.ones(self.faces.shape[0], dtype=int)
            self.computeCentersVolumesNormals(checkOrientation=True)
            self.image = np.ones((self.faces.shape[0], 1))
            self.imageDim = 1
            self.imNames = ['0']
        elif issubclass(type(mesh), Surface):
            g = GeometryBuilder()
            g.add_geometry(mesh.vertices, mesh.faces)
            mesh_info = tet.MeshInfo()
            g.set(mesh_info)
            vol = mesh.surfVolume()
            f = tet.build(mesh_info, options= tet.Options(switches='q1.2/10'), verbose=True, max_volume=vol/volumeRatio)
            self.vertices = np.array(f.points)
            self.faces = np.array(f.elements, dtype=int)
            self.dim = 3
            self.component = np.zeros(self.faces.shape[0], dtype=int)
            if weights is None:
                w=1
            else:
                w = weights
            #self.vertex_weights = w * np.ones(self.vertices.shape[0], dtype=int)
            self.weights = w * np.ones(self.faces.shape[0], dtype=int)
            self.computeCentersVolumesNormals()
            self.image = np.ones((self.faces.shape[0], 1))
            self.imageDim = 1
            self.imNames = ['0']
            self.computeCentersVolumesNormals(checkOrientation=True)
            print(f'Mesh: {self.vertices.shape[0]} vertices, {self.faces.shape[0]} cells')
        else:
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.volumes = np.empty(0)
            self.normals = np.empty(0)
            self.component = np.empty(0)
            self.weights = np.empty(0)
            self.vertex_weights = np.empty(0)
            self.image = np.empty(0)
            self.imageDim = 0
            self.imNames = []
            self.dim = 0

        self.edges = None
        self.facesOfEdges = None
        self.edgesOfFaces = None
        self.bdry_indices = None
        self.bdry = None

    def read(self, filename):
        (mainPart, ext) = os.path.splitext(filename)
        if ext=='.vtk':
            self.readVTK(filename)
        else:
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.volumes = np.empty(0)
            self.normals = np.empty(0)
            self.component = np.empty(0)
            self.weights = np.empty(0)
            self.vertex_weights = np.empty(0)
            self.dim = 0
            self.imNames = None
            raise NameError('Unknown Mesh Extension: '+filename)

    # face centers and area weighted normal
    def computeCentersVolumesNormals(self, checkOrientation= False):
        self.centers, self.volumes, self.normals, self.vertex_weights =\
            computeCentersVolumesNormals__(self.faces, self.vertices, self.weights, checkOrientation=checkOrientation)

    def updateWeights(self, w0):
        if np.isscalar(w0):
            self.weights = w0 * np.ones(self.faces.shape[0])
        else:
            self.weights = np.copy(w0)
        self.vertex_weights = np.zeros(self.vertices.shape[0])
        face_per_vertex = np.zeros(self.vertices.shape[0], dtype=int)
        for k in range(self.faces.shape[0]):
            for j in range(self.faces.shape[1]):
                self.vertex_weights[self.faces[k, j]] += self.weights[k] * self.volumes[k]
                face_per_vertex[self.faces[k, j]] += self.volumes[k]
        mv = self.volumes.sum()/self.volumes.shape[0]
        for k in range(face_per_vertex.shape[0]):
            if face_per_vertex[k] > 1e-10*mv:
                self.vertex_weights[k] /= face_per_vertex[k]

    def shrinkTriangles(self, ratio=0.5):
        newv = np.zeros((self.faces.shape[0]*self.faces.shape[1], self.vertices.shape[1]))
        newf = np.zeros(self.faces.shape)
        for k in range(self.faces.shape[0]):
            for j in range(self.faces.shape[1]):
                newf[k,j] = self.faces.shape[1]*k + j
                newv[3*k+j, :] = self.centers[k,:] + ratio*(self.vertices[self.faces[k,j], :] - self.centers[k,:])
        neww = self.weights/(ratio** self.dim)
        newm = Mesh(mesh=(newf, newv), weights=neww, image=self.image, imNames=self.imNames)
        return newm

    def rescaleUnits(self, scale):
        self.weights /= scale**self.dim
        self.updateVertices(self.vertices*scale)
        #self.updateWeights(self.weights/())
        #self.updateWeights(self.weights/(scale**self.dim))


    # modify vertices without toplogical change
    def updateVertices(self, x0, checkOrientation=False):
        self.vertices = np.copy(x0)
        if self.bdry is not None:
            self.bdry.updateVertices(x0[self.bdry_indices])
        self.computeCentersVolumesNormals(checkOrientation=checkOrientation)

    def updateImage(self, img):
        self.image = np.copy(img)
        self.imageDim = img.shape[1]

    def computeSparseMatrices(self):
        self.v2f0 = scp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,0]))).transpose(copy=False)
        self.v2f1 = scp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,1]))).transpose(copy=False)
        self.v2f2 = scp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,2]))).transpose(copy=False)

    def computeVertexVolume(self):
        V = self.vertices
        F = self.faces
        nv = V.shape[0]
        nf = F.shape[0]
        AV = np.zeros(nv)
        df = F.shape[1]
        vol = self.volumes/df
        for k in range(nf):
            for j in range(df):
                AV[F[k,j]] += vol[k]
        return AV

    def summary(self):
        out = f'Number of vertices: {self.vertices.shape[0]}\n'
        out += f'Number of simplices: {self.faces.shape[0]}\n'
        out += f'Min-Max-Mean volume: {self.volumes.min():.6f} {self.volumes.max():.6f} {self.volumes.mean():.6f}\n'
        out += f'Min-Max-Mean weight: {self.weights.min():.6f} {self.weights.max():.6f} {self.weights.mean():.6f}\n'
        out += f'Min-Max-Mean vertex weight: {self.vertex_weights.min():.6f} {self.vertex_weights.max():.6f} {self.vertex_weights.mean():.6f}\n'
        vw = self.weights * self.volumes
        out += f'Min-Max-Mean cells per simplex: {vw.min():.4f} {vw.max():.4f} {vw.mean():.4f}\n'
        return out

    # def computeVertexNormals(self):
    #     self.computeCentersAreas()
    #     normals = np.zeros(self.vertices.shape)
    #     F = self.faces
    #     for k in range(F.shape[0]):
    #         normals[F[k,0]] += self.surfel[k]
    #         normals[F[k,1]] += self.surfel[k]
    #         normals[F[k,2]] += self.surfel[k]
    #     af = np.sqrt( (normals**2).sum(axis=1))
    #     #logging.info('min area = %.4f'%(af.min()))
    #     normals /=af.reshape([self.vertices.shape[0],1])
    #
    #     return normals

    # def computeAreaWeightedVertexNormals(self):
    #     self.computeCentersAreas()
    #     normals = np.zeros(self.vertices.shape)
    #     F = self.faces
    #     for k in range(F.shape[0]):
    #         normals[F[k,0]] += self.surfel[k]
    #         normals[F[k,1]] += self.surfel[k]
    #         normals[F[k,2]] += self.surfel[k]
    #
    #     return normals
         

    # Computes edges from vertices/faces
    def getEdges(self):
        self.edges, self.facesOfEdges, self.edgesOfFaces, bdry = get_edges_(self.faces)
        I = np.nonzero(bdry)[0]
        J = np.zeros(self.vertices.shape[0], dtype=int)
        for k in I:
            for j in range(self.edges.shape[1]):
                J[self.edges[k,j]] = 1

        J = np.nonzero(J)[0]
        self.bdry_indices = J
        newindx = np.zeros(self.vertices.shape[0], dtype=int)
        newindx[J] = np.arange(len(J))
        V = self.vertices[J,:]
        F = newindx[self.edges[I,:]]
        if self.dim == 3:
            self.bdry = Surface(surf=(F,V))
        else:
            self.bdry = Curve(curve=(F,V))


    def toPolyData(self):
        if gotVTK:
            points = vtkPoints()
            for k in range(self.vertices.shape[0]):
                if self.dim == 3:
                    points.InsertNextPoint(self.vertices[k,0], self.vertices[k,1], self.vertices[k,2])
                else:
                    points.InsertNextPoint(self.vertices[k, 0], self.vertices[k, 1], 0)
            polys = vtkCellArray()
            df = self.faces.shape[1]
            for k in range(self.faces.shape[0]):
                polys.InsertNextCell(df)
                for kk in range(df):
                    polys.InsertCellPoint(self.faces[k,kk])
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            return polydata
        else:
            raise Exception('Cannot run toPolyData without VTK')

    def fromUnstructuredGridData(self, g, scales=None):
        npoints = int(g.GetNumberOfPoints())
        nfaces = int(g.GetNumberOfCells())
        logging.info('Dimensions: %d %d %d' %(npoints, nfaces, g.GetNumberOfCells()))
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(g.GetPoint(kk))
            #print kk, V[kk]
            #print kk, np.array(g.GetPoint(kk))
        F = np.zeros([nfaces, 4], dtype=int)
        for kk in range(g.GetNumberOfCells()):
            c = g.GetCell(kk)
            npt = c.GetNumberOfPoints()
            if kk == 0:
                self.dim = npt -1
            for ll in range(npt):
                F[kk,ll] = c.GetPointId(ll)
                #print kk, gf, F[gf]

        if self.dim == 2:
            F = F[:,:3]
            V = V[:,:2]
                #self.vertices = np.multiply(data.shape-V-1, scales)
        if scales is None:
            self.vertices = V
        else:
            self.vertices = V * scales
        self.faces = F
        self.component = np.zeros(self.faces.shape[0], dtype = int)
        self.weights = np.ones(self.faces.shape[0])
        self.computeCentersVolumesNormals()



    # def subDivide(self, number=1):
    #     if gotVTK:
    #         polydata = self.toPolyData()
    #         subdivisionFilter = vtkLinearSubdivisionFilter()
    #         if vtkVersion.GetVTKMajorVersion() >= 6:
    #             subdivisionFilter.SetInputData(polydata)
    #         else:
    #             subdivisionFilter.SetInput(polydata)
    #         subdivisionFilter.SetNumberOfSubdivisions(number)
    #         subdivisionFilter.Update()
    #         self.fromPolyData(subdivisionFilter.GetOutput())
    #     else:
    #         raise Exception('Cannot run subDivide without VTK')
                        
            
    # def Simplify(self, target=1000.0, deciPro=False):
    #     if gotVTK:
    #         polydata = self.toPolyData()
    #         red = 1 - min(np.float(target) / polydata.GetNumberOfPoints(), 1)
    #         if deciPro:
    #             dc = vtkDecimatePro()
    #             if vtkVersion.GetVTKMajorVersion() >= 6:
    #                 dc.SetInputData(polydata)
    #             else:
    #                 dc.SetInput(polydata)
    #             dc.SetTargetReduction(red)
    #             dc.PreserveTopologyOn()
    #             dc.Update()
    #         else:
    #             dc = vtkQuadricDecimation()
    #             dc.SetTargetReduction(red)
    #             if vtkVersion.GetVTKMajorVersion() >= 6:
    #                 dc.SetInputData(polydata)
    #             else:
    #                 dc.SetInput(polydata)
    #             dc.Update()
    #         g = dc.GetOutput()
    #         self.fromPolyData(g)
    #         z= self.surfVolume()
    #         if (z > 0):
    #             self.flipFaces()
    #             logging.info('flipping volume {0:f} {1:f}'.format(z, self.surfVolume()))
    #     else:
    #         raise Exception('Cannot run Simplify without VTK')

    def flipFaces(self):
        if self.dim == 2:
            self.faces = self.faces[:, [0,2,1]]
        else:
            self.faces = self.faces[:, [0, 1, 3, 2]]
        self.computeCentersVolumesNormals()



    #
    # def plot(self, fig=1, ec = 'b', fc = 'r', al=.5, lw=1, azim = 100, elev = 45, setLim=True, addTo = False):
    #     f = plt.figure(fig)
    #     plt.clf()
    #     ax = Axes3D(f, azim=azim, elev=elev)
    #     self.addToPlot(ax, ec=ec, fc=fc, al=al, setLim=setLim)
    #     plt.axis('off')
    #
    #
    #
    # def addToPlot(self, ax, ec = 'b', fc = 'r', al=.5, lw=1, setLim=True):
    #     x = self.vertices[self.faces[:,0],:]
    #     y = self.vertices[self.faces[:,1],:]
    #     z = self.vertices[self.faces[:,2],:]
    #     a = np.concatenate([x,y,z], axis=1)
    #     poly = [ [a[i,j*3:j*3+3] for j in range(3)] for i in range(a.shape[0])]
    #     tri = Poly3DCollection(poly, alpha=al, linewidths=lw)
    #     tri.set_edgecolor(ec)
    #     ls = LightSource(90, 45)
    #     fc = np.array(colors.to_rgb(fc))
    #     normals = self.surfel/np.sqrt((self.surfel**2).sum(axis=1))[:,None]
    #     shade = ls.shade_normals(normals, fraction=1.0)
    #     fc = fc[None, :] * shade[:, None]
    #     tri.set_facecolor(fc)
    #     ax.add_collection3d(tri)
    #     xlim = [self.vertices[:,0].min(),self.vertices[:,0].max()]
    #     ylim = [self.vertices[:,1].min(),self.vertices[:,1].max()]
    #     zlim = [self.vertices[:,2].min(),self.vertices[:,2].max()]
    #     if setLim:
    #         ax.set_xlim(xlim[0], xlim[1])
    #         ax.set_ylim(ylim[0], ylim[1])
    #         ax.set_zlim(zlim[0], zlim[1])
    #     return [xlim, ylim, zlim]



    # Computes surface volume
    def meshVolume(self):
        return self.volumes.sum()

    def toImage(self, resolution=1, background = 0, margin = 10, index = None, bounds=None):
        interp = LinearNDInterpolator(self.centers, self.weights[:, None]*self.image, fill_value=background)

        if bounds is None:
            imin = self.vertices.min(axis=0) - margin
            imax = self.vertices.max(axis=0) + margin
        else:
            imin = bounds[0]
            imax = bounds[1]

        imDim = np.ceil((imax - imin)/resolution).astype(int)

        if np.prod(imDim) * self.image.shape[1] > 1e7:
            print('Image to big to create', imDim, self.image.shape[1])
            return

        if self.dim > 3:
            print('Image cannot be created: dimensions too large', self.dim)
            return

        X = np.linspace(imin[0] , imax[0], imDim[0])
        if self.dim == 2:
            Y = np.linspace(imin[1], imax[1], imDim[1])
            X, Y = np.meshgrid(X, Y)
            outIm = interp(X,Y)
            outIm = np.flip(outIm, axis=0)
        elif self.dim == 3:
            Y = np.linspace(imin[1] - margin, imax[1] + margin, imDim[1])
            Z = np.linspace(imin[2] - margin, imax[2] + margin, imDim[2])
            X, Y, Z = np.meshgrid(X, Y, Z)
            outIm = interp(X,Y,Z)
        else:
            outIm = interp(X)

        # outIm = np.full(s, background, dtype=float)
        # I = np.ceil((self.centers - imin[None, :])/resolution).astype(int) + margin
        # for j in range(self.faces.shape[0]):
        #     if self.dim == 2:
        #         outIm[I[j, 0], I[j, 1], :] += self.volumes[j] * self.image[j,:]
        #     elif self.dim == 3:
        #         outIm[I[j, 0], I[j, 1], I[j,2], :] += self.volumes[j] * self.image[j,:]
        # outIm /= resolution**self.dim

        if index is None:
            return outIm
        else:
            return outIm[...,index]




    def saveVTK_OLD(self, fileName, scalars = None, normals = None, tensors=None, scal_name='scalars',
                vectors=None, vect_name='vectors', fields=None, field_name='fields'):
        vf = vtkFields()
        #print scalars
        vf.scalars.append('weights')
        vf.scalars.append(self.weights)
        if not (scalars is None):
            vf.scalars.append(scal_name)
            vf.scalars.append(scalars)
        if not (vectors is None):
            vf.vectors.append(vect_name)
            vf.vectors.append(vectors)
        if not (normals is None):
            vf.normals.append('normals')
            vf.normals.append(normals)
        if not (tensors is None):
            vf.tensors.append('tensors')
            vf.tensors.append(tensors)
        vf.fields.append('image')
        vf.fields.append(self.image)
        if not (fields is None):
            vf.tensors.append(field_name)
            vf.tensors.append(fields)
        self.saveVTK2(fileName, vf)

    # Saves in .vtk format
    def saveVTK(self, fileName, vtk_fields = ()):
        F = self.faces
        V = self.vertices

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nMesh Data\nASCII\nDATASET UNSTRUCTURED_GRID\n')
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n')
                for kk in range(self.dim):
                    fvtkout.write(f'{V[ll,kk]: f} ')
                if self.dim == 2:
                    fvtkout.write('0')
            fvtkout.write('\nCELLS {0:d} {1:d}'.format(F.shape[0], (self.dim+2)*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write(f'\n{self.dim+1} ')
                for kk in range(self.dim+1):
                    fvtkout.write(f'{F[ll,kk]: d} ')

            if self.dim == 2:
                ctype = 5
            else:
                ctype = 10
            fvtkout.write('\nCELL_TYPES {0:d}'.format(F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write(f'\n{ctype} ')

            cell_data = False
            point_data = False
            for v in vtk_fields:
                if v.data_type == 'CELL_DATA':
                    cell_data = True
                    if not 'image' in v.fields.keys():
                        v.fields['image'] = self.image
                    if not 'weights' in v.scalars.keys():
                        v.scalars['weights'] = self.weights
                    if not 'volumes' in v.scalars.keys():
                        v.scalars['volumes'] = self.volumes
                    v.write(fvtkout)
                elif v.data_type == 'POINT_DATA':
                    point_data = True
                    if not 'vertex_weights' in v.scalars.keys():
                        v.scalars['vertex_weights'] = self.vertex_weights
                    v.write(fvtkout)
            if not cell_data:
                v = vtkFields('CELL_DATA', self.faces.shape[0], fields = {'image':self.image},
                              scalars = {'weights':self.weights, 'volumes':self.volumes})
                v.write(fvtkout)
            if not point_data:
                v = vtkFields('POINT_DATA', self.vertices.shape[0], scalars = {'vertex_weights':self.vertex_weights})
                v.write(fvtkout)

            # fvtkout.write(('\nCELL_DATA {0: d}').format(F.shape[0]))
            # fvtkout.write('\nSCALARS labels int 1\nLOOKUP_TABLE default')
            # for ll in range(F.shape[0]):
            #     fvtkout.write('\n {0:d}'.format(self.component[ll]))
            # fvtkout.write('\nSCALARS weights float 1\nLOOKUP_TABLE default')
            # for ll in range(F.shape[0]):
            #     fvtkout.write('\n {0:d}'.format(self.weights[ll]))
            # if 'cell_data' in datakeys:
            #     dcell = vtkFields['cell_data']
            #     kcell = dcell.keys()
            #     if 'scalars' in kcell:
            #         for name in dcell['scalars'].keys():
            #             fvtkout.write(f'\nSCALARS {name} float 1\nLOOKUP_TABLE default')
            #             for ll in range(F.shape[0]):
            #                 fvtkout.write('\n {0:d}'.format(dcell['scalar'][ll]))
            #
            # fvtkout.write(f'\nFIELD IMAGE 1')
            # fvtkout.write(f'\nimage {self.imageDim} {F.shape[0]} float')
            # for ll in range(F.shape[0]):
            #     fvtkout.write('\n')
            #     for lll in range(self.imageDim):
            #         fvtkout.write(f'{self.image[ll, lll]:.4f} ')
            #
            #     if 'point_data' in datakeys:
            #         fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            #
            #     wrote_pd_hdr = False
            #     if len(vtkFields.scalars) > 0:
            #         if not wrote_pd_hdr:
            #             fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            #             wrote_pd_hdr = True
            #         nf = len(vtkFields.scalars)//2
            #         for k in range(nf):
            #             fvtkout.write('\nSCALARS '+ vtkFields.scalars[2*k] +' float 1\nLOOKUP_TABLE default')
            #             for ll in range(V.shape[0]):
            #                 #print scalars[ll]
            #                 fvtkout.write('\n {0: .5f}'.format(vtkFields.scalars[2*k+1][ll]))
            #     if len(vtkFields.vectors) > 0:
            #         if not wrote_pd_hdr:
            #             fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            #             wrote_pd_hdr = True
            #         nf = len(vtkFields.vectors)//2
            #         for k in range(nf):
            #             fvtkout.write('\nVECTORS '+ vtkFields.vectors[2*k] +' float')
            #             vectors = vtkFields.vectors[2*k+1]
            #             for ll in range(V.shape[0]):
            #                 fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
            #     if len(vtkFields.normals) > 0:
            #         if not wrote_pd_hdr:
            #             fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            #             wrote_pd_hdr = True
            #         nf = len(vtkFields.normals)//2
            #         for k in range(nf):
            #             fvtkout.write('\nNORMALS '+ vtkFields.normals[2*k] +' float')
            #             vectors = vtkFields.normals[2*k+1]
            #             for ll in range(V.shape[0]):
            #                 fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
            #     if len(vtkFields.tensors) > 0:
            #         if not wrote_pd_hdr:
            #             fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            #             wrote_pd_hdr = True
            #         nf = len(vtkFields.tensors)//2
            #         for k in range(nf):
            #             fvtkout.write('\nTENSORS '+ vtkFields.tensors[2*k] +' float')
            #             tensors = vtkFields.tensors[2*k+1]
            #             for ll in range(V.shape[0]):
            #                 for kk in range(2):
            #                     fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(tensors[ll, kk, 0], tensors[ll, kk, 1], tensors[ll, kk, 2]))
            #     if len(vtkFields.fields) > 0:
            #         if not wrote_pd_hdr:
            #             fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            #             wrote_pd_hdr = True
            #         nf = len(vtkFields.tensors)//2
            #         for k in range(nf):
            #             fvtkout.write('\nTENSORS '+ vtkFields.tensors[2*k] +' float')
            #             tensors = vtkFields.tensors[2*k+1]
            #             for ll in range(V.shape[0]):
            #                 for kk in range(2):
            #                     fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(tensors[ll, kk, 0], tensors[ll, kk, 1], tensors[ll, kk, 2]))
            #     fvtkout.write('\n')
            #

    def save(self, fileName, vtkFields = ()):
        self.saveVTK(fileName, vtk_fields=vtkFields)

    # Reads .vtk file
    def readVTK(self, fileName):
        if gotVTK:
            #u = vtkPolyDataReader()
            u = vtkUnstructuredGridReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            w = v.GetCellData().GetScalars('weights')
            lab = v.GetCellData().GetScalars('labels')
            image = v.GetCellData().GetArray('image')
            #print v
            self.fromUnstructuredGridData(v)
            npoints = self.vertices.shape[0]
            nfaces = self.faces.shape[0]
            # V = np.zeros([npoints, 3])
            # for kk in range(npoints):
            #     V[kk, :] = np.array(v.GetPoint(kk))

            if lab:
                Lab = np.zeros(nfaces, dtype=int)
                for kk in range(nfaces):
                    Lab[kk] = lab.GetTuple(kk)[0]
            else:
                Lab = np.zeros(nfaces, dtype=int)

            if image:
                nt = image.GetNumberOfTuples()
                if nt==nfaces:
                    self.imageDim = image.GetNumberOfComponents()
                    IM = np.zeros((nfaces, self.imageDim))
                    kj=0
                    for k in range(nfaces):
                        for j in range(self.imageDim):
                            IM[k,j] = image.GetValue(kj)
                            kj += 1
                else:
                    IM = np.ones(nfaces)
            else:
                IM = np.ones(nfaces)
            if w:
                W = np.zeros(nfaces)
                for kk in range(nfaces):
                    W[kk] = w.GetTuple(kk)[0]
            else:
                W = np.ones(nfaces)

            # F = np.zeros([nfaces, 3])
            # for kk in range(nfaces):
            #     c = v.GetCell(kk)
            #     for ll in range(3):
            #         F[kk,ll] = c.GetPointId(ll)
            #
            # self.vertices = V
            self.weights = W
            self.image = IM
            #self.faces = np.int_(F)
            self.computeCentersVolumesNormals()
            self.imNames = None
            # xDef1 = self.vertices[self.faces[:, 0], :]
            # xDef2 = self.vertices[self.faces[:, 1], :]
            # xDef3 = self.vertices[self.faces[:, 2], :]
            # self.centers = (xDef1 + xDef2 + xDef3) / 3
            # self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
            self.component = Lab #np.zeros(self.faces.shape[0], dtype=int)
        else:
            raise Exception('Cannot run readVTK without VTK')
    

    def concatenate(self, fvl):
        nv = 0
        nf = 0
        for fv in fvl:
            nv += fv.vertices.shape[0]
            nf += fv.faces.shape[0]
        self.dim = self.fvl[0].dim
        self.vertices = np.zeros([nv,self.dim])
        self.weights = np.zeros(nv)
        self.faces = np.zeros([nf,self.dim+1], dtype='int')
        self.component = np.zeros(nf, dtype='int')

        nv0 = 0
        nf0 = 0
        c = 0
        for fv in fvl:
            nv = nv0 + fv.vertices.shape[0]
            nf = nf0 + fv.faces.shape[0]
            self.vertices[nv0:nv, :] = fv.vertices
            self.weights[nv0:nv] = fv.weights
            self.faces[nf0:nf, :] = fv.faces + nv0
            self.component[nf0:nf] = fv.component + c
            nv0 = nv
            nf0 = nf
            c = self.component[:nf].max() + 1
        self.computeCentersVolumesNormals()

    # def connected_components(self, split=False):
    #     self.getEdges()
    #     N = self.edges.max()+1
    #     A = csr_matrix((np.ones(self.edges.shape[0]), (self.edges[:,0], self.edges[:,1])), shape=(N,N))
    #     nc, labels = connected_components(A, directed=False)
    #     self.component = labels[self.faces[:,0]]
    #     logging.info(f'Found {nc} connected components')
    #     if split:
    #         return self.split_components(labels)
    #
    # def split_components(self, labels):
    #     nc = labels.max() + 1
    #     res = []
    #     for i in range(nc):
    #         J = np.nonzero(labels == i)[0]
    #         V = self.vertices[J,:]
    #         w = self.weights[J]
    #         newI = -np.ones(self.vertices.shape[0], dtype=int)
    #         newI[J] = np.arange(0, J.shape[0])
    #         F = newI[self.faces]
    #         I = np.amax(F, axis=1) >= 0
    #         F = F[I, :]
    #         res.append(Surface(surf=(F,V), weights=w))
    #     return res
    #
    # def extract_components(self, comp=None, comp_info=None):
    #     #labels = np.zeros(self.vertices.shape[0], dtype = int)
    #     #for j in range(self.faces.shape[1]):
    #      #   labels[self.faces[:,i]] = self.component
    #
    #     #print('extracting components')
    #     if comp_info is not None:
    #         F, J, E, FE, EF = comp_info
    #     elif comp is not None:
    #         if self.edges is None:
    #             F, J, E, FE, EF = extract_components_(comp, self.vertices.shape[0], self.faces, self.component,
    #                                                         edge_info = None)
    #         else:
    #             F, J, E, FE, EF = extract_components_(comp, self.vertices.shape[0], self.faces, self.component,
    #                                                         edge_info = (self.edges, self.edgesOfFaces, self.facesOfEdges))
    #     else:
    #         res = Surface
    #         J = np.zeros(self.vertices.shape[0], dtype=bool)
    #         return res, J
    #
    #     V = self.vertices[J,:]
    #     w = self.weights[J]
    #     res = Surface(surf=(F,V), weights=w)
    #     if self.edges is not None:
    #         res.edges = E
    #         res.edgesOfFaces = FE
    #         res.facesOfEdges = EF
    #
    #     #print(f'End of extraction: vertices: {res.vertices.shape[0]} faces: {res.faces.shape[0]}')
    #     return res, J

    # def normGrad(self, phi):
    #     v1 = self.vertices[self.faces[:,0],:]
    #     v2 = self.vertices[self.faces[:,1],:]
    #     v3 = self.vertices[self.faces[:,2],:]
    #     l1 = ((v2-v3)**2).sum(axis=1)
    #     l2 = ((v1-v3)**2).sum(axis=1)
    #     l3 = ((v1-v2)**2).sum(axis=1)
    #     phi1 = phi[self.faces[:,0],:]
    #     phi2 = phi[self.faces[:,1],:]
    #     phi3 = phi[self.faces[:,2],:]
    #     a = 4*np.sqrt((self.surfel**2).sum(axis=1))
    #     u = l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1)
    #     res = (u/a).sum()
    #     return res
    #
    # def laplacian(self, phi, weighted=False):
    #     res = np.zeros(phi.shape)
    #     v1 = self.vertices[self.faces[:,0],:]
    #     v2 = self.vertices[self.faces[:,1],:]
    #     v3 = self.vertices[self.faces[:,2],:]
    #     l1 = (((v2-v3)**2).sum(axis=1))[...,np.newaxis]
    #     l2 = (((v1-v3)**2).sum(axis=1))[...,np.newaxis]
    #     l3 = (((v1-v2)**2).sum(axis=1))[...,np.newaxis]
    #     phi1 = phi[self.faces[:,0],:]
    #     phi2 = phi[self.faces[:,1],:]
    #     phi3 = phi[self.faces[:,2],:]
    #     a = 8*(np.sqrt((self.surfel**2).sum(axis=1)))[...,np.newaxis]
    #     r1 = (l1 * (phi2 + phi3-2*phi1) + (l2-l3) * (phi2-phi3))/a
    #     r2 = (l2 * (phi1 + phi3-2*phi2) + (l1-l3) * (phi1-phi3))/a
    #     r3 = (l3 * (phi1 + phi2-2*phi3) + (l2-l1) * (phi2-phi1))/a
    #     for k,f in enumerate(self.faces):
    #         res[f[0],:] += r1[k,:]
    #         res[f[1],:] += r2[k,:]
    #         res[f[2],:] += r3[k,:]
    #     if weighted:
    #         av = self.computeVertexArea()
    #         return res/av[0]
    #     else:
    #         return res
    #
    # def diffNormGrad(self, phi):
    #     res = np.zeros((self.vertices.shape[0],phi.shape[1]))
    #     v1 = self.vertices[self.faces[:,0],:]
    #     v2 = self.vertices[self.faces[:,1],:]
    #     v3 = self.vertices[self.faces[:,2],:]
    #     l1 = (((v2-v3)**2).sum(axis=1))
    #     l2 = (((v1-v3)**2).sum(axis=1))
    #     l3 = (((v1-v2)**2).sum(axis=1))
    #     phi1 = phi[self.faces[:,0],:]
    #     phi2 = phi[self.faces[:,1],:]
    #     phi3 = phi[self.faces[:,2],:]
    #     #a = ((self.surfel**2).sum(axis=1))
    #     a = 2*np.sqrt((self.surfel**2).sum(axis=1))
    #     u = l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1)
    #     #u = (2*u/a**2)[...,np.newaxis]
    #     u = (u/a**3)[...,np.newaxis]
    #     a = a[...,np.newaxis]
    #
    #     r1 = - u * np.cross(v2-v3,self.surfel) + 2*((v1-v3) *(((phi3-phi2)*(phi1-phi2)).sum(axis=1))[:,np.newaxis]
    #         + (v1-v2)*(((phi1-phi3)*(phi2-phi3)).sum(axis=1)[:,np.newaxis]))/a
    #     r2 = - u * np.cross(v3-v1,self.surfel) + 2*((v2-v1) *(((phi1-phi3)*(phi2-phi3)).sum(axis=1))[:,np.newaxis]
    #         + (v2-v3)*(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis])/a
    #     r3 = - u * np.cross(v1-v2,self.surfel) + 2*((v3-v2) *(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis]
    #         + (v3-v1)*(((phi3-phi2)*(phi1-phi2)).sum(axis=1)[:,np.newaxis]))/a
    #     for k,f in enumerate(self.faces):
    #         res[f[0],:] += r1[k,:]
    #         res[f[1],:] += r2[k,:]
    #         res[f[2],:] += r3[k,:]
    #     return res/2
    #
    # def meanCurvatureVector(self):
    #     res = np.zeros(self.vertices.shape)
    #     v1 = self.vertices[self.faces[:,0],:]
    #     v2 = self.vertices[self.faces[:,1],:]
    #     v3 = self.vertices[self.faces[:,2],:]
    #     a = np.sqrt(((self.surfel**2).sum(axis=1)))
    #     a = a[...,np.newaxis]
    #
    #     r1 = - np.cross(v2-v3,self.surfel)/a
    #     r2 = - np.cross(v3-v1,self.surfel)/a
    #     r3 = - np.cross(v1-v2,self.surfel)/a
    #     for k,f in enumerate(self.faces):
    #         res[f[0],:] += r1[k,:]
    #         res[f[1],:] += r2[k,:]
    #         res[f[2],:] += r3[k,:]
    #     return res

@jit(nopython=True)
def count__(g, sp, inv):
    g_ = g
    for k in range(scp.shape[0]):
        g_[sp[k], inv[k]] += 1
    return g_

@jit(nopython=True)
def select_faces__(g, points, simplices, threshold = 1e-10):
    keepface = np.nonzero(np.fabs(g).sum(axis=1) > threshold)[0]
    newf_ = np.zeros((keepface.shape[0], simplices.shape[1]), dtype=int64)
    for k in range(keepface.shape[0]):
        for j in range(simplices.shape[1]):
            newf_[k,j] = simplices[keepface[k], j]
    keepvert = np.zeros(points.shape[0], dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(simplices.shape[1]):
            keepvert[newf_[k, j]] = 1
    keepvert = np.nonzero(keepvert)[0]
    newv = np.zeros((keepvert.shape[0], points.shape[1]))
    for k in range(keepvert.shape[0]):
        for j in range(points.shape[1]):
            newv[k,j] = points[keepvert[k], j]
    newI = - np.ones(points.shape[0], dtype=int64)
    for k in range(keepvert.shape[0]):
        newI[keepvert[k]] = k
    newf = np.zeros(newf_.shape, dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(newf_.shape[1]):
            newf[k, j] = newI[newf_[k,j]]
    # newf = newI[newf]
    g = np.copy(g[keepface, :])
    return newv, newf, g, keepface

def select_faces2__(points, simplices, threshold = 1e-10, g=None, removeBackground = True, small = 0):
    int64 = 'int'
    edges, facesOfEdges, edgesOfFaces, bdry = get_edges_(simplices)
    if g is None:
        g = np.ones((simplices.shape[0],1))
    gsum = np.fabs(g).sum(axis=1) > threshold
    N = simplices.shape[0]
    A = lil_matrix((N, N), dtype=int)
    for k in range(facesOfEdges.shape[0]):
        f0 = facesOfEdges[k,0]
        f1 = facesOfEdges[k,1]
        if f0>=0 and f1>=0 and ((gsum[f0] and gsum[f1]) or (not gsum[f0] and not gsum[f1])):
            A[f0, f1] = 1
            A[f1, f0] = 1
    nc, labels = connected_components(A, directed=False)
    m_ = Mesh(mesh=(simplices, points), weights=labels, image=gsum[:,None])
    rd = np.random.permutation(nc)
    labels = rd[labels]
    # m_.saveVTK(f'full_mesh{nc}.vtk')
    logging.info(f'found {nc} connected components')
    centers = np.zeros((simplices.shape[0], points.shape[1]))
    for j in range(simplices.shape[1]):
        centers += points[simplices[:,j], :]
    centers /= simplices.shape[1]
    if removeBackground:
        diag = centers.sum(axis=1)
        k = np.argmin(diag)
        background = labels[k]
    else:
        background = -1
    for j in range(nc):
        I = np.nonzero(labels==j)[0]
        if len(I) < small:
            labels[I] = background
    keepface = np.nonzero(np.fabs(labels - background))[0]
    newf_ = np.zeros((keepface.shape[0], simplices.shape[1]), dtype=int64)
    for k in range(keepface.shape[0]):
        for j in range(simplices.shape[1]):
            newf_[k,j] = simplices[keepface[k], j]
    keepvert = np.zeros(points.shape[0], dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(simplices.shape[1]):
            keepvert[newf_[k, j]] = 1
    keepvert = np.nonzero(keepvert)[0]
    newv = np.zeros((keepvert.shape[0], points.shape[1]))
    for k in range(keepvert.shape[0]):
        for j in range(points.shape[1]):
            newv[k,j] = points[keepvert[k], j]
    newI = - np.ones(points.shape[0], dtype=int64)
    for k in range(keepvert.shape[0]):
        newI[keepvert[k]] = k
    newf = np.zeros(newf_.shape, dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(newf_.shape[1]):
            newf[k, j] = newI[newf_[k,j]]
    # newf = newI[newf]
    g = np.copy(g[keepface, :])
    return newv, newf, g, keepface

def buildMeshFromFullListHR(x0, y0, genes, radius = 20, threshold = 1e-10):
    dx =  (x0.max()- x0.min())/20
    minx = x0.min() - dx
    maxx = x0.max() + dx
    dy =  (y0.max()- y0.min())/20
    miny = y0.min() - dy
    maxy = y0.max() + dy
    ugenes, inv = np.unique(genes, return_inverse=True)
    logging.info(f'{x0.shape[0]} input points, {ugenes.shape[0]} unique genes')

    spacing = radius/2

    ul = np.array((minx, miny))
    ur = np.array((maxx, miny))
    ll = np.array((minx, maxy))
    v0 = ur - ul
    v1 = ll - ul

    nv0 = np.sqrt((v0 ** 2).sum())
    nv1 = np.sqrt((v1 ** 2).sum())
    npt0 = int(np.ceil(nv0 / spacing))
    npt1 = int(np.ceil(nv1 / spacing))

    t0 = np.linspace(0, 1, npt0)

    t1 = np.linspace(0, 1, npt1)
    x, y = np.meshgrid(t0, t1)
    x = np.ravel(x)
    y = np.ravel(y)
    pts = ul[None, :] + x[:, None] * v0[None, :] + y[:, None] * v1[None, :]

    tri = Delaunay(pts)
    vert = np.zeros((tri.points.shape[0], 3))
    vert[:,:2] = tri.points
    centers = np.zeros((x0.shape[0], 2))
    centers[:,0] = x0
    centers[:,1] = y0

    g = np.zeros((tri.simplices.shape[0], ugenes.shape[0]))
    #logging.info('find')
    sp = tri.find_simplex(centers)
    #logging.info('count')
    g = count__(g, sp, inv)
    # for k in range(centers.shape[0]):
    #     g[sp[k], inv[k]] += 1
    logging.info(f'Creating {centers.shape[0]} faces')
    # if radius is not None:
    #     ico = twelve_vertexes(dimension=2)
    #     for j in range(12):
    #         logging.info(f'radius {j}')
    #         sp = tri.find_simplex(centers + 0.5*radius*ico[j,:])
    #         g = count__(g, sp, inv)
    #         # for k in range(centers.shape[0]):
    #         #     g[sp[k], inv[k]] += 1
    #         sp = tri.find_simplex(centers + radius * ico[j, :])
    #         g = count__(g, sp, inv)
    #         # for k in range(centers.shape[0]):
    #         #     g[sp[k], inv[k]] += 1
    #     g /= 25

    logging.info('face selection')
    #newv, newf, newg, foo = select_faces__(g, tri.points, tri.simplices, threshold=threshold)
    newv, newf, newg, foo = select_faces2__(tri.points, tri.simplices, threshold=threshold, g=g)
    # keepface = np.nonzero((g ** 2).sum(axis=1) > 1e-10)[0]
    # newf = tri.simplices[keepface, :]
    # keepvert = np.zeros(tri.points.shape[0], dtype=bool)
    # for j in range(tri.simplices.shape[1]):
    #     keepvert[newf[:, j]] = True
    # keepvert = np.nonzero(keepvert)[0]
    # newv = tri.points[keepvert, :]
    # newI = - np.ones(tri.points.shape[0], dtype=int)
    # newI[keepvert] = np.arange(newv.shape[0])
    # newf = newI[newf]
    # g = g[keepface, :]

    logging.info(f'mesh construction: {newv.shape[0]} vertices {newf.shape[0]} faces')
    fv0 = Mesh(mesh=(newf, newv), image=newg)
    # fv0.saveVTK('essaiHR.vtk')
    #fv0.image /= fv0.volumes[:, None]
    return fv0

@jit(nopython=True)
def buildImageFromFullListHR(x0, y0, genes, radius = 20.):
    dx =  (x0.max()- x0.min())/20
    minx = x0.min() - dx
    maxx = x0.max() + dx
    dy =  (y0.max()- y0.min())/20
    miny = y0.min() - dy
    maxy = y0.max() + dy
    #ugenes, inv = np.unique(genes, return_inverse=True)
    #ng = ugenes.shape[0]
    ng = genes.max() + 1

    spacing = radius/2

    ul = np.array((minx, miny))
    ur = np.array((maxx, miny))
    ll = np.array((minx, maxy))
    v0 = ur - ul
    v1 = ll - ul

    nv0 = np.sqrt((v0 ** 2).sum())
    nv1 = np.sqrt((v1 ** 2).sum())
    npt0 = int(np.ceil(nv0 / spacing))
    npt1 = int(np.ceil(nv1 / spacing))

    # t0 = np.linspace(0, 1, npt0)
    # t1 = np.linspace(0, 1, npt1)
    img = np.zeros((npt0, npt1, ng))
    ik = np.floor((x0 - minx)/spacing).astype(int64)
    jk = np.floor((y0 - miny)/spacing).astype(int64)

    for k in range(x0.shape[0]):
        img[ik[k], jk[k], genes[k]] += 1

    return img, (minx, miny, spacing)

def buildMeshFromFullList(x0, y0, genes, resolution=100, HRradius=20, HRthreshold=0.5):
    logging.info('Building High-resolution mesh')
    fvHR = buildMeshFromFullListHR(x0, y0, genes, radius=HRradius, threshold=HRthreshold)
    if np.isscalar(resolution):
        resolution = (resolution,)
    fv0 = [fvHR]
    for r in resolution:
        logging.info(f'Buiding meshes at resolution {r:.0f}')
        fv0.append(buildMeshFromCentersCounts(fvHR.centers, fvHR.image, resolution=r, weights=fvHR.volumes))
    return fv0

## Builds a mesh structure from cell centers and multivariate counts
## If radius is not none: creates small triangles around each center
def buildMeshFromCentersCounts(centers, cts, resolution=100, radius = None, weights=None, minCounts = 1e-10,
                               minComponentSize = 1,
                               threshold = None):

    if threshold is not None:
        logging.warning('buildMeshFromCentersCounts: threshold is deprecated. Use minCounts instead')
        minCounts = threshold

    ## Create extended domain
    dim = centers.shape[1]
    deltax = np.zeros(dim)
    minx = np.zeros(dim)
    maxx = np.zeros(dim)
    for i in range(dim):
        deltax[i] =  (centers[:, i].max()- centers[:, i].min())/20
        minx[i] = centers[:, i].min() - deltax[i]
        maxx[i] = centers[:, i].max() + deltax[i]
    # dy =  (centers[:, 1].max()- centers[:, 1].min())/20
    # miny = centers[:, 1].min() - dy
    # maxy = centers[:, 1].max() + dy

    if radius is None:
        #spacing = max(maxy - miny, maxx - minx) / resolution
        spacing = resolution #max(maxy - miny, maxx - minx) / resolution
    else:
        spacing = radius/2

    if weights is None:
        weights = np.ones(centers.shape[0])

    #building grid
    # ul = np.array((minx, miny))
    # ur = np.array((maxx, miny))
    # ll = np.array((minx, maxy))
    v = np.zeros((dim, dim))
    npt = np.zeros(dim, dtype=int)
    t = []
    for i in range(dim):
        v[i,i] = maxx[i] - minx[i]
        npt[i] = int(np.ceil(v[i,i] / spacing))
        t.append(np.linspace(0, 1, npt[i]))
    # v0 = ur - ul
    # v1 = ll - ul

    # nv0 = np.sqrt((v0 ** 2).sum())
    # nv1 = np.sqrt((v1 ** 2).sum())
    # npt0 = int(np.ceil(nv0 / spacing))
    # npt1 = int(np.ceil(nv1 / spacing))
    #
    # t0 = np.linspace(0, 1, npt0)
    #
    # t1 = np.linspace(0, 1, npt1)

    ntotal = npt.prod()
    allts = np.copy(t[0])
    for i in range(1, dim):
        allts = np.column_stack((np.repeat(allts, t[i].shape[0], axis=0), np.tile(t[i], allts.shape[0])))
    # x = np.zeros((ntotal, dim))
    # x, y = np.meshgrid(t[0], t[1])
    # x = np.ravel(x)
    # y = np.ravel(y)
    pts = np.zeros(allts.shape)
    pts[:,:] = minx[None, :]
    for i in range(dim):
        pts +=  np.outer(allts[:,i], v[i,:])
    # pts = ul[None, :] + x[:, None] * v0[None, :] + y[:, None] * v1[None, :]

    if dim == 2:
        cube = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float')
        tri = Delaunay(cube, qhull_options='Qc Qz')
        ns0 = tri.simplices.shape[0]
        #tri = Delaunay(cube)
        ncubes = np.prod(npt - 1)
        cubes = np.zeros((ncubes, 4), dtype=int)
        u1 = 1
        u0 = npt[1]
        template = np.array([0, u1, u0, u1 + u0], dtype=int)
        c=0
        for i0 in range(npt[0]-1):
            for i1 in range(npt[1]-1):
                cubes[c, :] = i0*u0 + i1 + template
                c += 1

        dx = pts[npt[1], 0] - pts[0, 0]
        dy = pts[1, 1] - pts[0, 1]
        alld = np.array([dx, dy])
        rk = np.floor((centers-pts[0,:])/alld[None,:]).astype(int)
        resid = (centers-pts[0,:])/alld[None, :] - rk
        sp0 = tri.find_simplex(resid)
        sp = ns0 * (rk[:, 0] * (u0-1) + rk[:, 1] * u1) + sp0

        faces = np.zeros((ns0*ncubes, 3), dtype=int)
        cc = np.arange(ncubes, dtype = int)
        for j in range(tri.simplices.shape[0]):
            x0 = tri.points[tri.simplices[j, 0], :]
            x1 = tri.points[tri.simplices[j, 1], :]
            x2 = tri.points[tri.simplices[j, 2], :]
            vol = np.cross(x1 - x0, x2 - x0)
            if vol < 0:
                k = tri.simplices[j, -2]
                tri.simplices[j, -2] = tri.simplices[j, -1]
                tri.simplices[j, -1] = k
        for j in range(ns0):
            for q in range(dim+1):
                faces[ns0 * cc + j, q] = cubes[cc, tri.simplices[j, q]]


    else:
        cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                        dtype='float')
        tri = Delaunay(cube, qhull_options='Qc Qz')
        ns0 = tri.simplices.shape[0]
        ncubes = np.prod(npt-1)
        cubes = np.zeros((ncubes, 8), dtype=int)
        u2 = 1
        u1 = npt[2]
        u0 = npt[1]*npt[2]
        uu1 = npt[2]-1
        uu0 = (npt[1]-1)*(npt[2]-1)
        template = np.array([0, u2, u1, u0, u2+u1, u2+u0, u1+u0, u2+u1+u0], dtype=int)
        c=0
        for i0 in range(npt[0]-1):
            for i1 in range(npt[1]-1):
                for i2 in range(npt[2]-1):
                    cubes[c, :] = i0*u0 + i1*u1 + i2 + template
                    c += 1
        # for c in range(ncubes):
        #     cubes[c, :] = c + template


        dx = pts[npt[1] * npt[2], 0] - pts[0, 0]
        dy = pts[npt[2], 1] - pts[0, 1]
        dz = pts[1, 2] - pts[0, 2]
        alld = np.array([dx,dy,dz])
        rk = np.floor((centers-pts[0,:])/alld[None,:]).astype(int)
        resid = (centers-pts[0,:])/alld[None, :] - rk
        sp0 = tri.find_simplex(resid)
        u2 = 1
        sp = ns0 * (rk[:, 0] * uu0 + rk[:,1] * uu1 + rk[:,2] * u2) + sp0

        faces = np.zeros((ns0*ncubes, 4), dtype=int)
        cc = np.arange(ncubes, dtype = int)
        for j in range(ns0):
            x0 = tri.points[tri.simplices[j, 0], :]
            x1 = tri.points[tri.simplices[j, 1], :]
            x2 = tri.points[tri.simplices[j, 2], :]
            x3 = tri.points[tri.simplices[j, 3], :]
            vol = ((x1-x0)*np.cross(x2 - x0, x3 - x0)).sum()
            if vol < 0:
                k = tri.simplices[j, -2]
                tri.simplices[j, -2] = tri.simplices[j, -1]
                tri.simplices[j, -1] = k
        for j in range(ns0):
            for q in range(dim+1):
                faces[ns0 * cc + j, q] = cubes[cc, tri.simplices[j, q]]



    vert = np.zeros((pts.shape[0], 3))
    vert[:,:dim] = pts

    g = np.zeros((faces.shape[0], cts.shape[1]))
    # sp = tri.find_simplex(centers)
    wgts = np.zeros(faces.shape[0])
    nc = np.zeros(faces.shape[0], dtype=int)
    for k in range(centers.shape[0]):
        g[sp[k], :] += cts[k, :]
        nc[sp[k]] += 1
        wgts[sp[k]] += weights[k]

    # if radius is not None:
    #     ico = twelve_vertexes(dimension=dim)
    #     for j in range(12):
    #         sp = tri.find_simplex(centers + 0.5*radius*ico[j,:])
    #         for k in range(centers.shape[0]):
    #             g[sp[k], :] += cts[k, :]
    #             nc[sp[k]] += 1
    #             wgts[sp[k]] += weights[k]
    #         sp = tri.find_simplex(centers + radius * ico[j, :])
    #         for k in range(centers.shape[0]):
    #             g[sp[k], :] += cts[k, :]
    #             nc[sp[k]] += 1
    #             wgts[sp[k]] += weights[k]

    ## g contains average count values per simplex
    ## wgts: number of centers per simplex
    g /= np.maximum(1e-10, nc[:, None])

    # Mesh(mesh=(faces,pts), image=g).save('meshTemp.vtk')


    ## First cleaning pass: remove background
    newv, newf, newg, keepface = select_faces2__(pts, faces, threshold=minCounts, g=g,
                                                 removeBackground=True, small = 0)
    wgts = wgts[keepface]

    ## Second cleaning pass: remove small components
    newv, newf, newg2, keepface = select_faces2__(newv, newf, threshold=minCounts, removeBackground = False,
                                                  small=minComponentSize)
    wgts = wgts[keepface]
    newg = np.copy(newg[keepface, :])
    logging.info(f'Mesh with {newv.shape[0]} vertices and {newf.shape[0]} faces; {wgts.sum()} cells')
    fv0 = Mesh(mesh=(newf, newv), image=newg, weights=wgts)
    fv0.updateWeights(wgts/fv0.volumes)
    return fv0


def buildMeshFromImageData(img, geneSet = None, resolution=25, radius = None,
                           bounding_box = (0,1, 0, 1)):

    xi = np.linspace(bounding_box[0], bounding_box[1], img.shape[0])
    yi = np.linspace(bounding_box[2], bounding_box[3], img.shape[1])
    (x, y) = np.meshgrid(yi, xi)
    ng = img.shape[2]
    cts = img.reshape((x.size, ng))
    if geneSet is not None:
        img = img[:, geneSet]
    centers = np.zeros((x.size, 2))
    centers[:, 0] = np.ravel(x) # bounding_box[0] + bounding_box[1]*x/img.shape[0]
    centers[:, 1] = np.ravel(y) #bounding_box[2] + bounding_box[3]*y/img.shape[1]
    return buildMeshFromCentersCounts(centers, cts, resolution=resolution, radius = radius)

def buildMeshFromMerfishData(fileCounts, fileData, geneSet = None, resolution=100, radius = None,
                             coordinate_columns = ('center_x', 'center_y')):
    counts = pd.read_csv(fileCounts)
    if geneSet is not None:
        counts = counts.loc[:, geneSet]
    data = pd.read_csv(fileData)
    centers = np.zeros((data.shape[0], 2))
    centers[:, 0] = data.loc[:, coordinate_columns[0]]
    centers[:, 1] = data.loc[:, coordinate_columns[1]]
    cts = counts.to_numpy()
    # minx = data.loc[:, 'min_x'].min()
    # miny = data.loc[:, 'min_y'].min()
    # maxx = data.loc[:, 'max_x'].max()
    # maxy = data.loc[:, 'max_y'].max()
    return buildMeshFromCentersCounts(centers, cts, resolution=resolution, radius = radius)

