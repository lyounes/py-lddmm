import matplotlib
import os
import numpy as np
from numba import jit, int64
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import scipy as sp
import scipy.linalg as LA
from scipy.sparse import csr_matrix
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


def det3D(x1, x2, x3):
    return (x1 * np.cross(x2, x3)).sum(axis = 1)


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


class Mesh:
    def __init__(self, mesh=None, weights=None, image=None):
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
                    self.vertex_weights = np.ones(self.vertices.shape[0], dtype=int)
                elif np.isscalar(weights):
                    self.vertex_weights = weights*np.ones(self.vertices.shape[0], dtype=int)
                    self.weights = weights*np.ones(self.faces.shape[0], dtype=int)
                else:
                    self.updateWeights(weights)
                if image is None:
                    self.image = np.ones((self.faces.shape[0], 1))
                    self.imageDim = 1
                else:
                    self.image = np.copy(image)
                    self.imageDim = self.image.shape[1]
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
        elif issubclass(type(mesh), Curve):
            g = GeometryBuilder()
            g.add_geometry(mesh.vertices, mesh.faces)
            mesh_info = tri.MeshInfo()
            g.set(mesh_info)
            vol = mesh.enclosedArea()
            f = tri.build(mesh_info, verbose=False, max_volume=vol/1000)
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
        elif issubclass(type(mesh), Surface):
            g = GeometryBuilder()
            g.add_geometry(mesh.vertices, mesh.faces)
            mesh_info = tet.MeshInfo()
            g.set(mesh_info)
            vol = mesh.surfVolume()
            f = tet.build(mesh_info, options= tet.Options(switches='q1.2/10'), verbose=True, max_volume=vol/1000)
            self.vertices = np.array(f.points)
            self.faces = np.array(f.elements, dtype=int)
            self.dim = 3
            self.component = np.zeros(self.faces.shape[0], dtype=int)
            if weights is None:
                w=1
            else:
                w = weights
            self.vertex_weights = w * np.ones(self.vertices.shape[0], dtype=int)
            self.weights = w * np.ones(self.faces.shape[0], dtype=int)
            self.computeCentersVolumesNormals()
            self.image = np.ones((self.faces.shape[0], 1))
            self.imageDim = 1
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
            self.dim = 0

        self.edges = None
        self.edgeFaces = None
        self.faceEdges = None

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
            raise NameError('Unknown Mesh Extension: '+filename)

    # face centers and area weighted normal
    def computeCentersVolumesNormals(self, checkOrientation= False):
        if self.dim == 2:
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            self.centers = (xDef1 + xDef2 + xDef3) / 3
            x12 = xDef2-xDef1
            x13 = xDef3-xDef1
            self.volumes =  np.cross(x12, x13)/2
            if checkOrientation:
                if self.volumes.min() < -1e-12:
                    if self.volumes.max() > 1e-12:
                        print('Warning: mesh has inconsistent orientation')
                    else:
                        self.faces = self.faces[:, [0,2,1]]
                        xDef2 = self.vertices[self.faces[:, 1], :]
                        xDef3 = self.vertices[self.faces[:, 2], :]
                        x12 = xDef2-xDef1
                        x13 = xDef3-xDef1
                        self.volumes =  np.cross(x12, x13)/2
            J = np.array([[0, -1], [1,0]])
            self.normals = np.zeros((3, self.faces.shape[0], 2))
            self.normals[0, :, :] = (xDef3 - xDef2) @ J.T
            self.normals[1, :, :] = (xDef1 - xDef3) @ J.T
            self.normals[2, :, :] = (xDef2 - xDef1) @ J.T
        elif self.dim == 3:
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            xDef4 = self.vertices[self.faces[:, 3], :]
            self.centers = (xDef1 + xDef2 + xDef3 + xDef4) / 4
            x12 = xDef2-xDef1
            x13 = xDef3-xDef1
            x14 = xDef4-xDef1
            self.volumes =  det3D(x12, x13, x14)/6
            if checkOrientation:
                if self.volumes.min() < -1e-12:
                    if self.volumes.max() > 1e-12:
                        print('Warning: mesh has inconsistent orientation')
                    else:
                        self.faces = self.faces[:, [0,1,3,2]]
                        xDef3 = self.vertices[self.faces[:, 2], :]
                        xDef4 = self.vertices[self.faces[:, 3], :]
                        x13 = xDef3 - xDef1
                        x14 = xDef4 - xDef1
                        self.volumes = det3D(x12, x13, x14) / 6
            self.normals = np.zeros((4, self.faces.shape[0], 3))
            self.normals[0, :, :] = np.cross(xDef4 - xDef2, xDef3 - xDef2)
            self.normals[1, :, :] = np.cross(xDef2 - xDef1, xDef4 - xDef1)
            self.normals[2, :, :] = np.cross(xDef4 - xDef1, xDef2 - xDef1)
            self.normals[3, :, :] = np.cross(xDef2 - xDef1, xDef3 - xDef1)

        self.vertex_weights = np.zeros(self.vertices.shape[0])
        face_per_vertex = np.zeros(self.vertices.shape[0], dtype=int)
        for k in range(self.faces.shape[0]):
            self.vertex_weights[self.faces[k, :]] += self.weights[k]
            face_per_vertex[self.faces[k, :]] += 1
        self.vertex_weights /= face_per_vertex

    def updateWeights(self, w0):
        self.weights = np.copy(w0)
        self.vertex_weights = np.zeros(self.vertices.shape[0])
        face_per_vertex = np.zeros(self.vertices.shape[0], dtype=int)
        for k in range(self.faces.shape[0]):
            self.vertex_weights[self.faces[k, :]] += self.weights[k]
            face_per_vertex[self.faces[k, :]] += 1
        self.vertex_weights /= face_per_vertex

    # modify vertices without toplogical change
    def updateVertices(self, x0):
        self.vertices = np.copy(x0)
        self.computeCentersVolumesNormals()

    def computeSparseMatrices(self):
        self.v2f0 = sp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,0]))).transpose(copy=False)
        self.v2f1 = sp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,1]))).transpose(copy=False)
        self.v2f2 = sp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,2]))).transpose(copy=False)

    def computeVertexVolume(self):
        # compute areas of faces and vertices
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
    # def getEdges(self):
    #     self.edges, self.edgeFaces, self.faceEdges, self.bdry = get_edges_(self.faces)


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
                    v.write(fvtkout)
                elif v.data_type == 'POINT_DATA':
                    point_data = True
                    if not 'vertex_weights' in v.scalars.keys():
                        v.scalars['vertex_weights'] = self.vertex_weights
                    v.write(fvtkout)
            if not cell_data:
                v = vtkFields('CELL_DATA', self.faces.shape[0], fields = {'image':self.image},
                              scalars = {'weights':self.weights})
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

    def save(self, fileName, vtkFields = None):
        self.saveVTK(fileName)

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
    #                                                         edge_info = (self.edges, self.faceEdges, self.edgeFaces))
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
    #         res.faceEdges = FE
    #         res.edgeFaces = EF
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



def buildMeshFromCentersCounts(centers, cts, resolution=100, radius = None):
    minx = centers[:, 0].min() - 50
    maxx = centers[:, 0].max() + 50
    miny = centers[:, 1].min() - 50
    maxy = centers[:, 1].max() + 50
    if radius is None:
        spacing = max(maxy - miny, maxx - minx) / resolution
    else:
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

    g = np.zeros((tri.simplices.shape[0], cts.shape[1]))
    sp = tri.find_simplex(centers)
    for k in range(centers.shape[0]):
        g[sp[k], :] += cts[k, :]
    if radius is not None:
        ico = twelve_vertexes(dimension=2)
        for j in range(12):
            sp = tri.find_simplex(centers + 0.5*radius*ico[j,:])
            for k in range(centers.shape[0]):
                g[sp[k], :] += cts[k, :]
            sp = tri.find_simplex(centers + radius * ico[j, :])
            for k in range(centers.shape[0]):
                g[sp[k], :] += cts[k, :]
        g /= 25

    keepface = np.nonzero((g ** 2).sum(axis=1) > 1e-10)[0]
    newf = tri.simplices[keepface, :]
    keepvert = np.zeros(tri.points.shape[0], dtype=bool)
    for j in range(tri.simplices.shape[1]):
        keepvert[newf[:, j]] = True
    keepvert = np.nonzero(keepvert)[0]
    newv = tri.points[keepvert, :]
    newI = - np.ones(tri.points.shape[0], dtype=int)
    newI[keepvert] = np.arange(newv.shape[0])
    newf = newI[newf]
    g = g[keepface, :]

    fv0 = Mesh(mesh=(newf, newv), image=g)
    fv0.image /= fv0.volumes[:, None]
    return fv0


def buildMeshFromImageData(img, geneSet = None, resolution=100, radius = None,
                           bounding_box = (0,1, 0, 1)):

    (x, y) = meshgrid(img.shape[1], img.shape[2])
    ng = img.shape[0]
    cts = img.reshape((ng,centers.shape[0])
    if geneSet is not None:
        img = img[geneSet, :]
    centers = np.zeros((x.shape[0], 2))
    centers[:, 0] = bounding_box[0] + bounding_box[1]*x/img.shape[1]
    centers[:, 1] = bounding_box[2] + bounding_box[3]*y/img.shape[2]
    buildMeshFromCentersCounts(centers, cts, resolution=resolution, radius = radius)

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
    buildMeshFromCentersCounts(centers, cts, resolution=resolution, radius = radius)

