import matplotlib
matplotlib.use("QT5Agg")
import os
import numpy as np
import scipy as sp
import scipy.linalg as LA
from skimage import measure
from vtk import vtkCellArray, vtkPoints, vtkPolyData, vtkVersion,\
    vtkLinearSubdivisionFilter, vtkQuadricDecimation,\
    vtkWindowedSincPolyDataFilter, vtkImageData, VTK_FLOAT,\
    vtkDoubleArray, vtkContourFilter, vtkPolyDataConnectivityFilter,\
    vtkCleanPolyData, vtkPolyDataReader, vtkOBJReader, vtkSTLReader,\
    vtkDecimatePro, VTK_UNSIGNED_CHAR, vtkPolyDataToImageStencil,\
    vtkImageStencil
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import diffeo
except ImportError:
    from . import diffeo
import scipy.linalg as spLA
from scipy.sparse import coo_matrix
import glob
import logging
try:
    from vtk import *
    import vtk.util.numpy_support as v2n
    gotVTK = True
except ImportError:
    v2n = None
    print('could not import VTK functions')
    gotVTK = False


class vtkFields:
    def __init__(self):
        self.scalars = [] 
        self.vectors = []
        self.normals = []
        self.tensors = []
        
# General surface class
class Surface:
    def __init__(self, surf=None, filename=None, FV = None):
        if surf == None:
            if FV == None:
                if filename == None:
                    self.vertices = np.empty(0)
                    self.centers = np.empty(0)
                    self.faces = np.empty(0)
                    self.surfel = np.empty(0)
                    self.component = np.empty(0)
                else:
                    if type(filename) is list:
                        fvl = []
                        for name in filename:
                            fvl.append(Surface(filename=name))
                        self.concatenate(fvl)
                    else:
                        self.read(filename)
            else:
                self.vertices = np.copy(FV[1])
                self.faces = np.int_(FV[0])
                self.component = np.zeros(self.faces.shape[0], dtype=int)
                self.computeCentersAreas()
        else:
            if type(surf) is list:
                self.concatenate(surf)
            else:
                self.vertices = np.copy(surf.vertices)
                self.faces = np.copy(surf.faces)
                #self.surfel = np.copy(surf.surfel)
                #self.centers = np.copy(surf.centers)
                self.component = np.copy(surf.component)
                self.computeCentersAreas()

    def read(self, filename):
        (mainPart, ext) = os.path.splitext(filename)
        if ext == '.byu':
            self.readbyu(filename)
        elif ext=='.off':
            self.readOFF(filename)
        elif ext=='.vtk':
            self.readVTK(filename)
        elif ext == '.stl':
            self.readSTL(filename)
        elif ext == '.obj':
            self.readOBJ(filename)
        else:
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.component = np.empty(0)
            self.surfel = np.empty(0)
            raise NameError('Unknown Surface Extension: '+filename)

    # face centers and area weighted normal
    def computeCentersAreas(self):
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2

    # modify vertices without toplogical change
    def updateVertices(self, x0):
        self.vertices = np.copy(x0) 
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2

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

    def computeVertexArea(self):
        # compute areas of faces and vertices
        V = self.vertices
        F = self.faces
        nv = V.shape[0]
        nf = F.shape[0]
        AF = np.zeros(nf)
        AV = np.zeros(nv)
        for k in range(nf):
            # determining if face is obtuse
            x12 = V[F[k,1], :] - V[F[k,0], :]
            x13 = V[F[k,2], :] - V[F[k,0], :]
            n12 = np.sqrt((x12**2).sum())
            n13 = np.sqrt((x13**2).sum())
            c1 = (x12*x13).sum()/(n12*n13)
            x23 = V[F[k,2], :] - V[F[k,1], :]
            n23 = np.sqrt((x23**2).sum())
            #n23 = norm(x23) ;
            c2 = -(x12*x23).sum()/(n12*n23)
            c3 = (x13*x23).sum()/(n13*n23)
            AF[k] = np.sqrt((np.cross(x12, x13)**2).sum())/2
            if (c1 < 0):
                #face obtuse at vertex 1
                AV[F[k,0]] += AF[k]/2
                AV[F[k,1]] += AF[k]/4
                AV[F[k,2]] += AF[k]/4
            elif (c2 < 0):
                #face obuse at vertex 2
                AV[F[k,0]] += AF[k]/4
                AV[F[k,1]] += AF[k]/2
                AV[F[k,2]] += AF[k]/4
            elif (c3 < 0):
                #face obtuse at vertex 3
                AV[F[k,0]] += AF[k]/4
                AV[F[k,1]] += AF[k]/4
                AV[F[k,2]] += AF[k]/2
            else:
                #non obtuse face
                cot1 = c1 / np.sqrt(1-c1**2) 
                cot2 = c2 / np.sqrt(1-c2**2) 
                cot3 = c3 / np.sqrt(1-c3**2) 
                AV[F[k,0]] += ((x12**2).sum() * cot3 + (x13**2).sum() * cot2)/8 
                AV[F[k,1]] += ((x12**2).sum() * cot3 + (x23**2).sum() * cot1)/8 
                AV[F[k,2]] += ((x13**2).sum() * cot2 + (x23**2).sum() * cot1)/8 

        for k in range(nv):
            if (np.fabs(AV[k]) <1e-10):
                logging.info('Warning: vertex {0:1} has no face; use removeIsolated'.format(k))
        #print 'sum check area:', AF.sum(), AV.sum()
        return AV, AF

    def computeVertexNormals(self):
        self.computeCentersAreas() 
        normals = np.zeros(self.vertices.shape)
        F = self.faces
        for k in range(F.shape[0]):
            normals[F[k,0]] += self.surfel[k]
            normals[F[k,1]] += self.surfel[k]
            normals[F[k,2]] += self.surfel[k]
        af = np.sqrt( (normals**2).sum(axis=1))
        #logging.info('min area = %.4f'%(af.min()))
        normals /=af.reshape([self.vertices.shape[0],1])

        return normals

    def computeAreaWeightedVertexNormals(self):
        self.computeCentersAreas() 
        normals = np.zeros(self.vertices.shape)
        F = self.faces
        for k in range(F.shape[0]):
            normals[F[k,0]] += self.surfel[k]
            normals[F[k,1]] += self.surfel[k]
            normals[F[k,2]] += self.surfel[k]

        return normals
         

    # Computes edges from vertices/faces
    def getEdges(self):
        self.edges = []
        for k in range(self.faces.shape[0]):
            for kj in (0,1,2):
                u = [self.faces[k, kj], self.faces[k, (kj+1)%3]]
                if (u not in self.edges) & (u.reverse() not in self.edges):
                    self.edges.append(u)
        self.edgeFaces = []
        for u in self.edges:
            self.edgeFaces.append([])
        for k in range(self.faces.shape[0]):
            for kj in (0,1,2):
                u = [self.faces[k, kj], self.faces[k, (kj+1)%3]]
                if u in self.edges:
                    kk = self.edges.index(u)
                else:
                    u.reverse()
                    kk = self.edges.index(u)
                self.edgeFaces[kk].append(k)
        self.edges = np.int_(np.array(self.edges))
        self.bdry = np.int_(np.zeros(self.edges.shape[0]))
        for k in range(self.edges.shape[0]):
            if len(self.edgeFaces[k]) < 2:
                self.bdry[k] = 1

    # computes the signed distance function in a small neighborhood of a shape 
    def LocalSignedDistance(self, data, value):
        d2 = 2*np.array(data >= value) - 1
        c2 = np.cumsum(d2, axis=0)
        for j in range(2):
            c2 = np.cumsum(c2, axis=j+1)
        (n0, n1, n2) = c2.shape

        rad = 3
        diam = 2*rad+1
        (x,y,z) = np.mgrid[-rad:rad+1, -rad:rad+1, -rad:rad+1]
        cube = (x**2+y**2+z**2)
        maxval = (diam)**3
        s = 3.0*rad**2
        res = d2*s
        u = maxval*np.ones(c2.shape)
        u[rad+1:n0-rad, rad+1:n1-rad, rad+1:n2-rad] = (c2[diam:n0, diam:n1, diam:n2]
        - c2[0:n0-diam, diam:n1, diam:n2] - c2[diam:n0, 0:n1-diam, diam:n2] - c2[diam:n0, diam:n1, 0:n2-diam] 
        + c2[0:n0-diam, 0:n1-diam, diam:n2] + c2[diam:n0, 0:n1-diam, 0:n2-diam] + c2[0:n0-diam, diam:n1, 0:n2-diam]
        - c2[0:n0-diam, 0:n1-diam, 0:n2-diam])

        I = np.nonzero(np.fabs(u) < maxval)
        #print len(I[0])

        for k in range(len(I[0])):
            p = np.array((I[0][k], I[1][k], I[2][k]))
            bmin = p-rad
            bmax = p+rad + 1
            #print p, bmin, bmax
            if (d2[p[0],p[1], p[2]] > 0):
                #print u[p[0],p[1], p[2]]
                #print d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]].sum()
                res[p[0],p[1], p[2]] = \
                    min(cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1],
                                        bmin[2]:bmax[2]] < 0)])-.25
            else:
                res[p[0],p[1], p[2]] =- \
                    min(cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1],
                                        bmin[2]:bmax[2]] > 0)])-.25
                
        return res

    def toPolyData(self):
        if gotVTK:
            points = vtkPoints()
            for k in range(self.vertices.shape[0]):
                points.InsertNextPoint(self.vertices[k,0], self.vertices[k,1], self.vertices[k,2])
            polys = vtkCellArray()
            for k in range(self.faces.shape[0]):
                polys.InsertNextCell(3)
                for kk in range(3):
                    polys.InsertCellPoint(self.faces[k,kk])
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            return polydata
        else:
            raise Exception('Cannot run toPolyData without VTK')

    def fromPolyData(self, g, scales=(1.,1.,1.)):
        npoints = int(g.GetNumberOfPoints())
        nfaces = int(g.GetNumberOfPolys())
        logging.info('Dimensions: %d %d %d' %(npoints, nfaces, g.GetNumberOfCells()))
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(g.GetPoint(kk))
            #print kk, V[kk]
            #print kk, np.array(g.GetPoint(kk))
        F = np.zeros([nfaces, 3])
        gf = 0
        for kk in range(g.GetNumberOfCells()):
            c = g.GetCell(kk)
            if(c.GetNumberOfPoints() == 3):
                for ll in range(3):
                    F[gf,ll] = c.GetPointId(ll)
                    #print kk, gf, F[gf]
                gf += 1

                #self.vertices = np.multiply(data.shape-V-1, scales)
        self.vertices = np.multiply(V, scales)
        self.faces = np.int_(F[0:gf, :])
        self.component = np.zeros(self.faces.shape[0], dtype = int)
        self.computeCentersAreas()

    def subDivide(self, number=1):
        if gotVTK:
            polydata = self.toPolyData()
            subdivisionFilter = vtkLinearSubdivisionFilter()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                subdivisionFilter.SetInputData(polydata)
            else:
                subdivisionFilter.SetInput(polydata)
            subdivisionFilter.SetNumberOfSubdivisions(number)
            subdivisionFilter.Update()
            self.fromPolyData(subdivisionFilter.GetOutput())
        else:
            raise Exception('Cannot run subDivide without VTK')
                        
            
    def Simplify(self, target=1000.0, deciPro=False):
        if gotVTK:
            polydata = self.toPolyData()
            red = 1 - min(np.float(target) / polydata.GetNumberOfPoints(), 1)
            if deciPro:
                dc = vtkDecimatePro()
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    dc.SetInputData(polydata)
                else:
                    dc.SetInput(polydata)
                dc.SetTargetReduction(red)
                dc.PreserveTopologyOn()
                dc.Update()
            else:
                dc = vtkQuadricDecimation()
                dc.SetTargetReduction(red)
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    dc.SetInputData(polydata)
                else:
                    dc.SetInput(polydata)
                dc.Update()
            g = dc.GetOutput()
            self.fromPolyData(g)
            z= self.surfVolume()
            if (z > 0):
                self.flipFaces()
                logging.info('flipping volume {0:f} {1:f}'.format(z, self.surfVolume()))
        else:
            raise Exception('Cannot run Simplify without VTK')

    def flipFaces(self):
        self.faces = self.faces[:, [0,2,1]]
        self.computeCentersAreas()
          


    def smooth(self, n=30, smooth=0.1):
        if gotVTK:
            g = self.toPolyData()
            smoother= vtkWindowedSincPolyDataFilter()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                smoother.SetInputData(g)
            else:
                smoother.SetInput(g)
            #smoother.SetInput(g)
            smoother.SetNumberOfIterations(n)
            smoother.SetPassBand(smooth)   
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.GenerateErrorScalarsOn() 
            #smoother.GenerateErrorVectorsOn()
            smoother.Update()
            g = smoother.GetOutput()
            self.fromPolyData(g)
        else:
            raise Exception('Cannot run smooth without VTK')


    def Isosurface_ski(self, data, value=0.5, step = 1):
        verts,faces,n,v = measure.marching_cubes_lewiner(data, allow_degenerate=False, level=value,step_size=step)
        self.__init__(FV=(faces,verts))



    # Computes isosurfaces using vtk               
    def Isosurface(self, data, value=0.5, target=1000.0, scales = (1., 1., 1.),
                   smooth = 0.1, fill_holes = 1., orientation=1):
        if gotVTK:
            #data = self.LocalSignedDistance(data0, value)
            if isinstance(data, vtkImageData):
                img = data
            else:
                img = vtkImageData()
                img.SetDimensions(data.shape)
                img.SetOrigin(0,0,0)
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    img.AllocateScalars(VTK_FLOAT,1)
                else:
                    img.SetNumberOfScalarComponents(1)
                v = vtkDoubleArray()
                v.SetNumberOfValues(data.size)
                v.SetNumberOfComponents(1)
                for ii,tmp in enumerate(np.ravel(data, order='F')):
                    v.SetValue(ii,tmp)
                    img.GetPointData().SetScalars(v)
                
            cf = vtkContourFilter()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                cf.SetInputData(img)
            else:
                cf.SetInput(img)
            cf.SetValue(0,value)
            cf.SetNumberOfContours(1)
            cf.Update()
            #print cf
            connectivity = vtkPolyDataConnectivityFilter()
            connectivity.ScalarConnectivityOff()
            connectivity.SetExtractionModeToLargestRegion()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                connectivity.SetInputData(cf.GetOutput())
            else:
                connectivity.SetInput(cf.GetOutput())
            connectivity.Update()
            g = connectivity.GetOutput()
    
            if smooth > 0:
                smoother= vtkWindowedSincPolyDataFilter()
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    smoother.SetInputData(g)
                else:
                    smoother.SetInput(g)
                #     else:
                # smoother.SetInputConnection(contour.GetOutputPort())    
                smoother.SetNumberOfIterations(30)
                #this has little effect on the error!
                #smoother.BoundarySmoothingOff()
                #smoother.FeatureEdgeSmoothingOff()
                #smoother.SetFeatureAngle(120.0)
                smoother.SetPassBand(smooth)        #this increases the error a lot!
                smoother.NonManifoldSmoothingOn()
                #smoother.NormalizeCoordinatesOn()
                #smoother.GenerateErrorScalarsOn() 
                #smoother.GenerateErrorVectorsOn()
                smoother.Update()
                g = smoother.GetOutput()

            #dc = vtkDecimatePro()
            if target>0:
                red = 1 - min(np.float(target)/g.GetNumberOfPoints(), 1)
                #print 'Reduction: ', red
                dc = vtkQuadricDecimation()
                dc.SetTargetReduction(red)
                #dc.AttributeErrorMetricOn()
                #dc.SetDegree(10)
                #dc.SetSplitting(0)
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    dc.SetInputData(g)
                else:
                    dc.SetInput(g)
                    #dc.SetInput(g)
                #print dc
                dc.Update()
                g = dc.GetOutput()
            #print 'points:', g.GetNumberOfPoints()
            cp = vtkCleanPolyData()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                cp.SetInputData(g)
            else:
                cp.SetInput(g)
                #        cp.SetInput(dc.GetOutput())
            #cp.SetPointMerging(1)
            cp.ConvertPolysToLinesOn()
            cp.SetAbsoluteTolerance(1e-5)
            cp.Update()
            g = cp.GetOutput()
            self.fromPolyData(g,scales)
            z= self.surfVolume()
            if (orientation*z < 0):
                self.flipFaces()
                #print 'flipping volume', z, self.surfVolume()
                logging.info('flipping volume %.2f %.2f' % (z, self.surfVolume()))

        else:
            raise Exception('Cannot run Isosurface without VTK')
    
    # Ensures that orientation is correct
    def edgeRecover(self):
        v = self.vertices
        f = self.faces
        nv = v.shape[0]
        nf = f.shape[0]
        # faces containing each oriented edge
        edg0 = np.int_(np.zeros((nv, nv)))
        # number of edges between each vertex
        edg = np.int_(np.zeros((nv, nv)))
        # contiguous faces
        edgF = np.int_(np.zeros((nf, nf)))
        for (kf, c) in enumerate(f):
            if (edg0[c[0],c[1]] > 0):
                edg0[c[1],c[0]] = kf+1  
            else:
                edg0[c[0],c[1]] = kf+1
                
            if (edg0[c[1],c[2]] > 0):
                edg0[c[2],c[1]] = kf+1  
            else:
                edg0[c[1],c[2]] = kf+1  

            if (edg0[c[2],c[0]] > 0):
                edg0[c[0],c[2]] = kf+1  
            else:
                edg0[c[2],c[0]] = kf+1  

            edg[c[0],c[1]] += 1
            edg[c[1],c[2]] += 1
            edg[c[2],c[0]] += 1


        for kv in range(nv):
            I2 = np.nonzero(edg0[kv,:])
            for kkv in I2[0].tolist():
                if edg0[kkv,kv] > 0:
                    edgF[edg0[kkv,kv]-1,edg0[kv,kkv]-1] = kv+1

        isOriented = np.int_(np.zeros(f.shape[0]))
        isActive = np.int_(np.zeros(f.shape[0]))
        I = np.nonzero(np.squeeze(edgF[0,:]))
        # list of faces to be oriented
        # Start with face 0 and its neighbors
        activeList = [0]+I[0].tolist()
        lastOriented = 0
        isOriented[0] = True
        for k in activeList:
            isActive[k] = True 

        while lastOriented < len(activeList)-1:
            #next face to be oriented
            j = activeList[lastOriented +1]
            # find an already oriented face among all neighbors of j
            I = np.nonzero(edgF[j,:])
            foundOne = False
            for kk in I[0].tolist():
                if (foundOne==False) and (isOriented[kk]):
                    foundOne = True
                    u1 = edgF[j,kk] -1
                    u2 = edgF[kk,j] - 1
                    if not ((edg[u1,u2] == 1) and (edg[u2,u1] == 1)): 
                        # reorient face j
                        edg[f[j,0],f[j,1]] -= 1
                        edg[f[j,1],f[j,2]] -= 1
                        edg[f[j,2],f[j,0]] -= 1
                        a = f[j,1]
                        f[j,1] = f[j,2]
                        f[j,2] = a
                        edg[f[j,0],f[j,1]] += 1
                        edg[f[j,1],f[j,2]] += 1
                        edg[f[j,2],f[j,0]] += 1
                elif (not isActive[kk]):
                    activeList.append(kk)
                    isActive[kk] = True
            if foundOne:
                lastOriented = lastOriented+1
                isOriented[j] = True
                #print 'oriented face', j, lastOriented,  'out of',  nf,  ';  total active', len(activeList) 
            else:
                logging.info('Unable to orient face {0:d}'.format(j))
                return
        self.vertices = v
        self.faces = f

        z= self.surfVolume()
        if (z > 0):
            self.flipFaces()

    def removeIsolated(self):
        N = self.vertices.shape[0]
        inFace = np.int_(np.zeros(N))
        for k in range(3):
            inFace[self.faces[:,k]] = 1
        J = np.nonzero(inFace)
        self.vertices = self.vertices[J[0], :]
        logging.info('Found %d isolated vertices'%(N-J[0].shape[0]))
        Q = -np.ones(N)
        for k,j in enumerate(J[0]):
            Q[j] = k
        self.faces = np.int_(Q[self.faces])


    def removeDuplicates(self, c=0.0001):
        c2 = c ** 2
        N0 = self.vertices.shape[0]
        w = np.zeros(N0, dtype=int)

        newv = np.zeros(self.vertices.shape)
        removed = np.zeros(self.vertices.shape[0], dtype=bool)
        newv[0, :] = self.vertices[0, :]
        N = 1
        for kj in range(1, N0):
            dist = ((self.vertices[kj, :] - newv[0:N, :]) ** 2).sum(axis=1)
            # print dist.shape
            J = np.nonzero(dist < c2)
            J = J[0]
            # print kj, ' ', J, len(J)
            if (len(J) > 0):
                logging.info("duplicate: {0:d} {1:d}".format(kj, J[0]))
                removed[kj] = True
                w[kj] = J[0]
            else:
                w[kj] = N
                newv[N, :] = self.vertices[kj, :]
                N = N + 1

        newv = newv[0:N, :]
        self.vertices = newv
        self.faces = w[self.faces]

        newf = np.zeros(self.faces.shape, dtype=int)
        Nf = self.faces.shape[0]
        nj = 0
        for kj in range(Nf):
            if len(set(self.faces[kj,:]))< 3:
                logging.info('Empty face: {0:d} {1:d}'.format(kj, nj))
            else:
                newf[nj, :] = self.faces[kj, :]
                #newc[nj] = self.component[kj]
                nj += 1
        self.faces = newf[0:nj, :]
        self.computeCentersAreas()
        return removed
        #self.component = newc[0:nj]


    def addFace(self, f, faces, edges):
        faces.append(f)
        edges[(f[0],f[1])] = len(faces)-1
        edges[(f[1],f[2])] = len(faces)-1
        edges[(f[2],f[0])] = len(faces)-1


    def split_in_4(self, faces, vert, val, popped, edges, i, with_vertex = None):
        if not popped[i]:
            vali = np.array(val)[faces[i]]
            kmax = faces[i][vali.argmax()]
            kmin = faces[i][vali.argmin()]
            kmid = [j for j in faces[i] if j != kmin and j!= kmax][0]
            if val[kmid] > 0:
                kpivot = kmin
            else:
                kpivot = kmax
            if val[kmax] > 0:
                k0 = faces[i][0]
                k1 = faces[i][1]
                k2 = faces[i][2]

                if [kmin, kmid, kmax] in [[k0,k1,k2], [k1,k2,k0], [k2,k0,k1]]:
                    pos = True
                else:
                    pos = False

                # select[res.faces[i,:]] = True
                eps = 1e-10 * np.sign(val[kpivot])
                if val[kmin] < 0:
                    r0 = val[kmax]/(val[kmax] - val[kmin])
                    v0 = (1-r0) * vert[kmax] + r0 * vert[kmin]
                    knew0 = len(vert)
                    vert.append(v0)
                    val.append(0)
                    if val[kmid] < 1e-10 and val[kmid] > -1e-10:
                        if pos:
                            self.addFace([kmid, kmax, knew0], faces, edges)
                        else:
                            self.addFace([kmax, kmid, knew0], faces, edges)
                        popped += [False]
                        if pos:
                            self.addFace([kmid, knew0, kmin], faces, edges)
                        else:
                            self.addFace([knew0, kmid, kmin], faces, edges)
                        popped += [False]

                    if pos:
                        if (kmin,kmax) in edges:
                            jf = edges[(kmin, kmax)]
                            valf = np.array(val)[faces[jf]]
                            if valf.min() > 0 or valf.max() < 0:
                                jj = [j for j in faces[jf] if j not in faces[i]][0]
                                self.addFace([kmax, jj, knew0], faces, edges)
                                self.addFace([kmin, knew0, jj], faces, edges)
                                popped += [False]*2
                                popped[jf] = True
                    else:
                        if (kmax, kmin) in edges:
                            jf = edges[(kmax, kmin)]
                            valf = np.array(val)[faces[jf]]
                            if valf.min() > 0 or valf.max() < 0:
                                jj = [j for j in faces[jf] if j not in faces[i]][0]
                                self.addFace([kmin, jj, knew0], faces, edges)
                                self.addFace([kmax, knew0, jj], faces, edges)
                                popped += [False]*2
                                popped[jf] = True

                    if val[kmid] > 1e-10:
                        r1 = val[kmid] / (val[kmid]-val[kmin])
                        v1 = (1-r1) * vert[kmid] + r1 * vert[kmin]
                        knew1 = len(vert)
                        vert.append(v1)
                        val.append(0)
                        v2 = (vert[kmid]+vert[kmax])/2
                        knew2 = len(vert)
                        vert.append(v2)
                        val.append((val[kmid]+val[kmax])/2)
                        if pos:
                            self.addFace([knew1, kmid, knew2], faces, edges)
                            self.addFace([knew2, kmax, knew0], faces, edges)
                            self.addFace([knew0, knew1, knew2], faces, edges)
                            self.addFace([knew0, kmin, knew1], faces, edges)
                            popped += [False] * 4
                            if (kmid, kmin) in edges:
                                jf = edges[(kmid, kmin)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmin, jj, knew1], faces, edges)
                                    self.addFace([kmid, knew1, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmax, kmid) in edges:
                                jf = edges[(kmax, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmid, jj, knew2], faces, edges)
                                    self.addFace([kmax, knew2, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                        else:
                            self.addFace([kmid, knew1, knew2], faces, edges)
                            self.addFace([kmax, knew2, knew0], faces, edges)
                            self.addFace([knew1, knew0, knew2], faces, edges)
                            self.addFace([kmin, knew0, knew1], faces, edges)
                            popped += [False] * 4
                            if (kmin, kmid) in edges:
                                jf = edges[(kmin, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmin, knew1], faces, edges)
                                    self.addFace([knew1, kmid, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmid, kmax) in edges:
                                jf = edges[(kmid, kmax)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmid, knew2], faces, edges)
                                    self.addFace([knew2, kmax, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True

                    if val[kmid] < -1e-10:
                        r1 = val[kmax] / (val[kmax]-val[kmid])
                        v1 = (1-r1) * vert[kmax] + r1 * vert[kmid]
                        knew1 = len(vert)
                        vert.append(v1)
                        val.append(0)
                        v2 = (vert[kmid]+vert[kmin])/2
                        knew2 = len(vert)
                        vert.append(v2)
                        val.append((val[kmid]+val[kmin])/2)
                        if pos:
                            self.addFace([knew1, knew2, kmid], faces, edges)
                            self.addFace([knew2, knew0, kmin], faces, edges)
                            self.addFace([knew0, knew1, knew1], faces, edges)
                            self.addFace([knew1, kmax, knew0], faces, edges)
                            popped += [False] * 4
                            if (kmax, kmid) in edges:
                                jf = edges[(kmax, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmid, jj, knew1], faces, edges)
                                    self.addFace([kmax, knew1, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmid, kmin) in edges:
                                jf = edges[(kmid, kmin)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmin, jj, knew2], faces, edges)
                                    self.addFace([kmid, knew2, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                        else:
                            self.addFace([knew2, knew1, kmid], faces, edges)
                            self.addFace([knew0, knew2, kmin], faces, edges)
                            self.addFace([knew1, knew0, knew1], faces, edges)
                            self.addFace([kmax, knew1, knew0], faces, edges)
                            popped += [False] * 4
                            if (kmid, kmax) in edges:
                                jf = edges[(kmid, kmax)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmid, knew1], faces, edges)
                                    self.addFace([knew1, kmax, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmin, kmid) in edges:
                                jf = edges[(kmin, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmin, knew2], faces, edges)
                                    self.addFace([knew2, kmid, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                    popped[i] = True



    def truncate(self, ineq = (), val = None):
        res = Surface(surf=self)
        if val is None:
            val = 1e100*np.ones(res.vertices.shape[0])
        for a in ineq:
            val = np.minimum(val, (res.vertices*a[0:3]).sum(axis=1) - a[3])

        val = list(val)
        vert = list(res.vertices)
        faces = list(res.faces)
        edges = {}
        for i,f in enumerate(faces):
            edges[(f[0],f[1])] = i
            edges[(f[1], f[2])] = i
            edges[(f[2], f[0])] = i

        popped = [False]*len(faces)
        nf = len(faces)
        i = 0
        while i < nf:
            res.split_in_4(faces, vert, val, popped, edges, i)
            i += 1

        npopped = 0
        nf = len(faces)
        for i in range(nf):
            if popped[i]:
                faces.pop(i-npopped)
                npopped += 1


        res = Surface(FV=(np.array(faces, dtype=int), np.array(vert)))
        removed = res.removeDuplicates()
        val = np.array(val)[np.logical_not(removed)]
        #val = (res.vertices*a[0:3]).sum(axis=1) - a[3]
        res = res.cut(val>-1e-5)

        return res



    def cut(self, select):
        res = Surface()
        res.vertices = self.vertices[select, :]
        newindx = np.arange(self.vertices.shape[0], dtype=int)
        newindx[select] = np.arange(select.sum(), dtype=int)
        res.faces = np.zeros(self.faces.shape, dtype= int)
        j = 0
        for i in range(self.faces.shape[0]):
            if np.all(select[self.faces[i, :]]):
                res.faces[j, :] = newindx[np.copy(self.faces[i, :])]
                j += 1
        res.faces = res.faces[0:j, :]
        res.computeCentersAreas()
        return res

    def createFlatApproximation(self, thickness=None, M=50):
        # Compute center
        a = self.computeVertexArea()[0]
        A = a.sum()
        x0 = (self.vertices * a[:,np.newaxis]).sum(axis=0)/A

        # Inertial operator
        vm = self.vertices - x0
        J = ((vm[:, :, np.newaxis] * vm[:, np.newaxis, :])*a[:, np.newaxis, np.newaxis]).sum(axis=0)/A
        w,v = LA.eigh(J)
        emax = 3*np.sqrt(w.max())
        #J = LA.inv(J)
        #c0 = (np.dot(ftmpl.vertices, J)*ftmpl.vertices).sum()/ftmpl.vertices.shape[0]
        #dst = np.sqrt(((ftmpl.vertices - x0)**2).sum(axis=1)).mean(axis=0)
        #M = 100
        if thickness is None:
            w0 = w[0]
        else:
            w0 = (thickness/2)**2

        [x,y,z] = np.mgrid[0:2*M+1, 0:2*M+1, 0:2*M+1] / M
        x = emax*(x-1)
        y = emax*(y-1)
        z = emax*(z-1)
        I1 = np.logical_and((z ** 2 / w[2] + y ** 2 / w[1]) < 3, x ** 2 < w0)
        h = Surface()
        h.Isosurface_ski(data=I1, value=.5, step=3)
        labels = np.zeros(h.vertices.shape[0], dtype=int)
        u2 = (h.vertices[:,0] - M) * (emax/M)
        for j in range(labels.shape[0]):
            #u2 = x[0]  * v[0, 0] + x[1] * v[1, 0] + x[2] * v[2, 0]
            if np.fabs(u2[j]-np.sqrt(w0)) < emax/M:
                labels[j] = 1
            elif np.fabs(u2[j]+np.sqrt(w0)) < emax/M:
                labels[j] = 2

        u = np.dot(h.vertices-M, v.T)
        # u = np.zeros(hv.shape)
        # u[0,:] = x*v[0,0] + y*v[1,0] + z*v[2,0]
        # u[1,:] = x*v[0,1] + y*v[1,1] + z*v[2,1]
        # u[2,:] = x*v[0,2] + y*v[1,2] + z*v[2,2]
        #h.updateVertices(u)
        # I1 = np.logical_and((u2 ** 2 / w[2] + u1 ** 2 / w[1]) < 3, u0 ** 2 < w0)
        # h = Surface()
        # h.Isosurface_ski(data=I1, value=.5, step=3)
        print('Vertices', h.vertices.shape[0])
        #h.Isosurface(I1, value = 1, target=max(1000, self.vertices.shape[0]), scales=[1, 1, 1], smooth=0.0001)
        h.updateVertices(x0 + (u)*emax/M)
        # labels = np.zeros(h.vertices.shape[0], dtype=int)
        # for j in range(labels.shape[0]):
        #     x = h.vertices[j,:] - x0
        #     u2 = x[0]  * v[0, 0] + x[1] * v[1, 0] + x[2] * v[2, 0]
        #     if np.fabs(u2-np.sqrt(w0)) < emax/M:
        #         labels[j] = 1
        #     elif np.fabs(u2+np.sqrt(w0)) < emax/M:
        #         labels[j] = 2
        return h, labels, np.sqrt(w0)


    def compute3DVolumeImage(self, xmin = 0, xmax = 100, spacing = 1., origin = None):
        ln = xmax - xmin + 1
        vpoints = self.toPolyData()
        whiteImage = vtkImageData()
        whiteImage.SetSpacing(spacing ,spacing ,spacing)
        whiteImage.SetDimensions(ln ,ln ,ln)
        whiteImage.SetExtent(xmin , xmax , xmin ,xmax ,xmin,xmax)
        bounds = vpoints.GetBounds()
        if origin is None:
            origin = np.zeros(3)
            # for i in range(3):
            #     origin[i] = bounds[2*i] + spacing / 2
        whiteImage.SetOrigin(origin[0] ,origin[1] ,origin[2])
        whiteImage.AllocateScalars(VTK_UNSIGNED_CHAR ,1)
        count = whiteImage.GetNumberOfPoints()
        inval = 255
        outval = 0
        for i in range(count):
            whiteImage.GetPointData().GetScalars().SetTuple1(i ,inval)

        orig2 = self.vertices.mean(axis=0)
        orig2 = np.zeros(3)
        for i in range(3):
             orig2[i] = bounds[2*i]

        pol2stenc = vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(vpoints)
        pol2stenc.SetOutputOrigin((orig2[0], orig2[1], orig2[2]))
        pol2stenc.SetOutputSpacing(spacing, spacing, spacing)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        img2stenc = vtkImageStencil()
        img2stenc.SetInputData(whiteImage)
        img2stenc.SetStencilConnection(pol2stenc.GetOutputPort())
        img2stenc.ReverseStencilOff()
        img2stenc.SetBackgroundValue(outval)
        img2stenc.Update()

        result = img2stenc.GetOutput()

        img = v2n.vtk_to_numpy(result.GetPointData().GetScalars())
        # dims = result.GetDimensions()
        #
        # img = img.reshape(dims[2], dims[1], dims[0])
        # img = img.transpose(2, 1, 0)
        img = np.reshape(img, (ln,ln,ln)).astype(float)
        # img = np.zeros((ln,ln,ln))
        # ii = 0
        # for j in range(ln):
        #     for i in range(ln):
        #         for k in range(ln):
        #             img[j,i,k] = result.GetPointData().GetScalars().GetTuple1(ii)
        #             ii += 1

        return img, origin, orig2

    def addToPlot(self, ax, ec = 'b', fc = 'r', al=.5, lw=1):
        x = self.vertices[self.faces[:,0],:]
        y = self.vertices[self.faces[:,1],:]
        z = self.vertices[self.faces[:,2],:]
        a = np.concatenate([x,y,z], axis=1)
        poly = [ [a[i,j*3:j*3+3] for j in range(3)] for i in range(a.shape[0])]
        tri = Poly3DCollection(poly, alpha=al, linewidths=lw)
        tri.set_edgecolor(ec)
        tri.set_facecolor(fc)
        ax.add_collection3d(tri)
        xlim = [self.vertices[:,0].min(),self.vertices[:,0].max()]
        ylim = [self.vertices[:,1].min(),self.vertices[:,1].max()]
        zlim = [self.vertices[:,2].min(),self.vertices[:,2].max()]
        return [xlim, ylim, zlim]


    def distToTriangle(self, x0, T):
        n0 = (x0**2).sum()
        S = np.dot(T, T.T)
        SI = LA.inv(S)
        y0 = np.dot(S.T,x0)
        v = SI.sum(axis = 1)
        u = np.dot(SI,y0)
        v = SI.sum(axis=1)
        L = (1 - u.sum())/v.sum()
        u += L*v
        if min(u) >= 0:
            dist = n0 - (y0*u).sum() + L
            return dist

        dx0 = ((x0-T[0,:])**2).sum()
        dx1 = ((x0 - T[1, :]) ** 2).sum()
        dx2 = ((x0 - T[2, :]) ** 2).sum()
        d01 = S[0,0] - 2*S[0,1] + S[1,1]
        d02 = S[0,0] - 2*S[0,2] + S[2,2]
        d12 = S[1,1] - 2*S[1,2] + S[2,2]
        s0 = ((x0 - T[0,:])*(T[1,:]-T[0,:])).sum()
        s1 = ((x0 - T[1,:])*(T[2,:]-T[1,:])).sum()
        s2 = ((x0 - T[2,:])*(T[0,:]-T[2,:])).sum()
        dist = -1
        if s0 >= 0 and s0 <= d01:
            dist = dx0 - s0**2/d01
            if s1 >= 0 and s1 <= d12:
                dist = min(dist, dx1 - s1 ** 2 / d12)
                if s2 >= 0 and s2 <= d02:
                    dist = min(dist, dx2 - s2 ** 2 / d02)
        elif s1 >= 0 and s1 <= d12:
            dist = dx1 - s1 ** 2 / d12
            if s2 >= 0 and s2 <= d02:
                dist = min(dist, dx2 - s2 ** 2 / d02)
        elif s2 >= 0 and s2 <= d02:
            dist = dx2 - s2 ** 2 / d02

        if dist > -0.5:
            return dist

        return min(dx0, dx1, dx2)

    def laplacianMatrix(self):
        F = self.faces
        V = self.vertices
        nf = F.shape[0]
        nv = V.shape[0]

        AV, AF = self.computeVertexArea()

        # compute edges and detect boundary
        #edm = sp.lil_matrix((nv,nv))
        edm = -np.ones((nv,nv), dtype=int)
        E = np.zeros((3*nf, 2), dtype=int)
        j = 0
        for k in range(nf):
            if edm[F[k,0], F[k,1]]== -1:
                edm[F[k,0], F[k,1]] = j
                edm[F[k,1], F[k,0]] = j
                E[j, :] = [F[k,0], F[k,1]]
                j = j+1
            if (edm[F[k,1], F[k,2]]== -1):
                edm[F[k,1], F[k,2]] = j
                edm[F[k,2], F[k,1]] = j
                E[j, :] = [F[k,1], F[k,2]]
                j = j+1
            if (edm[F[k,0], F[k,2]]== -1):
                edm[F[k,2], F[k,0]] = j
                edm[F[k,0], F[k,2]] = j
                E[j, :] = [F[k,2], F[k,0]]
                j = j+1
        E = E[0:j, :]
        
        edgeFace = np.zeros([j, nf], dtype=int)
        ne = j
        #print E
        for k in range(nf):
            edgeFace[edm[F[k,0], F[k,1]], k] = 1 
            edgeFace[edm[F[k,1], F[k,2]], k] = 1 
            edgeFace[edm[F[k,2], F[k,0]], k] = 1 
    
        bEdge = np.zeros([ne, 1])
        bVert = np.zeros([nv, 1])
        edgeAngles = np.zeros([ne, 2])
        for k in range(ne):
            I = np.flatnonzero(edgeFace[k, :])
            #print 'I=', I, F[I, :], E.shape
            #print 'E[k, :]=', k, E[k, :]
            #print k, edgeFace[k, :]
            for u in range(len(I)):
                f = I[u]
                i1l = np.flatnonzero(F[f, :] == E[k,0])
                i2l = np.flatnonzero(F[f, :] == E[k,1])
                #print f, F[f, :]
                #print i1l, i2l
                i1 = i1l[0]
                i2 = i2l[0]
                s = i1+i2
                if s == 1:
                    i3 = 2
                elif s==2:
                    i3 = 1
                elif s==3:
                    i3 = 0
                x1 = V[F[f,i1], :] - V[F[f,i3], :]
                x2 = V[F[f,i2], :] - V[F[f,i3], :]
                a = (np.cross(x1, x2) * np.cross(V[F[f,1], :] - V[F[f,0], :], V[F[f, 2], :] - V[F[f, 0], :])).sum()
                b = (x1*x2).sum()
                if (a  > 0):
                    edgeAngles[k, u] = b/np.sqrt(a)
                else:
                    edgeAngles[k, u] = b/np.sqrt(-a)
            if (len(I) == 1):
                # boundary edge
                bEdge[k] = 1
                bVert[E[k,0]] = 1
                bVert[E[k,1]] = 1
                edgeAngles[k,1] = 0 
        

        # Compute Laplacian matrix
        L = np.zeros([nv, nv])

        for k in range(ne):
            L[E[k,0], E[k,1]] = (edgeAngles[k,0] + edgeAngles[k,1]) /2
            L[E[k,1], E[k,0]] = L[E[k,0], E[k,1]]

        for k in range(nv):
            L[k,k] = - L[k, :].sum()

        A = np.zeros([nv, nv])
        for k in range(nv):
            A[k, k] = AV[k]

        return L,A

    def graphLaplacianMatrix(self):
        F = self.faces
        V = self.vertices
        nf = F.shape[0]
        nv = V.shape[0]

        # compute edges and detect boundary
        #edm = sp.lil_matrix((nv,nv))
        edm = -np.ones([nv,nv])
        E = np.zeros([3*nf, 2])
        j = 0
        for k in range(nf):
            if (edm[F[k,0], F[k,1]]== -1):
                edm[F[k,0], F[k,1]] = j
                edm[F[k,1], F[k,0]] = j
                E[j, :] = [F[k,0], F[k,1]]
                j = j+1
            if (edm[F[k,1], F[k,2]]== -1):
                edm[F[k,1], F[k,2]] = j
                edm[F[k,2], F[k,1]] = j
                E[j, :] = [F[k,1], F[k,2]]
                j = j+1
            if (edm[F[k,0], F[k,2]]== -1):
                edm[F[k,2], F[k,0]] = j
                edm[F[k,0], F[k,2]] = j
                E[j, :] = [F[k,2], F[k,0]]
                j = j+1
        E = E[0:j, :]
        
        ne = j
        #print E

        # Compute Laplacian matrix
        L = np.zeros([nv, nv])

        for k in range(ne):
            L[E[k,0], E[k,1]] = 1
            L[E[k,1], E[k,0]] = 1

        for k in range(nv):
            L[k,k] = - L[k, :].sum()

        return L


    def laplacianSegmentation(self, k):
        (L, AA) =  self.laplacianMatrix()
        #print (L.shape[0]-k-1, L.shape[0]-2)
        (D, y) = spLA.eigh(L, AA, eigvals= (L.shape[0]-k, L.shape[0]-1))
        #V = real(V) ;
        #print D
        N = y.shape[0]
        d = y.shape[1]
        I = np.argsort(y.sum(axis=1))
        I0 =np.floor((N-1)*sp.linspace(0, 1, num=k)).astype(int)
        #print y.shape, L.shape, N, k, d
        C = y[I0, :].copy()

        eps = 1e-20
        Cold = C.copy()
        u = ((C.reshape([k,1,d]) - y.reshape([1,N,d]))**2).sum(axis=2)
        T = u.min(axis=0).sum()/(N)
        #print T
        j=0
        while j< 5000:
            u0 = u - u.min(axis=0).reshape([1, N])
            w = np.exp(-u0/T) ;
            w = w / (eps + w.sum(axis=0).reshape([1,N]))
            #print w.min(), w.max()
            cost = (u*w).sum() + T*(w*np.log(w+eps)).sum()
            C = np.dot(w, y) / (eps + w.sum(axis=1).reshape([k,1]))
            #print j, 'cost0 ', cost

            u = ((C.reshape([k,1,d]) - y.reshape([1,N,d]))**2).sum(axis=2)
            cost = (u*w).sum() + T*(w*np.log(w+eps)).sum()
            err = np.sqrt(((C-Cold)**2).sum(axis=1)).sum()
            #print j, 'cost ', cost, err, T
            if ( j>100) & (err < 1e-4 ):
                break
            j = j+1
            Cold = C.copy()
            T = T*0.99

            #print k, d, C.shape
        dst = ((C.reshape([k,1,d]) - y.reshape([1,N,d]))**2).sum(axis=2)
        md = dst.min(axis=0)
        idx = np.zeros(N).astype(int)
        for j in range(N):
            I = np.flatnonzero(dst[:,j] < md[j] + 1e-10) 
            idx[j] = I[0]
        I = -np.ones(k).astype(int)
        kk=0
        for j in range(k):
            if True in (idx==j):
                I[j] = kk
                kk += 1
        idx = I[idx]
        if idx.max() < (k-1):
            logging.info('Warning: kmeans convergence with %d clusters instead of %d' %(idx.max(), k))
            #ml = w.sum(axis=1)/N
        nc = idx.max()+1
        C = np.zeros([nc, self.vertices.shape[1]])
        a, foo = self.computeVertexArea()
        for k in range(nc):
            I = np.flatnonzero(idx==k)
            nI = len(I)
            #print a.shape, nI
            aI = a[I]
            ak = aI.sum()
            C[k, :] = (self.vertices[I, :]*aI).sum(axis=0)/ak
        
        

        return idx, C



    # Computes surface volume
    def surfVolume(self):
        f = self.faces
        v = self.vertices
        z = 0
        for c in f:
            z += np.linalg.det(v[c[:], :])/6
        return z

    # Computes surface area
    def surfArea(self):
        return np.sqrt((self.surfel**2).sum(axis=1)).sum()

    # Reads from .off file
    def readOFF(self, offfile):
        with open(offfile,'r') as f:
            ln0 = readskip(f,'#')
            ln = ln0.split()
            if ln[0].lower() != 'off':
                logging.info('Not OFF format')
                return
            ln = readskip(f,'#').split()
            # read header
            npoints = int(ln[0])  # number of vertices
            nfaces = int(ln[1]) # number of faces
                                #print ln, npoints, nfaces
                        #fscanf(fbyu,'%d',1);		% number of edges
                        #%ntest = fscanf(fbyu,'%d',1);		% number of edges
            # read data
            self.vertices = np.empty([npoints, 3])
            for k in range(npoints):
                ln = readskip(f,'#').split()
                self.vertices[k, 0] = float(ln[0]) 
                self.vertices[k, 1] = float(ln[1]) 
                self.vertices[k, 2] = float(ln[2])

            self.faces = np.int_(np.empty([nfaces, 3]))
            for k in range(nfaces):
                ln = readskip(f,'#').split()
                if (int(ln[0]) != 3):
                    logging.info('Reading only triangulated surfaces')
                    return
                self.faces[k, 0] = int(ln[1]) 
                self.faces[k, 1] = int(ln[2]) 
                self.faces[k, 2] = int(ln[3])

        self.computeCentersAreas()


        
    # Reads from .byu file
    def readbyu(self, byufile):
        with open(byufile,'r') as fbyu:
            ln0 = fbyu.readline()
            ln = ln0.split()
            # read header
            ncomponents = int(ln[0])	# number of components
            npoints = int(ln[1])  # number of vertices
            nfaces = int(ln[2]) # number of faces
                        #fscanf(fbyu,'%d',1);		% number of edges
                        #%ntest = fscanf(fbyu,'%d',1);		% number of edges
            for k in range(ncomponents):
                fbyu.readline() # components (ignored)
            # read data
            self.vertices = np.empty([npoints, 3])
            k=-1
            while k < npoints-1:
                ln = fbyu.readline().split()
                k=k+1 ;
                self.vertices[k, 0] = float(ln[0]) 
                self.vertices[k, 1] = float(ln[1]) 
                self.vertices[k, 2] = float(ln[2])
                if len(ln) > 3:
                    k=k+1 ;
                    self.vertices[k, 0] = float(ln[3])
                    self.vertices[k, 1] = float(ln[4]) 
                    self.vertices[k, 2] = float(ln[5])

            self.faces = np.empty([nfaces, 3])
            ln = fbyu.readline().split()
            kf = 0
            j = 0
            while ln:
                if kf >= nfaces:
                    break
                #print nfaces, kf, ln
                for s in ln:
                    self.faces[kf,j] = int(sp.fabs(int(s)))
                    j = j+1
                    if j == 3:
                        kf=kf+1
                        j=0
                ln = fbyu.readline().split()
        self.faces = np.int_(self.faces) - 1
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
        self.component = np.zeros(self.faces.shape[0], dtype=int)

    #Saves in .byu format
    def savebyu(self, byufile):
        #FV = readbyu(byufile)
        #reads from a .byu file into matlab's face vertex structure FV

        with open(byufile,'w') as fbyu:
            # copy header
            ncomponents = 1	    # number of components
            npoints = self.vertices.shape[0] # number of vertices
            nfaces = self.faces.shape[0]		# number of faces
            nedges = 3*nfaces		# number of edges

            str = '{0: d} {1: d} {2: d} {3: d} 0\n'.format(ncomponents, npoints, nfaces,nedges)
            fbyu.write(str) 
            str = '1 {0: d}\n'.format(nfaces)
            fbyu.write(str) 


            k=-1
            while k < (npoints-1):
                k=k+1 
                str = '{0: f} {1: f} {2: f} '.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                fbyu.write(str) 
                if k < (npoints-1):
                    k=k+1
                    str = '{0: f} {1: f} {2: f}\n'.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                    fbyu.write(str) 
                else:
                    fbyu.write('\n')

            j = 0 
            for k in range(nfaces):
                for kk in (0,1):
                    fbyu.write('{0: d} '.format(self.faces[k,kk]+1))
                    j=j+1
                    if j==16:
                        fbyu.write('\n')
                        j=0

                fbyu.write('{0: d} '.format(-self.faces[k,2]-1))
                j=j+1
                if j==16:
                    fbyu.write('\n')
                    j=0

    def saveVTK(self, fileName, scalars = None, normals = None, tensors=None, scal_name='scalars', vectors=None, vect_name='vectors'):
        vf = vtkFields()
        #print scalars
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
        self.saveVTK2(fileName, vf)

    # Saves in .vtk format
    def saveVTK2(self, fileName, vtkFields = None):
        F = self.faces
        V = self.vertices

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET POLYDATA\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll,0], V[ll,1], V[ll,2]))
            fvtkout.write('\nPOLYGONS {0:d} {1:d}'.format(F.shape[0], 4*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n3 {0: d} {1: d} {2: d}'.format(F[ll,0], F[ll,1], F[ll,2]))
            if vtkFields is not  None:
                wrote_pd_hdr = False
                if len(vtkFields.scalars) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.scalars)//2
                    for k in range(nf):
                        fvtkout.write('\nSCALARS '+ vtkFields.scalars[2*k] +' float 1\nLOOKUP_TABLE default')
                        for ll in range(V.shape[0]):
                            #print scalars[ll]
                            fvtkout.write('\n {0: .5f}'.format(vtkFields.scalars[2*k+1][ll]))
                if len(vtkFields.vectors) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.vectors)//2
                    for k in range(nf):
                        fvtkout.write('\nVECTORS '+ vtkFields.vectors[2*k] +' float')
                        vectors = vtkFields.vectors[2*k+1]
                        for ll in range(V.shape[0]):
                            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
                if len(vtkFields.normals) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.normals)//2
                    for k in range(nf):
                        fvtkout.write('\nNORMALS '+ vtkFields.normals[2*k] +' float')
                        vectors = vtkFields.normals[2*k+1]
                        for ll in range(V.shape[0]):
                            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
                if len(vtkFields.tensors) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.tensors)/2
                    for k in range(nf):
                        fvtkout.write('\nTENSORS '+ vtkFields.tensors[2*k] +' float')
                        tensors = vtkFields.tensors[2*k+1]
                        for ll in range(V.shape[0]):
                            for kk in range(2):
                                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(tensors[ll, kk, 0], tensors[ll, kk, 1], tensors[ll, kk, 2]))
                fvtkout.write('\n')


    # Reads .vtk file
    def readVTK(self, fileName):
        if gotVTK:
            u = vtkPolyDataReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            #print v
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))

            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk,ll] = c.GetPointId(ll)
            
            self.vertices = V
            self.faces = np.int_(F)
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            self.centers = (xDef1 + xDef2 + xDef3) / 3
            self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
            self.component = np.zeros(self.faces.shape[0], dtype=int)
        else:
            raise Exception('Cannot run readVTK without VTK')
    
    # Reads .vtk file
    def readFromImage(self, fileName, with_vfld = True):
        self.img = diffeo.gridScalars(fileName=fileName)
        self.img.data /= self.img.data.max() + 1e-10
        self.Isosurface(self.img.data, smooth=0.001)
        if with_vfld:
            #smoothData = cg.linearcg(lambda x: -diffeo.laplacian(x), -self.img.data, iterMax=500)
            smoothData = sp.sparse.linalg.cg(lambda x: -diffeo.laplacian(x), -self.img.data, maxiter=500)
            self.vfld = diffeo.gradient(smoothData)
    
    # Reads .vtk file
    def initFromImage(self, img):
        self.img = diffeo.gridScalars(data=img)
        self.img.data /= self.img.data.max() + 1e-10
        self.Isosurface(self.img.data)
        smoothData = sp.sparse.linalg.cg(lambda x: -diffeo.laplacian(x) ,-self.img.data ,maxiter=500)
        #smoothData = cg.linearcg(lambda x: -diffeo.laplacian(x), -self.img.data, iterMax=500)
        self.vfld = diffeo.gradient(smoothData)
    
    # Reads .obj file
    def readOBJ(self, fileName):
        if gotVTK:
            u = vtkOBJReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            #print v
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))
    
            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk,ll] = c.GetPointId(ll)
            
            self.vertices = V
            self.faces = np.int_(F)
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            self.centers = (xDef1 + xDef2 + xDef3) / 3
            self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
            self.component = np.zeros(self.faces.shape[0], dtype=int)
        else:
            raise Exception('Cannot run readOBJ without VTK')

    # Reads .stl file
    def readSTL(self, fileName):
        if gotVTK:
            u = vtkSTLReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))
            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk, ll] = c.GetPointId(ll)

            self.vertices = V
            self.faces = np.int_(F)
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            self.centers = (xDef1 + xDef2 + xDef3) / 3
            self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1) / 2
        else:
            raise Exception('Cannot run readSTL without VTK')

    def concatenate(self, fvl):
        nv = 0
        nf = 0
        for fv in fvl:
            nv += fv.vertices.shape[0]
            nf += fv.faces.shape[0]
        self.vertices = np.zeros([nv,3])
        self.faces = np.zeros([nf,3], dtype='int')
        self.component = np.zeros(nf, dtype='int')

        nv0 = 0
        nf0 = 0
        c = 0
        for fv in fvl:
            nv = nv0 + fv.vertices.shape[0]
            nf = nf0 + fv.faces.shape[0]
            self.vertices[nv0:nv, :] = fv.vertices
            self.faces[nf0:nf, :] = fv.faces + nv0
            self.component[nf0:nf] = c
            nv0 = nv
            nf0 = nf
            c += 1
        self.computeCentersAreas()
    
    def normGrad(self, phi):
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        l1 = ((v2-v3)**2).sum(axis=1)
        l2 = ((v1-v3)**2).sum(axis=1)
        l3 = ((v1-v2)**2).sum(axis=1)
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        phi3 = phi[self.faces[:,2],:]
        a = 4*np.sqrt((self.surfel**2).sum(axis=1))
        u = l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1)
        res = (u/a).sum()
        return res
    
    def laplacian(self, phi, weighted=False):
        res = np.zeros(phi.shape)
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        l1 = (((v2-v3)**2).sum(axis=1))[...,np.newaxis]
        l2 = (((v1-v3)**2).sum(axis=1))[...,np.newaxis]
        l3 = (((v1-v2)**2).sum(axis=1))[...,np.newaxis]
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        phi3 = phi[self.faces[:,2],:]
        a = 8*(np.sqrt((self.surfel**2).sum(axis=1)))[...,np.newaxis]
        r1 = (l1 * (phi2 + phi3-2*phi1) + (l2-l3) * (phi2-phi3))/a
        r2 = (l2 * (phi1 + phi3-2*phi2) + (l1-l3) * (phi1-phi3))/a
        r3 = (l3 * (phi1 + phi2-2*phi3) + (l2-l1) * (phi2-phi1))/a
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] += r2[k,:]
            res[f[2],:] += r3[k,:]
        if weighted:
            av = self.computeVertexArea()
            return res/av[0]
        else:
            return res

    def diffNormGrad(self, phi):
        res = np.zeros((self.vertices.shape[0],phi.shape[1]))
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        l1 = (((v2-v3)**2).sum(axis=1))
        l2 = (((v1-v3)**2).sum(axis=1))
        l3 = (((v1-v2)**2).sum(axis=1))
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        phi3 = phi[self.faces[:,2],:]
        #a = ((self.surfel**2).sum(axis=1))
        a = 2*np.sqrt((self.surfel**2).sum(axis=1))
        u = l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1)
        #u = (2*u/a**2)[...,np.newaxis]
        u = (u/a**3)[...,np.newaxis]
        a = a[...,np.newaxis]
        
        r1 = - u * np.cross(v2-v3,self.surfel) + 2*((v1-v3) *(((phi3-phi2)*(phi1-phi2)).sum(axis=1))[:,np.newaxis]
            + (v1-v2)*(((phi1-phi3)*(phi2-phi3)).sum(axis=1)[:,np.newaxis]))/a
        r2 = - u * np.cross(v3-v1,self.surfel) + 2*((v2-v1) *(((phi1-phi3)*(phi2-phi3)).sum(axis=1))[:,np.newaxis]
            + (v2-v3)*(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis])/a
        r3 = - u * np.cross(v1-v2,self.surfel) + 2*((v3-v2) *(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis]
            + (v3-v1)*(((phi3-phi2)*(phi1-phi2)).sum(axis=1)[:,np.newaxis]))/a
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] += r2[k,:]
            res[f[2],:] += r3[k,:]
        return res/2
    
    def meanCurvatureVector(self):
        res = np.zeros(self.vertices.shape)
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        a = np.sqrt(((self.surfel**2).sum(axis=1)))
        a = a[...,np.newaxis]
        
        r1 = - np.cross(v2-v3,self.surfel)/a
        r2 = - np.cross(v3-v1,self.surfel)/a
        r3 = - np.cross(v1-v2,self.surfel)/a
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] += r2[k,:]
            res[f[2],:] += r3[k,:]
        return res
    
# Reads several .byu files
def readMultipleByu(regexp, Nmax = 0):
    files = glob.glob(regexp)
    if Nmax > 0:
        nm = min(Nmax, len(files))
    else:
        nm = len(files)
    fv1 = []
    for k in range(nm):
        fv1.append(Surface(files[k]))
    return fv1

# saves time dependent surfaces (fixed topology)
def saveEvolution(fileName, fv0, xt):
    fv = Surface(fv0)
    for k in range(xt.shape[0]):
        fv.vertices = np.squeeze(xt[k, :, :])
        fv.savebyu(fileName+'{0: 02d}'.format(k)+'.byu')




def haussdorffDist(fv1, fv2):
    dst = ((fv1.vertices[:, np.newaxis, :] - fv2.vertices[np.newaxis, :, :])**2).sum(axis=2)
    res = np.amin(dst, axis=1).max() + np.amin(dst, axis=0).max()
    return res


# Current norm of fv1
def currentMagnitude(fv1, KparDist):
    c2 = fv1.centers
    cr2 = np.copy(fv1.surfel)
    obj = np.multiply(cr2, KparDist.applyK(c2, cr2)).sum()
    return obj

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentMagnitudeGradient(fvDef, KparDist):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = np.copy(fvDef.surfel)
    dim = c1.shape[1]

    z1 = KparDist.applyK(c1, cr1)/2
    dz1 = (1./3.) * KparDist.applyDiffKT(c1, cr1[np.newaxis,...], cr1[np.newaxis,...])

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:,0]
    crs = np.cross(xDef3 - xDef2, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,1]
    crs = np.cross(xDef1 - xDef3, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,2]
    crs = np.cross(xDef2 - xDef1, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    return 2*px


def currentNorm0(fv1, KparDist, weight=1.):
    return currentMagnitude(fv1, KparDist)


# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct 
def currentNormDef(fvDef, fv1, KparDist, weight=1.):
    c1 = fvDef.centers
    cr1 = np.copy(fvDef.surfel)
    c2 = fv1.centers
    cr2 = np.copy(fv1.surfel)
    obj = (np.multiply(cr1, KparDist.applyK(c1, cr1)).sum()
        - 2*np.multiply(cr1, KparDist.applyK(c2, cr2, firstVar=c1)).sum())
    return obj

# Returns |fvDef - fv1|^2 for current norm
def currentNorm(fvDef, fv1, KparDist, weight=1.):
    return currentNormDef(fvDef, fv1, KparDist) + currentNorm0(fv1, KparDist) 

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentNormGradient(fvDef, fv1, KparDist, weight=1.):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = np.copy(fvDef.surfel)
    c2 = fv1.centers
    cr2 = np.copy(fv1.surfel)
    dim = c1.shape[1]

    z1 = (KparDist.applyK(c1, cr1) - KparDist.applyK(c2, cr2, firstVar=c1))/2
    dz1 = (1./3.) * (KparDist.applyDiffKT(c1, cr1, cr1) -
                     KparDist.applyDiffKT(c2, cr1, cr2, firstVar=c1))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:,0]
    crs = np.cross(xDef3 - xDef2, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,1]
    crs = np.cross(xDef1 - xDef3, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,2]
    crs = np.cross(xDef2 - xDef1, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    return 2*px

# Measure norm of fv1
def measureNorm0(fv1, KparDist):
    c2 = fv1.centers
    cr2 = np.sqrt((fv1.surfel**2).sum(axis=1)+1e-10)[:,np.newaxis]
    return np.multiply(cr2, KparDist.applyK(c2, cr2)).sum()
        
    
# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct 
def measureNormDef(fvDef, fv1, KparDist):
    c1 = fvDef.centers
    cr1 = np.sqrt((fvDef.surfel**2).sum(axis=1)+1e-10)[:,np.newaxis]
    c2 = fv1.centers
    cr2 = np.sqrt((fv1.surfel**2).sum(axis=1)+1e-10)[:,np.newaxis]
    obj = (np.multiply(cr1, KparDist.applyK(c1, cr1)).sum()
        - 2*np.multiply(cr1, KparDist.applyK(c2, cr2, firstVar=c1)).sum())
    return obj

# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist) 


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    c2 = fv1.centers
    dim = c1.shape[1]
    a1 = np.sqrt((fvDef.surfel**2).sum(axis=1)+1e-10)
    a2 = np.sqrt((fv1.surfel**2).sum(axis=1)+1e-10)
    cr1 = fvDef.surfel / a1[:, np.newaxis]
    #cr2 = fv1.surfel / a2[:, np.newaxis]

    z1 = KparDist.applyK(c1, a1[:, np.newaxis]) - KparDist.applyK(c2, a2[:, np.newaxis], firstVar=c1)
    z1 = np.multiply(z1, cr1)
    #print a1.shape, c1.shape
    dz1 = (1./3.) * (KparDist.applyDiffKT(c1, a1[:,np.newaxis], a1[:,np.newaxis]) -
                      KparDist.applyDiffKT(c2, a1[:,np.newaxis], a2[:,np.newaxis], firstVar=c1))
                        

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:,0]
    crs = np.cross(xDef3 - xDef2, z1)/2
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,1]
    crs = np.cross(xDef1 - xDef3, z1)/2
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,2]
    crs = np.cross(xDef2 - xDef1, z1)/2
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    return 2*px

def varifoldNorm0(fv1, KparDist, weight=1.):
    d=weight
    c2 = fv1.centers
    a2 = np.sqrt((fv1.surfel**2).sum(axis=1)+1e-10)
    cr2 = fv1.surfel/a2[:,np.newaxis]
    cr2cr2 = (cr2[:,np.newaxis,:]*cr2[np.newaxis,:,:]).sum(axis=2)
    a2a2 = a2[:,np.newaxis]*a2[np.newaxis,:]
    beta2 = (1 + d*cr2cr2**2)*a2a2
    return KparDist.applyK(c2, beta2[...,np.newaxis], matrixWeights=True).sum()
        

# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct 
def varifoldNormDef(fvDef, fv1, KparDist, weight=1.):
    d=weight
    c1 = fvDef.centers
    c2 = fv1.centers
    a1 = np.sqrt((fvDef.surfel**2).sum(axis=1)+1e-10)
    a2 = np.sqrt((fv1.surfel**2).sum(axis=1)+1e-10)
    cr1 = fvDef.surfel/a1[:,np.newaxis]
    cr2 = fv1.surfel/a2[:,np.newaxis]

    cr1cr1 = (cr1[:,np.newaxis,:]*cr1[np.newaxis,:,:]).sum(axis=2)
    a1a1 = a1[:,np.newaxis]*a1[np.newaxis,:]
    cr1cr2 = (cr1[:,np.newaxis,:]*cr2[np.newaxis,:,:]).sum(axis=2)
    a1a2 = a1[:,np.newaxis]*a2[np.newaxis,:]

    beta1 = (1 + d*cr1cr1**2)*a1a1
    beta2 = (1 + d*cr1cr2**2)*a1a2

    obj = (KparDist.applyK(c1, beta1[...,np.newaxis], matrixWeights=True).sum()
        - 2*KparDist.applyK(c2, beta2[...,np.newaxis], firstVar=c1, matrixWeights=True).sum())
    return obj

# Returns |fvDef - fv1|^2 for current norm
def varifoldNorm(fvDef, fv1, KparDist, weight=1.):
    return varifoldNormDef(fvDef, fv1, KparDist, weight=weight) + varifoldNorm0(fv1, KparDist, weight=weight)

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def varifoldNormGradient(fvDef, fv1, KparDist, weight=1.):
    d=weight
    xDef = fvDef.vertices
    c1 = fvDef.centers
    c2 = fv1.centers
    dim = c1.shape[1]

    a1 = np.sqrt((fvDef.surfel**2).sum(axis=1)+1e-10)
    a2 = np.sqrt((fv1.surfel**2).sum(axis=1)+1e-10)
    cr1 = fvDef.surfel / a1[:, np.newaxis]
    cr2 = fv1.surfel / a2[:, np.newaxis]
    cr1cr1 =  (cr1[:, np.newaxis, :] * cr1[np.newaxis, :, :]).sum(axis=2)
    cr1cr2 =  (cr1[:, np.newaxis, :] * cr2[np.newaxis, :, :]).sum(axis=2)

    beta1 = a1[:,np.newaxis]*a1[np.newaxis,:] * (1 + d*cr1cr1**2) 
    beta2 = a1[:,np.newaxis]*a2[np.newaxis,:] * (1 + d*cr1cr2**2)

    u1 = (2*d*cr1cr1[...,np.newaxis]*cr1[np.newaxis,...]
          - d*(cr1cr1**2)[...,np.newaxis]*cr1[:,np.newaxis,:]
          + cr1[:,np.newaxis,:])*a1[np.newaxis,:,np.newaxis]
    u2 = (2*d*cr1cr2[...,np.newaxis]*cr2[np.newaxis,...]
          - d*(cr1cr2**2)[...,np.newaxis]*cr1[:,np.newaxis,:]
          + cr1[:,np.newaxis,:])*a2[np.newaxis,:,np.newaxis]

    z1 = KparDist.applyK(c1, u1,matrixWeights=True) - KparDist.applyK(c2, u2, firstVar=c1, matrixWeights=True)
    #print a1.shape, c1.shape
    dz1 = (1./3.) * (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))
                        
    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:,0]
    crs = np.cross(xDef3 - xDef2, z1)/2
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,1]
    crs = np.cross(xDef1 - xDef3, z1)/2
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,2]
    crs = np.cross(xDef2 - xDef1, z1)/2
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    return 2*px


def L2Norm0(fv1):
    return np.fabs(fv1.surfVolume())
    
def L2Norm(fvDef, vfld):
    vf = np.zeros((fvDef.centers.shape[0], 3))
    for k in range(3):
        vf[:,k] = diffeo.multilinInterp(vfld[k, ...], fvDef.centers.T)
    #vf = fvDef.centers/3 - 2*vf
    #vf =  - 2*vf
    #print 'volume: ', fvDef.surfVolume(), (vf*fvDef.surfel).sum()
    return np.fabs(fvDef.surfVolume()) - 2*(vf*fvDef.surfel).sum()
    
def L2NormGradient(fvDef, vfld):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.surfel
    dvf = np.zeros((fvDef.faces.shape[0], 3, 3))
    for k in range(3):
        dvf[:,k,:] = diffeo.multilinInterpGradient(vfld[k, ...], c1.T).T
    gradc = cr1/3
    for k in range(3):
        gradc[:,k] -= 2*(dvf[:,:,k]*cr1).sum(axis=1)
    gradc = gradc/3
    
    gradn = np.zeros((fvDef.faces.shape[0],3))
    for k in range(3):
        gradn[:,k] = diffeo.multilinInterp(vfld[k, ...], c1.T)
    gradn = c1/3 - 2*gradn
    gradn = gradn/2
    
    grad = np.zeros((xDef.shape[0], 3))
    xDef0 = xDef[fvDef.faces[:, 0], :]
    xDef1 = xDef[fvDef.faces[:, 1], :]
    xDef2 = xDef[fvDef.faces[:, 2], :]

    crs = np.cross(xDef2 - xDef1, gradn)
    I = fvDef.faces[:,0]
    for k in range(I.size):
        grad[I[k], :] = grad[I[k], :] + gradc[k,:] - crs[k,:]

    crs = np.cross(xDef0 - xDef2, gradn)
    I = fvDef.faces[:,1]
    for k in range(I.size):
        grad[I[k], :] = grad[I[k], :] + gradc[k,:] - crs[k,:]

    crs = np.cross(xDef1 - xDef0, gradn)
    I = fvDef.faces[:,2]
    for k in range(I.size):
        grad[I[k], :] = grad[I[k], :] + gradc[k,:] - crs[k,:]
        
    return grad


def normGrad(fv, phi):
    v1 = fv.vertices[fv.faces[:,0],:]
    v2 = fv.vertices[fv.faces[:,1],:]
    v3 = fv.vertices[fv.faces[:,2],:]
    l1 = ((v2-v3)**2).sum(axis=1)
    l2 = ((v1-v3)**2).sum(axis=1)
    l3 = ((v1-v2)**2).sum(axis=1)
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    phi3 = phi[fv.faces[:,2],:]
    a = 4*np.sqrt((fv.surfel**2).sum(axis=1))
    u = l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1)
    res = (u/a).sum()
    return res

def laplacian(fv, phi, weighted=False):
    res = np.zeros(phi.shape)
    v1 = fv.vertices[fv.faces[:,0],:]
    v2 = fv.vertices[fv.faces[:,1],:]
    v3 = fv.vertices[fv.faces[:,2],:]
    l1 = (((v2-v3)**2).sum(axis=1))[...,np.newaxis]
    l2 = (((v1-v3)**2).sum(axis=1))[...,np.newaxis]
    l3 = (((v1-v2)**2).sum(axis=1))[...,np.newaxis]
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    phi3 = phi[fv.faces[:,2],:]
    a = 8*(np.sqrt((fv.surfel**2).sum(axis=1)))[...,np.newaxis]
    r1 = (l1 * (phi2 + phi3-2*phi1) + (l2-l3) * (phi2-phi3))/a
    r2 = (l2 * (phi1 + phi3-2*phi2) + (l1-l3) * (phi1-phi3))/a
    r3 = (l3 * (phi1 + phi2-2*phi3) + (l2-l1) * (phi2-phi1))/a
    for k,f in enumerate(fv.faces):
        res[f[0],:] += r1[k,:]
        res[f[1],:] += r2[k,:]
        res[f[2],:] += r3[k,:]
    if weighted:
        av = fv.computeVertexArea()
        return res/av[0]
    else:
        return res

def diffNormGrad(fv, phi, variables='both'):
    v1 = fv.vertices[fv.faces[:,0],:]
    v2 = fv.vertices[fv.faces[:,1],:]
    v3 = fv.vertices[fv.faces[:,2],:]
    l1 = (((v2-v3)**2).sum(axis=1))
    l2 = (((v1-v3)**2).sum(axis=1))
    l3 = (((v1-v2)**2).sum(axis=1))
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    phi3 = phi[fv.faces[:,2],:]
    #a = ((fv.surfel**2).sum(axis=1))
    a = 2*np.sqrt((fv.surfel**2).sum(axis=1))
    a2 = 2*a[...,np.newaxis]
    if variables == 'both' or variables == 'phi':
        r1 = (l1[:, np.newaxis] * (phi2 + phi3-2*phi1) + (l2-l3)[:, np.newaxis] * (phi2-phi3))/a2
        r2 = (l2[:, np.newaxis] * (phi1 + phi3-2*phi2) + (l1-l3)[:, np.newaxis] * (phi1-phi3))/a2
        r3 = (l3[:, np.newaxis] * (phi1 + phi2-2*phi3) + (l2-l1)[:, np.newaxis] * (phi2-phi1))/a2
        gradphi = np.zeros(phi.shape)
        for k,f in enumerate(fv.faces):
            gradphi[f[0],:] -= r1[k,:]
            gradphi[f[1],:] -= r2[k,:]
            gradphi[f[2],:] -= r3[k,:]

    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        u = (l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) 
        + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1))
        #u = (2*u/a**2)[...,np.newaxis]
        u = (u/(a**3))[...,np.newaxis]
        r1 = (- u * np.cross(v2-v3,fv.surfel) + 2*((v1-v3) *(((phi3-phi2)*(phi1-phi2)).sum(axis=1))[:,np.newaxis]
            + (v1-v2)*(((phi1-phi3)*(phi2-phi3)).sum(axis=1)[:,np.newaxis]))/a2)
        r2 = (- u * np.cross(v3-v1,fv.surfel) + 2*((v2-v1) *(((phi1-phi3)*(phi2-phi3)).sum(axis=1))[:,np.newaxis]
            + (v2-v3)*(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis])/a2)
        r3 = (- u * np.cross(v1-v2,fv.surfel) + 2*((v3-v2) *(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis]
            + (v3-v1)*(((phi3-phi2)*(phi1-phi2)).sum(axis=1)[:,np.newaxis]))/a2)
        for k,f in enumerate(fv.faces):
            gradx[f[0],:] += r1[k,:]
            gradx[f[1],:] += r2[k,:]
            gradx[f[2],:] += r3[k,:]

    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        logging.info('Incorrect option in diffNormGrad')
    





def readskip(f, c):
    ln0 = f.readline()
    #print ln0
    while (len(ln0) > 0 and ln0[0] == c):
        ln0 = f.readline()
    return ln0

# class MultiSurface:
#     def __init__(self, pattern):
#         self.surf = []
#         files = glob.glob(pattern)
#         for f in files:
#             self.surf.append(Surface(filename=f))
