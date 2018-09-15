import numpy as np
import scipy as sp
import os
import glob
try:
    from vtk import *
    gotVTK = True
except ImportError:
    print 'could not import VTK functions'
    gotVTK = False
    
#import kernelFunctions as kfun
import matplotlib.pyplot as plt

# General curve class
class Curve:
    def __init__(self, curve=None, filename=None, FV = None, pointSet=None, isOpen=False):
        if curve is None:
            if FV == None:
                if filename is None:
                    if pointSet is None:
                        self.vertices = np.empty(0)
                        self.centers = np.empty(0)
                        self.faces = np.empty(0)
                        self.linel = np.empty(0)
                        self.component = np.empty(0)
                    else:
                        self.vertices = np.copy(pointSet)
                        self.faces = np.zeros([pointSet.shape[0], 2], dtype=int)
                        self.component = np.zeros(pointSet.shape[0], dtype=int)
                        for k in range(pointSet.shape[0]-1):
                            self.faces[k,:] = (k, k+1)
                        if isOpen == False:
                            self.faces[pointSet.shape[0]-1, :] = (pointSet.shape[0]-1, 0)
                        self.computeCentersLengths()
                else:
                    if type(filename) is list:
                        fvl = []
                        for name in filename:
                            fvl.append(Curve(filename=name))
                        self.concatenate(fvl)
                    else:
                        self.read(filename)
            else:
                self.vertices = np.copy(FV[1])
                self.faces = np.int_(np.copy(FV[0]))
                self.component = np.zeros(self.faces.shape[0], dtype=int)
                self.computeCentersLengths()
        else:
            if type(curve) is list:
                self.concatenate(curve)
            else:
                self.vertices = np.copy(curve.vertices)
                self.linel = np.copy(curve.linel)
                self.faces = np.copy(curve.faces)
                self.centers = np.copy(curve.centers)
                self.component = np.copy(curve.component)

    def read(self, filename):
        (mainPart, ext) = os.path.splitext(filename)
        if ext == '.dat':
            self.readCurve(filename)
        elif ext=='.txt':
            self.readTxt(filename)
        elif ext=='.vtk':
            self.readVTK(filename)
        else:
            print 'Unknown Curve Extension:', ext
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.component = np.empty(0)
            self.faces = np.empty(0)
            self.linel = np.empty(0)


    def concatenate(self, fvl):
        nv = 0
        nf = 0
        for fv in fvl:
            dim = fv.vertices.shape[1]
            nv += fv.vertices.shape[0]
            nf += fv.faces.shape[0]
        self.vertices = np.zeros([nv,dim])
        self.faces = np.zeros([nf,2], dtype='int')
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
        self.computeCentersLengths()
        self.removeDuplicates()


    # face centers and area weighted normal
    def computeCentersLengths(self):
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        self.centers = (xDef1 + xDef2) / 2
        #self.linel = np.zeros([self.faces.shape[0], self.vertices.shape[1]]) ;
        self.linel = xDef2 - xDef1

    # modify vertices without toplogical change
    def updateVertices(self, x0):
        self.vertices = np.copy(x0) 
        self.computeCentersLengths()

    def computeVertexLength(self):
        a = np.zeros(self.vertices.shape[0])
        n = np.zeros(self.vertices.shape[0])
        af = np.sqrt((self.linel**2).sum(axis=1))
        for jj in range(3):
            I = self.faces[:,jj]
            for k in range(I.size):
                a[I[k]] += af[k]
                n[I[k]] += 1
        a = np.divide(a,n)
        return a
        
    def computeUnitFaceNormals(self):
        a = np.sqrt((self.linel**2).sum(axis=1))
        normals = np.zeros(self.faces.shape)
        normals[:,0] = - self.linel[:,1]/a
        normals[:,1] = self.linel[:,0]/a
        return normals

    def computeCurvature(self):
        e = self.vertices[self.faces[:,1] ,:] - self.vertices[self.faces[:,0] ,:]
        th = np.arctan2(e[:,1], e[:,0])
        #print th.shape
        ll = np.sqrt((self.linel**2).sum(axis=1))/2
        ka = (th[np.mod(range(1,self.faces.shape[0]+1), self.faces.shape[0])] - th[range(0,self.faces.shape[0])])/ll
        nrm = np.zeros(self.linel.shape)
        nrm[:,0] = -self.linel[:,1]
        nrm[:,1] = self.linel[:,0]
        lnrm = np.sqrt((nrm**2).sum(axis=1))[:, np.newaxis]
        nrm = nrm/lnrm
        nrm = (nrm[np.mod(range(1,self.faces.shape[0]+1), self.faces.shape[0])] + nrm[range(0,self.faces.shape[0])])/2
        kan = self.linel/ll[:,np.newaxis]
        ka = (kan[np.mod(range(1,self.faces.shape[0]+1), self.faces.shape[0]),:] - kan[range(0,self.faces.shape[0]),:])
        
        return ka, nrm, kan
            
    # Computes isocontours using vtk               
    def Isocontour(self, data, value=0.5, target=100.0, scales = [1., 1.], smooth = 30, fill_holes = 1., singleComponent = True):
        if gotVTK:
            #data = self.LocalSignedDistance(data0, value)
            img = vtkImageData()
            img.SetDimensions(data.shape[0], data.shape[1], 1)
            if vtkVersion.GetVTKMajorVersion() >= 6:
                img.AllocateScalars(VTK_FLOAT,1)
            else:
                img.SetNumberOfScalarComponents(1)
            img.SetOrigin(0,0, 0)
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
#            tg = 1 - min(1, float(target)/cf0.GetOutput().GetNumberOfLines())
#            print 'tg=', tg
#            if tg > 1e-10:
#                cf = vtkDecimatePro()
#                if vtkVersion.GetVTKMajorVersion() >= 6:
#                    cf.SetInputData(cf0.GetOutput())
#                else:
#                    cf.SetInput(cf0.GetOutput())
#                print 'pts', cf.GetInput().GetNumberOfPoints()
#                cf.SetTargetReduction(tg)
#                cf.PreserveTopologyOn()
#                cf.Update()
#                # return cf
#                # #print cf
#            else:
#                cf = cf0
            if singleComponent:
                connectivity = vtkPolyDataConnectivityFilter()
                connectivity.ScalarConnectivityOff()
                connectivity.SetExtractionModeToLargestRegion()
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    connectivity.SetInputData(cf.GetOutput())
                else:
                    connectivity.SetInput(cf.GetOutput())
                connectivity.Update()
                g = connectivity.GetOutput()
            else:
                g = cf.GetOutput()
            
            npoints = int(g.GetNumberOfPoints())
            nfaces = int(g.GetNumberOfLines())
            print 'Dimensions:', npoints, nfaces, g.GetNumberOfCells()
            V = np.zeros([npoints, 2])
            for kk in range(npoints):
                V[kk, :] = np.array(g.GetPoint(kk)[0:2])
                #print kk, V[kk]
                #print kk, np.array(g.GetPoint(kk))
            F = np.zeros([nfaces, 2])
            gf = 0
            for kk in range(g.GetNumberOfCells()):
                c = g.GetCell(kk)
                if(c.GetNumberOfPoints() == 2):
                    for ll in range(2):
                        F[gf,ll] = c.GetPointId(ll)
                        #print kk, gf, F[gf]
                    gf += 1

                    #self.vertices = np.multiply(data.shape-V-1, scales)
            self.vertices = np.multiply(V, scales)
            self.faces = np.int_(F[0:gf, :])
            self.component = np.zeros(self.faces.shape[0], dtype = int)
            #self.checkEdges()
            #print self.faces.shape
            self.orientEdges()
            self.removeDuplicates()
            self.computeCentersLengths()
            #print self.faces.shape
            #self.checkEdges()
        else:
            raise Exception('Cannot run Isocontour without VTK')


    def orientEdges(self):
        isInFace = - np.ones([self.vertices.shape[0], 2])
        is0 = np.zeros(self.vertices.shape[0])
        is1 = np.zeros(self.vertices.shape[0])
        for k in range(self.faces.shape[0]):
            if isInFace[self.faces[k,0], 0] == -1:
                isInFace[self.faces[k,0], 0] = k
            else:
                isInFace[self.faces[k,0], 1] = k

            if isInFace[self.faces[k,1], 0] == -1:
                isInFace[self.faces[k,1], 0] = k
            else:
                isInFace[self.faces[k,1], 1] = k
            is0[self.faces[k,0]] += 1
            is1[self.faces[k,1]] += 1
        isInFace = np.int_(isInFace)
        is0 = np.int_(is0)
        is1 = np.int_(is1)

        if ((is0+is1).max() !=2) | ((is0+is1).min()!=2):
            print 'Problems with curve in orientEdges: wrong topology'
            return

        count = np.zeros(self.vertices.shape[0])
        usedFace = np.zeros(self.faces.shape[0])
        F = np.int_(np.zeros(self.faces.shape))
        F[0, :] = self.faces[0,:]
        usedFace[0] = 1
        count[F[0,0]] = 1
        count[F[0,1]] = 1
        #k0 = F[0,0]
        kcur = F[0,1]
        j=1
        while j < self.faces.shape[0]:
            #print j
            if usedFace[isInFace[kcur,0]]>0.5:
                kf = isInFace[kcur,1]
            else:
                kf = isInFace[kcur,0]
                #print kf
            usedFace[kf] = 1
            F[j, 0] = kcur
            if self.faces[kf,0] == kcur:
                F[j,1] = self.faces[kf,1]
            else:
                F[j,1] = self.faces[kf,0]
                #print kcur, self.faces[kf,:], F[j,:]
            if count[F[j,1]] > 0.5:
                j += 1
                if (j < self.faces.shape[0]):
                    print 'Early loop in curve:', j, self.faces.shape[0]
                break
            count[F[j,1]]=1
            kcur = F[j,1]
            j += 1
            #print j
            #print j, self.faces.shape[0]
        self.faces = np.int_(F[0:j, :])
        
    def removeDuplicates(self, c=0.0001):
        c2 = c**2
        N0 = self.vertices.shape[0]
        w = np.zeros(N0, dtype=int)

        newv = np.zeros(self.vertices.shape)
        newv[0,:] = self.vertices[0,:]
        N = 1
        for kj in range(1,N0):
            dist = ((self.vertices[kj,:]-newv[0:N,:])**2).sum(axis=1)
            #print dist.shape
            J = np.nonzero(dist<c2)
            J = J[0]
            #print kj, ' ', J, len(J)
            if (len(J)>0):
                print "duplicate:", kj, J[0]
                w[kj] = J[0]
            else:
                w[kj] = N
                newv[N, :] = self.vertices[kj,:] 
                N=N+1

        newv = newv[0:N,:]
        self.vertices = newv
        self.faces = w[self.faces]
        
        newf = np.zeros(self.faces.shape, dtype=int)
        newc = np.zeros(self.component.shape, dtype=int)
        Nf = self.faces.shape[0]
        nj = 0
        for kj in range(Nf):
            if np.fabs(self.faces[kj,0] - self.faces[kj,1]) == 0:
                print 'Empty face: ', kj, nj
            else:
                newf[nj,:] = self.faces[kj,:]
                newc[nj] = self.component[kj]
                nj += 1
        self.faces = newf[0:nj, :]
        self.component = newc[0:nj]
                
        

            

    def checkEdges(self):
        is0 = np.zeros(self.vertices.shape[0])
        is1 = np.zeros(self.vertices.shape[0])
        is0 = np.int_(is0)
        is1 = np.int_(is1)
        for k in range(self.faces.shape[0]):
            is0[self.faces[k,0]] += 1
            is1[self.faces[k,1]] += 1
        #print is0 + is1
        if ((is0.max() !=1) | (is0.min() !=1) | (is1.max() != 1) | (is1.min() != 1)):
            print 'Problem in Curve'
            #print is0+is1
            return 1
        else:
            return 0

    # Computes enclosed area
    def enclosedArea(self):
        f = self.faces
        v = self.vertices
        z = 0
        for c in f:
            z += np.linalg.det(v[c[:], :])/2
        return z

    def length(self):
        ll = np.sqrt((self.linel ** 2).sum(axis=1))
        return ll.sum()

    def diffArcLength(self):
        return np.sqrt((self.linel ** 2).sum(axis=1))

    def arclength(self):
        ll = np.sqrt((self.linel**2).sum(axis=1))
        return np.cumsum(ll)
                        

    # Reads from .txt file
    def readTxt(self, infile):
        with open(infile,'r') as ftxt:
            ln0 = ftxt.readline()
            ln = ln0.split()
            # read header
            dim = int(ln[0])	# number of components
            npoints = int(ln[1])  # number of vertices
            # read data
            self.vertices = np.zeros([npoints, dim]) ;
            for k in range(npoints):
                ln = ftxt.readline().split()
                for kk in range(dim):
                    self.vertices[k, kk] = float(ln[kk]) 

        self.faces = np.int_(np.empty([npoints, 2]))
        self.faces[:,0] = range(npoints)
        self.faces[0:npoints-1,1] = range(1,npoints)
        self.faces[npoints-1,1] = 0
        #print nfaces, kf, ln
        
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        self.centers = (xDef1 + xDef2) / 2
        #self.linel = np.zeros(self.faces.shape[0], self.vertices.shape[1]) ;
        self.linel = xDef2 - xDef1
        #self.linel[:,1] = xDef2[:,1] - xDef1[:,1] ; 

    # Reads from .dat file
    def readCurve(self, infile):
        with open(infile,'r') as fbyu:
            ln0 = fbyu.readline()
            ln = ln0.split()
            # read header
            ncomponents = int(ln[0])	# number of components
            npoints = int(ln[1])  # number of vertices
            nfaces = int(ln[2]) # number of faces
            for k in range(ncomponents):
                fbyu.readline() # components (ignored)
            # read data
            self.vertices = np.empty([npoints, 2])
            k=-1
            while k < npoints-1:
                ln = fbyu.readline().split()
                k=k+1
                self.vertices[k, 0] = float(ln[0]) 
                self.vertices[k, 1] = float(ln[1]) 
                if len(ln) > 2:
                    k=k+1
                    self.vertices[k, 0] = float(ln[2])
                    self.vertices[k, 1] = float(ln[3]) 

            self.faces = np.empty([nfaces, 2])
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
                    if j == 2:
                        kf=kf+1
                        j=0
                ln = fbyu.readline().split()
        self.faces = np.int_(self.faces) - 1
        self.computeCentersLengths()

    #Saves in .byu format
    def saveCurve(self, outfile):
        #FV = readbyu(byufile)
        #reads from a .byu file into matlab's face vertex structure FV

        with open(outfile,'w') as fbyu:
            # copy header
            ncomponents = 1	    # number of components
            npoints = self.vertices.shape[0] # number of vertices
            nfaces = self.faces.shape[0]		# number of faces
            nedges = 2*nfaces		# number of edges

            str = '{0: d} {1: d} {2: d} {3: d} 0\n'.format(ncomponents, npoints, nfaces,nedges)
            fbyu.write(str) 
            str = '1 {0: d}\n'.format(nfaces)
            fbyu.write(str) 


            k=-1
            while k < (npoints-1):
                k=k+1 
                str = '{0: f} {1: f} '.format(self.vertices[k, 0], self.vertices[k, 1])
                fbyu.write(str) 
                if k < (npoints-1):
                    k=k+1
                    str = '{0: f} {1: f}\n'.format(self.vertices[k, 0], self.vertices[k, 1])
                    fbyu.write(str) 
                else:
                    fbyu.write('\n')

            j = 0 
            for k in range(nfaces):
                fbyu.write('{0: d} '.format(self.faces[k,0]+1))
                j=j+1
                if j==16:
                    fbyu.write('\n')
                    j=0

                fbyu.write('{0: d} '.format(-self.faces[k,1]-1))
                j=j+1
                if j==16:
                    fbyu.write('\n')
                    j=0

    # Saves in .vtk format 
    def saveVTK(self, fileName, scalars = None, normals = None, scal_name='scalars'):
        F = self.faces ;
        V = np.copy(self.vertices) ;
        if V.shape[1] == 2: 
            #print V.shape, np.zeros([V.shape[0], 1]).shape
            V = np.concatenate((V, np.zeros([V.shape[0], 1])), axis=1)


        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET POLYDATA\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll,0], V[ll,1], V[ll,2]))
            fvtkout.write('\nLINES {0:d} {1:d}'.format(F.shape[0], 3*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(F[ll,0], F[ll,1]))
            if (not (scalars is None)) | (not (normals is None)):
                fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            if not (scalars is None):
                fvtkout.write('\nSCALARS '+scal_name+' float 1\nLOOKUP_TABLE default')
                for ll in range(V.shape[0]):
                    fvtkout.write('\n {0: .5f}'.format(scalars[ll]))
            if not (normals is None):
                fvtkout.write('\nNORMALS normals float')
                for ll in range(V.shape[0]):
                    fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(normals[ll, 0], normals[ll, 1], 0))
            fvtkout.write('\n')


    # Reads .vtk file
    def readVTK(self, fileName):
        if gotVTK:
            u = vtkPolyDataReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfLines())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))
            if np.fabs(V[:,2]).max() < 1e-20:
                V = V[:,0:2]

            F = np.zeros([nfaces, 2])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(2):
                    F[kk,ll] = c.GetPointId(ll)
        
            self.vertices = V
            self.faces = np.int_(F)
            self.computeCentersLengths()
        else:
            raise Exception('Cannot read VTK files without VTK functions')

    def resample(self, ds):
        ll = np.sqrt((self.linel**2).sum(axis=1))
        if ll.max() < ds:
            return
        v = np.zeros([2*self.vertices.shape[0], self.vertices.shape[1]])  
        f = np.zeros([2*self.faces.shape[0], self.faces.shape[1]], dtype=int)
        v[0:self.vertices.shape[0],:] = self.vertices
        lv = self.vertices.shape[0]
        lf = 0
        for k in range(self.faces.shape[0]):  
            if ll[k] < ds:
                f[lf,:] = self.faces[k,:]
                lf += 1
            else:
                c = 0.5*(v[self.faces[k,0],:] + v[self.faces[k,1],:])
                v[lv, :] = c
                f[lf,:] = (self.faces[k,0], lv)
                f[lf+1,:] = (lv, self.faces[k,1])
                lv += 1
                lf += 2
        self.vertices = np.copy(v[0:lv,:])
        self.faces = np.copy(f[0:lf,:])
        self.computeCentersLengths()
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        print 'resampling', self.length(), self.vertices.shape[0]
        self.resample(ds)


    def laplacian(self, phi, weighted=False):
        res = np.zeros(phi.shape)
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        a = (np.sqrt((self.linel**2).sum(axis=1)))[...,np.newaxis]
        r1 = (phi2 -phi1)/a
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] -= r1[k,:]
        if weighted:
            av = self.computeVertexLength()
            return res/av[0]
        else:
            return res

#    def diffNormGrad(self, phi):
#        res = np.zeros((self.vertices.shape[0],phi.shape[1]))
#        phi1 = phi[self.faces[:,0],:]
#        phi2 = phi[self.faces[:,1],:]
#        a = np.sqrt((self.linel**2).sum(axis=1))
#        u = ((phi2-phi1)**2).sum(axis=1)
#        u = (u/a**3)[...,np.newaxis]
#        
#        r1 =  u * self.linel
#        for k,f in enumerate(self.faces):
#            res[f[0],:] += r1[k,:]
#            res[f[1],:] -= r1[k,:]
#        return res

    def diffH1Alpha(self, phi):
        res = np.zeros((self.vertices.shape[0],phi.shape[1]))
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        a = np.sqrt((self.linel**2).sum(axis=1))
        L = a.sum()
        u = ((phi2-phi1)**2).sum(axis=1)/L
        u0 = (u/a).sum()/L
        u = (u/a**3 - u0/a)[...,np.newaxis]
        
        r1 =  u * self.linel
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] -= r1[k,:]
        return res



def remesh(x, N=100, closed=True, rhoTol=0.9):
    if closed:
        x1 = np.append(np.copy(x), [x[0,:]], axis=0)
    else:
        x1 = x
    linel = x1[1:x1.shape[0], :]-x1[0:x1.shape[0]-1, :]
    ll = np.sqrt((linel**2).sum(axis=1))
    s = np.insert(ll.cumsum(),0,[0],axis=0)
    L = s[-1]
    if closed:
        ds = L/N
    else:
        ds = L/(N-1)
    v = np.zeros((1000,2))
    v[0,:] = x[0,:]
    lsofar = 0
    kx = 0
    n = s.shape[0]
    nx = x.shape[0]
    #fig = plt.figure(2)
    for k in range(1,1000):
        pred = v[k-1,:]
        ls = np.sqrt(((x1[kx,:]-v[k-1,:])**2).sum())
        rhos = 1.
        while kx < nx-1 and ls <= ds and rhos>=rhoTol:
            pred = x1[kx,:]
            kx += 1
            if kx < n:
                ls += ll[kx-1]
                lsofar += ll[kx - 1]
            rhos = np.sqrt(((x1[kx,:]-v[k-1,:])**2).sum())/ls
        if rhos < rhoTol:
            kx -= 1
            lsofar -= ll[kx - 1]
            v[k, :] = x1[kx,:]
            # fig.clf()
            # plt.plot(v[0:k+1, 0], v[0:k+1,1], color=[1,0,0], marker='*')
            # plt.plot(x1[0:kx+1, 0], x1[0:kx+1,1], color=[0,0,1], marker='o')
        elif ls > ds:
            b = ((pred-v[k-1,:])*(x1[kx,:]-pred)).sum()
            a = ((x1[kx,:]-pred)**2).sum()
            c = ((pred-v[k-1,:])**2).sum() - ds**2
            aa = (-b + np.sqrt(b**2-a*c))/a
            #print k,aa
            if aa<0:
                v[k,:] = pred
            elif aa > 1:
                v[k,:] = x1[kx,:]
            else:
                v[k,:] = pred + aa * (x1[kx,:]-pred)
            # fig.clf()
            # plt.plot(v[0:k+1, 0], v[0:k+1, 1], color=[1, 0, 0], marker='*')
            # plt.plot(x1[0:kx+1, 0], x1[0:kx+1, 1], color=[0, 0, 1], marker='o')
        else:
            if not closed:
                v[k,:] = x1[kx, :]
                v = v[0:k+1, :]
                # fig.clf()
                # plt.plot(v[0:k+1, 0], v[0:k+1,1], color=[1,0,0], marker='*')
            else:
                ls = np.sqrt(((x1[-1, :] - v[k-1, :]) ** 2).sum())
                q = int(np.floor(ls/ds))
                for kk in range(0,q):
                    v[k+kk,:] = v[k-1,:] + (kk+1)*(ds/ls)*(x1[-1,:] - v[k-1,:])
                ls = np.sqrt(((x1[-1, :] - v[k+q-1, :]) ** 2).sum())
                if ls < ds/2:
                    v = v[0:k+q-1, :]
                else:
                    v = v[0:k+q, :]
            #     fig.clf()
            #     plt.plot(v[:, 0], v[:, 1], color=[1, 0, 0], marker='*')
            # plt.plot(x1[0:kx+1, 0], x1[0:kx+1,1], color=[0,0,1], marker='o')
            break
        #lsofar += np.sqrt(((v[k, :] - v[k-1, :]) ** 2).sum())


    return v


def smoothCurve(f, N, eps = 0.01, constantLength = True, constantArea = True):
    res = Curve(curve=f)
    l = f.length()
    a = f.enclosedArea()
    s = -np.sign(a)
    a = np.fabs(a)
    for k in range(N):
        k1, n, kan = res.computeCurvature()
        res.updateVertices(res.vertices + s*eps * kan)
        if constantLength:
            rho = l / res.length()
            res.updateVertices(res.vertices * (rho))
        elif constantArea:
            rho = a/np.fabs(res.enclosedArea())
            res.updateVertices(res.vertices*np.sqrt(rho))
    return res



def mergecurves(curves, tol=0.01):
    N = 0
    M = 0
    dim = curves[0].vertices.shape[1]
    for c in curves:
        N += c.vertices.shape[0]
        M += c.faces.shape[0]

    vertices = np.zeros([N,dim])
    faces = np.zeros([M,dim], dtype=int)
    component = np.zeros(M, dtype=int)
    N = 0
    M = 0
    C = 0
    for c in curves:
        N1 = c.vertices.shape[0]
        M1 = c.faces.shape[0]
        vertices[N:N+N1,:] = c.vertices
        faces[M:M+M1, :] = c.faces + N
        component[M:M+M1, :] = c.component + C
        N += N1
        M += M1
        C += 1 + c.component.max()
        #print N,M
    dist = np.sqrt(((vertices[:, np.newaxis, :]-vertices[np.newaxis,:,:])**2).sum(axis=2))
    j=0
    openV = np.ones(N)
    refIndex = -np.ones(N)
    for k in range(N):
        if openV[k]:
            #vertices[j,:] = np.copy(vertices[k,:])
            J = np.nonzero((dist[k,:] < tol) * openV==1)
            J = J[0]
            openV[J] = 0
            refIndex[J] = j
            j=j+1
    vert2 = np.zeros([j, dim])
    for k in range(j):
        J = np.nonzero(refIndex==k)
        J = J[0]
        #print vertices[J]
        vert2[k,:] = vertices[J].sum(axis=0)/len(J)
        #print J, len(J), J.shape
    #vertices = vertices[0:j, :]
    #print faces
    faces = refIndex[faces]
    faces2 = np.copy(faces)
    comp2 = np.copy(component)
    j = 0
    for k in range(faces.shape[0]):
        if faces[k,1] != faces[k,0]:
            faces2[j,:] = faces[k,:]
            comp2[j] = component[k]
            j += 1
            #print k,j
    faces2 = faces2[range(j), :]
    comp2 = comp2[range(j)]
    res = Curve(FV=(faces2,vert2))
    res.component = comp2
    return res

# Reads several .byu files
def readMultiplecurves(regexp, Nmax = 0):
    files = glob.glob(regexp)
    if Nmax > 0:
        nm = min(Nmax, len(files))
    else:
        nm = len(files)
    fv1 = []
    for k in range(nm):
        fv1.append(Curve(files[k]))
    return fv1

# saves time dependent curves (fixed topology)
def saveEvolution(fileName, fv0, xt):
    fv = Curve(fv0)
    for k in range(xt.shape[0]):
        fv.vertices = np.squeeze(xt[k, :, :])
        fv.saveCurve(fileName+'{0: 02d}'.format(k)+'.byu')


def L2Norm0(fv1):
    #return ((fv1.vertices**2).sum(axis=1)*fv1.diffArcLength()).sum()
    return ((fv1.vertices**2).sum(axis=1)).sum()

def L2NormDef(fvDef, fv1):
    # a1 = fv1.diffArcLength()
    # aDef = fvDef.diffArcLength()
    # return (-2*(fvDef.vertices*fv1.vertices).sum(axis=1)*np.sqrt(a1*aDef) + (fvDef.vertices**2).sum(axis=1)*aDef ).sum()
    return (-2*(fvDef.vertices*fv1.vertices).sum(axis=1) + (fvDef.vertices**2).sum(axis=1) ).sum()

def L2NormGradient(fvDef,fv1):
    # a1 = fv1.diffArcLength()[:, np.newaxis]
    # aDef = fvDef.diffArcLength()[:, np.newaxis]
    # z1 = 2*(fvDef.vertices*aDef-fv1.vertices * np.sqrt(a1*aDef))
    z1 = 2*(fvDef.vertices-fv1.vertices)
    return z1

def L2Norm(fvDev, fv1):
    return L2NormDef(fvDev, fv1) + L2Norm0(fv1)


# Current norm of fv1
def currentNorm0(fv1, KparDist=None, weight=None):
    c2 = fv1.centers
    cr2 = fv1.linel
    obj = (cr2 * KparDist.applyK(c2, cr2)).sum()
    if weight:
        cr2n = np.sqrt((cr2**2).sum(axis=1))[:,np.newaxis]
        obj += weight* (cr2n * KparDist.applyK(c2, cr2n)).sum()
    return obj


# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct 
def currentNormDef(fvDef, fv1, KparDist=None, weight=None):
    c1 = fvDef.centers
    cr1 =fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.linel
    obj = ((cr1*KparDist.applyK(c1, cr1)).sum() - 2*(cr1 * KparDist.applyK(c2, cr2, firstVar=c1)).sum())
    if weight:
        cr1n = np.sqrt((cr1**2).sum(axis=1)+1e-10)[:,np.newaxis]
        cr2n = np.sqrt((cr2**2).sum(axis=1)+1e-10)[:,np.newaxis]
        obj += weight* ((cr1n * KparDist.applyK(c1, cr1n)).sum() - 2*(cr1n * KparDist.applyK(c2, cr2n, firstVar=c1)).sum())
    return obj

# Returns |fvDef - fv1|^2 for current norm
def currentNorm(fvDef, fv1, KparDist=None, weight=None):
    return currentNormDef(fvDef, fv1, KparDist, weight=weight) + currentNorm0(fv1, KparDist, weight=weight)

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentNormGradient(fvDef, fv1, KparDist=None, weight=None):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.linel
    dim = c1.shape[1]



    z1 = KparDist.applyK(c1, cr1) - KparDist.applyK(c2, cr2, firstVar=c1)
    dz1 = .5*(KparDist.applyDiffKT(c1, cr1[np.newaxis,...], cr1[np.newaxis,...]) -
            KparDist.applyDiffKT(c2, cr1[np.newaxis,...], cr2[np.newaxis,...], firstVar=c1))

    if weight:
        a1 = np.sqrt((cr1 ** 2).sum(axis=1) + 1e-10)
        a2 = np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10)
        cr1n = cr1 / a1[:, np.newaxis]
        z01 = (KparDist.applyK(c1, a1[:, np.newaxis]) - KparDist.applyK(c2, a2[:, np.newaxis], firstVar=c1))
        z1 += weight * (z01*cr1n)
        dz1 += (weight/2.) * (KparDist.applyDiffKT(c1, a1[np.newaxis,:,np.newaxis], a1[np.newaxis,:,np.newaxis]) -
                      KparDist.applyDiffKT(c2, a1[np.newaxis,:,np.newaxis], a2[np.newaxis,:,np.newaxis], firstVar=c1))




    px = np.zeros([xDef.shape[0], dim])


    I = fvDef.faces[:,0]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - z1[k, :]

    I = fvDef.faces[:,1]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[k, :]


    return 2*px






# Measure norm of fv1
def measureNorm0(fv1, KparDist=None):
    c2 = fv1.centers
    cr2 = fv1.linel
    cr2 = np.sqrt((cr2**2).sum(axis=1))[:,np.newaxis]
    return np.multiply(cr2, KparDist.applyK(c2, cr2)).sum()

    
# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct 
def measureNormDef(fvDef, fv1, KparDist=None):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    cr1 = np.sqrt((cr1**2).sum(axis=1)+1e-10)[:,np.newaxis]
    c2 = fv1.centers
    cr2 = fv1.linel
    cr2 = np.sqrt((cr2**2).sum(axis=1)+1e-10)[:,np.newaxis]
    obj = (np.multiply(cr1, KparDist.applyK(c1, cr1)).sum()
        - 2*np.multiply(cr1, KparDist.applyK(c2, cr2, firstVar=c1)).sum())
    return obj

# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist=None):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist) 


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist=None):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.linel
    dim = c1.shape[1]
    a1 = np.sqrt((cr1**2).sum(axis=1)+1e-10)
    a2 = np.sqrt((cr2**2).sum(axis=1)+1e-10)
    cr1 = cr1 / a1[:, np.newaxis]
    #cr2 = cr2 / a2[:, np.newaxis]


    z1 = KparDist.applyK(c1, a1[:, np.newaxis]) - KparDist.applyK(c2, a2[:, np.newaxis], firstVar=c1)
    z1 = np.multiply(z1, cr1)

    dz1 = (1./2.) * (KparDist.applyDiffKT(c1, a1[np.newaxis,:,np.newaxis], a1[np.newaxis,:,np.newaxis]) -
                      KparDist.applyDiffKT(c2, a1[np.newaxis,:,np.newaxis], a2[np.newaxis,:,np.newaxis], firstVar=c1))
    # dz1 = (np.multiply(dg11.sum(axis=1).reshape((-1,1)), c1) - np.dot(dg11,c1) - np.multiply(dg12.sum(axis=1).reshape((-1,1)), c1) + np.dot(dg12,c2))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]

    px = np.zeros([xDef.shape[0], dim])
    ###########

    I = fvDef.faces[:,0]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - z1[k, :]

    I = fvDef.faces[:,1]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[k, :]

    return 2*px

def _varifoldNorm0(c2, cr2, KparDist=None, weight=1.):
    d=weight
    a2 = np.sqrt((cr2**2).sum(axis=1)+1e-10)
    cr2 = cr2/a2[:,np.newaxis]
    cr2cr2 = (cr2[:,np.newaxis,:]*cr2[np.newaxis,:,:]).sum(axis=2)
    a2a2 = a2[:,np.newaxis]*a2[np.newaxis,:]
    beta2 = (1 + d*cr2cr2**2)*a2a2
    return KparDist.applyK(c2, beta2[...,np.newaxis], matrixWeights=True).sum()
        

def varifoldNorm0(fv1, KparDist=None, weight=1.):
    c2 = fv1.centers
    cr2 = fv1.linel
    return _varifoldNorm0(c2, cr2, KparDist=KparDist, weight=weight)

def varifoldNormComponent0(fv1, KparDist=None, weight=1.):
    c2 = fv1.centers
    cr2 = fv1.linel
    cp = fv1.component
    ncp = cp.max()+1
    obj = 0
    for k in range(ncp):
        I = np.nonzero(cp==k)[0]
        obj += _varifoldNorm0(c2[I], cr2[I], KparDist=KparDist, weight=weight)
    return obj


# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot product 
def _varifoldNormDef(c1, c2, cr1, cr2, KparDist=None, weight=1.):
    d=weight
    a1 = np.sqrt((cr1**2).sum(axis=1)+1e-10)
    a2 = np.sqrt((cr2**2).sum(axis=1)+1e-10)
    cr1 = cr1/a1[:,np.newaxis]
    cr2 = cr2/a2[:,np.newaxis]

    cr1cr1 = (cr1[:,np.newaxis,:]*cr1[np.newaxis,:,:]).sum(axis=2)
    a1a1 = a1[:,np.newaxis]*a1[np.newaxis,:]
    cr1cr2 = (cr1[:,np.newaxis,:]*cr2[np.newaxis,:,:]).sum(axis=2)
    a1a2 = a1[:,np.newaxis]*a2[np.newaxis,:]

    beta1 = (1 + d*cr1cr1**2)*a1a1
    beta2 = (1 + d*cr1cr2**2)*a1a2

    obj = (KparDist.applyK(c1, beta1[...,np.newaxis], matrixWeights=True).sum()
        - 2*KparDist.applyK(c2, beta2[...,np.newaxis], firstVar=c1, matrixWeights=True).sum())
    return obj

def varifoldNormDef(fvDef, fv1, KparDist=None, weight=1.):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.linel
    return _varifoldNormDef(c1, c2, cr1, cr2, KparDist=KparDist, weight=weight)

def varifoldNormComponentDef(fvDef, fv1, KparDist=None, weight=1.):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.linel
    cp1 = fvDef.component
    cp2 = fv1.component
    ncp = cp1.max()+1
    obj = 0
    for k in range(ncp):
        I1 = np.nonzero(cp1==k)[0]
        I2 = np.nonzero(cp2==k)[0]
        obj += _varifoldNormDef(c1[I1], c2[I2], cr1[I1], cr2[I2], KparDist=KparDist, weight=weight)
    return obj

# Returns |fvDef - fv1|^2 for current norm
def varifoldNorm(fvDef, fv1, KparDist=None, weight=1.):
    return varifoldNormDef(fvDef, fv1, KparDist=KparDist, weight=weight) + varifoldNorm0(fv1, KparDist=KparDist, weight=weight)

def varifoldNormComponent(fvDef, fv1, KparDist=None, weight=1.):
    return varifoldNormComponentDef(fvDef, fv1, KparDist=KparDist, weight=weight) + varifoldNormComponent0(fv1, KparDist=KparDist, weight=weight)

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def _varifoldNormGradient(c1, c2, cr1, cr2, KparDist=None, weight=1.):
    d=weight

    a1 = np.sqrt((cr1**2).sum(axis=1)+1e-10)
    a2 = np.sqrt((cr2**2).sum(axis=1)+1e-10)
    cr1 = cr1 / a1[:, np.newaxis]
    cr2 = cr2 / a2[:, np.newaxis]
    cr1cr1 =  (cr1[:, np.newaxis, :] * cr1[np.newaxis, :, :]).sum(axis=2)
    cr1cr2 =  (cr1[:, np.newaxis, :] * cr2[np.newaxis, :, :]).sum(axis=2)

    beta1 = a1[:,np.newaxis]*a1[np.newaxis,:] * (1 + d*cr1cr1**2) 
    beta2 = a1[:,np.newaxis]*a2[np.newaxis,:] * (1 + d*cr1cr2**2)

    u1 = (2*d*cr1cr1[...,np.newaxis]*cr1[np.newaxis,...] - d*(cr1cr1**2)[...,np.newaxis]*cr1[:,np.newaxis,:]
          + cr1[:,np.newaxis,:])*a1[np.newaxis,:,np.newaxis]
    u2 = (2*d*cr1cr2[...,np.newaxis]*cr2[np.newaxis,...] - d*(cr1cr2**2)[...,np.newaxis]*cr1[:,np.newaxis,:]
          + cr1[:,np.newaxis,:])*a2[np.newaxis,:,np.newaxis]

    z1 = KparDist.applyK(c1, u1,matrixWeights=True) - KparDist.applyK(c2, u2, firstVar=c1, matrixWeights=True)
    #print a1.shape, c1.shape
    dz1 = (1./2.) * (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))
                        
    return z1,dz1

def varifoldNormGradient(fvDef, fv1, KparDist=None, weight=1.):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.linel
    foo = _varifoldNormGradient(c1, c2, cr1, cr2, KparDist=KparDist, weight=weight)
    z1 = foo[0]
    dz1 = foo[1]
    dim = c1.shape[1]

    px = np.zeros([fvDef.vertices.shape[0], dim])
    #I = fvDef.faces[:,0]
    #crs = np.cross(xDef3 - xDef2, z1)
    for k in range(fvDef.faces.shape[0]):
        px[fvDef.faces[k,0], :] += dz1[k, :] - z1[k,:]
        px[fvDef.faces[k,1], :] += dz1[k, :] + z1[k,:]

    return 2*px

def varifoldNormComponentGradient(fvDef, fv1, KparDist=None, weight=1.):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.linel
    cp1 = fvDef.component
    cp2 = fv1.component
    ncp = cp1.max()+1
    dim = c1.shape[1]

    z1 = np.zeros(c1.shape)
    dz1 = np.zeros(c1.shape)
    for k in range(ncp):
        I1 = np.nonzero(cp1==k)[0]
        I2 = np.nonzero(cp2==k)[0]
        foo = _varifoldNormGradient(c1[I1], c2[I2], cr1[I1], cr2[I2], KparDist=KparDist, weight=weight)
        z1[I1,:] = foo[0]
        dz1[I1,:] = foo[1]

    px = np.zeros([fvDef.vertices.shape[0], dim])
    for k in range(fvDef.faces.shape[0]):
        px[fvDef.faces[k,0], :] += dz1[k, :] - z1[k,:]
        px[fvDef.faces[k,1], :] += dz1[k, :] + z1[k,:]

    return 2*px

def normGrad___(fv, phi):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a = np.sqrt((fv.linel**2).sum(axis=1))
    #print min(a)
    res = (((phi2-phi1)**2).sum(axis=1)/a).sum()
    return res

def normGrad(fv, phi, weight=1.0):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a = np.sqrt((fv.linel**2).sum(axis=1))
    a1 = a[:, np.newaxis]
    tau = fv.linel
    nu = np.zeros(tau.shape)
    nu[:,0] = -tau[:,1]
    nu[:,1] = tau[:,0]
    #print min(a)
    dphi = (phi2-phi1)
    res = (((dphi*tau).sum(axis=1)**2 + weight*(dphi*nu).sum(axis=1)**2)/(a**3))
    #res0 = ((((dphi)*(tau/a1)).sum(axis=1)**2 + weight*((dphi)*(nu/a1)).sum(axis=1)**2)/(a))
    #res1 = (((dphi*tau).sum(axis=1)**2/(a**4) + weight*(dphi*nu).sum(axis=1)**2/(a**4))*(a))
    #res2 = normGrad___(fv, phi)
    res = res.sum()
    return res
    
def normGradInvariant(fv, phi):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a = np.sqrt((fv.linel**2).sum(axis=1))
    nl = fv.computeUnitFaceNormals()
    res = (((phi2-phi1)**2).sum(axis=1)/a).sum()
    nc = 1+fv.component.max()
    #print fv.faces.shape, phi.shape, nl.shape
    for comp in range(nc):
        I = np.nonzero(fv.component==comp)[0]
        c = ((phi2[I,:] - phi1[I,:])*nl[I,:]).sum()
        res -= (c**2)/a[I].sum()
    return res

def normGradInvariant3D(fv, phi):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a = np.sqrt((fv.linel**2).sum(axis=1))
    tl = fv.linel/a[...,np.newaxis]
    res = (((phi2-phi1)**2).sum(axis=1)/a).sum()
    nc = 1+fv.component.max()
    for comp in range(nc):
        I = np.nonzero(fv.component==comp)[0]
        M = ((a[I].sum()+1e-6)*np.eye(3) 
            - (fv.linel[I,:,np.newaxis]*tl[I,np.newaxis,:]).sum(axis=0))
        c = np.cross(phi2[I,:]-phi1[I,:], tl[I,:]).sum(axis=0)
        res -= (np.linalg.solve(M,c)*c).sum()
    #print res
    return res

def h1AlphaNorm(fv, phi):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    dphi = phi2-phi1
    a = np.sqrt((fv.linel**2).sum(axis=1))
    tl = fv.linel/a[...,np.newaxis]
    nc = 1+fv.component.max()
    res = 0 
    for comp in range(nc):
        I = np.nonzero(fv.component==comp)[0]
        #a = np.sqrt((fv.linel[I,:]**2).sum(axis=1))
        L = a[I].sum()
        res += ((dphi[I,:]**2).sum(axis=1)/a[I]).sum()/L
        res -= ((dphi[I,:]*tl[I,:]).sum()/L)**2
    return res
    
def h1AlphaNormInvariant(fv, phi):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    dphi = phi2-phi1
    a = np.sqrt((fv.linel**2).sum(axis=1))
    tl = fv.linel/a[...,np.newaxis]
    nl = np.zeros(tl.shape)
    nl[:,0] = -tl[:,1]
    nl[:,1] = tl[:,0]
    nc = 1+fv.component.max()
    res = 0 
    for comp in range(nc):
        I = np.nonzero(fv.component==comp)[0]
        L = a[I].sum()
        res += ((dphi[I,:]**2).sum(axis=1)/a[I]).sum()/L
        res -= ((dphi[I,:]*tl[I,:]).sum()/L)**2
        res -= ((dphi[I,:]*nl[I,:]).sum()/L)**2
    return res
    
def diffNormGradInvariant(fv, phi, variables='both'):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a2 = (fv.linel**2).sum(axis=1)
    a = np.sqrt(a2)
    tl = fv.linel/a[...,np.newaxis]
    nl = np.zeros(tl.shape)
    nl[:,0] = -tl[:,1]
    nl[:,1] = tl[:,0]
    dphi = phi2-phi1
    dphia = dphi/a[...,np.newaxis]
#    c = (dphi*nl).sum()
#    L = a.sum()
    nc = 1+fv.component.max()
    if variables == 'both' or variables == 'phi':
        r1 = 2 * dphia 
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            c = 2*(dphi[I,:]*nl[I,:]).sum()
            L = a[I].sum()
            r1[I,:] -= (c/L)*nl[I,:]
        gradphi = np.zeros(phi.shape)
        for k,f in enumerate(fv.faces):
            gradphi[f[0],:] -= r1[k,:]
            gradphi[f[1],:] += r1[k,:]
    
    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        r1 = -((dphia**2).sum(axis=1))[...,np.newaxis] * tl
        #nc = fv.component.max()
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            c = (dphi[I,:]*nl[I,:]).sum()
            L = a[I].sum()
            r1[I,:] += 2*(c/L)*((dphia[I,:]*tl[I,:]).sum(axis=1)[...,np.newaxis]*nl[I,:])
            r1[I,:] += ((c/L)**2)*tl[I,:]
        for k,f in enumerate(fv.faces):
            gradx[f[0],:] -= r1[k,:]
            gradx[f[1],:] += r1[k,:]
    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        print 'Incorrect option in diffNormGradInvariant'


def diffNormGradInvariant3D(fv, phi, variables='both'):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a2 = (fv.linel**2).sum(axis=1)
    a = np.sqrt(a2)
    tl = fv.linel/a[...,np.newaxis]
    dphi = phi2-phi1
    dphia = dphi/a[...,np.newaxis]
    M = ((a.sum()+1e-6)*np.eye(3) 
    - (fv.linel[:,:,np.newaxis]*tl[:,np.newaxis,:]).sum(axis=0))
    c = np.cross(dphi, tl).sum(axis=0)
    Mc = np.linalg.solve(M,c)
    #print Mc
    tlmc = (tl*Mc[np.newaxis,:]).sum(axis=1)
    nc = 1+fv.component.max()

    if variables == 'both' or variables == 'phi':
        r1 = 2 * dphia
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            M = ((a[I].sum()+1e-6)*np.eye(3) 
                - (fv.linel[I,:,np.newaxis]*tl[I,np.newaxis,:]).sum(axis=0))
            c = np.cross(dphi[I,:], tl[I,:]).sum(axis=0)
            Mc = np.linalg.solve(M,c)
            #print Mc
            r1[I,:] -= 2*np.cross(tl[I,:], Mc[np.newaxis, :])
        gradphi = np.zeros(phi.shape)
        for k,f in enumerate(fv.faces):
            gradphi[f[0],:] -= r1[k,:]
            gradphi[f[1],:] += r1[k,:]
    
    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        r1 = -((dphia**2).sum(axis=1))[...,np.newaxis] * tl
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            M = ((a[I].sum()+1e-6)*np.eye(3) 
                - (fv.linel[I,:,np.newaxis]*tl[I,np.newaxis,:]).sum(axis=0))
            c = np.cross(dphi[I,:], tl[I,:]).sum(axis=0)
            Mc = np.linalg.solve(M,c)
            #print Mc
            tlmc = (tl[I,:]*Mc[np.newaxis,:]).sum(axis=1)
            r1[I,:] -= 2 * np.cross(Mc[np.newaxis, :], dphia[I,:])
            r1[I,:] += 2*(Mc*np.cross(dphia[I,:], tl[I,:])).sum(axis=1)[:, np.newaxis] * tl[I,:]
            r1[I,:] += (Mc**2).sum()*tl[I,:]
            r1[I,:] -= 2 * (tlmc)[:,np.newaxis] * Mc[np.newaxis,:]
            r1[I,:] += (tlmc**2)[:,np.newaxis] * tl[I,:]
            
        for k,f in enumerate(fv.faces):
            gradx[f[0],:] -= r1[k,:]
            gradx[f[1],:] += r1[k,:]
    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        print 'Incorrect option in diffNormGradInvariant'

def diffNormGrad(fv, phi, variables='both', weight=1.):
    gradphi = None
    gradx = None
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a2 = (fv.linel**2).sum(axis=1)
    a = np.sqrt(a2)[:, np.newaxis]
    tau = fv.linel
    nu = np.zeros(tau.shape)
    nu[:,0] = -tau[:,1]
    nu[:,1] = tau[:,0]
    dphi = phi2-phi1
    dphiperp = np.zeros(dphi.shape)
    dphiperp[:,0] = dphi[:,1]
    dphiperp[:,1] = -dphi[:,0]
    dphitau = (dphi*tau).sum(axis=1)[:, np.newaxis]
    dphinu = (dphi*nu).sum(axis=1)[:, np.newaxis]

    #dphia = dphi/a[...,np.newaxis]
    if variables == 'both' or variables == 'phi':
        r1 = 2 * (dphitau*tau + weight*dphinu*nu) / a**3
        gradphi = np.zeros(phi.shape)
        for k,f in enumerate(fv.faces):
            gradphi[f[0],:] -= r1[k,:]
            gradphi[f[1],:] += r1[k,:]
    
    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        r1 = 2 * (dphitau*dphi + weight*dphinu*dphiperp) / a**3
        r1 -= 3 * (dphitau**2 + weight * dphinu**2) * tau / a**5
        for k,f in enumerate(fv.faces):
            gradx[f[0],:] -= r1[k,:]
            gradx[f[1],:] += r1[k,:]
            
    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        print 'Incorrect option in diffNormGrad'

def diffH1Alpha(fv, phi, variables='both'):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a2 = (fv.linel**2).sum(axis=1)
    a = np.sqrt(a2)
    tl = fv.linel/a[...,np.newaxis]
    nl = np.zeros(tl.shape)
    nl[:,0] = -tl[:,1]
    nl[:,1] = tl[:,0]
    dphi = phi2-phi1
    dphia = dphi/a[...,np.newaxis]
    nc = 1+fv.component.max()
    if variables == 'both' or variables == 'phi':
        r1 = 2 * dphia
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            L = a[I].sum()
            c = 2*(dphi[I,:]*tl[I,:]).sum()/L
            r1[I,:] = r1[I,:] / L - (c/L)*tl[I]
        gradphi = np.zeros(phi.shape)
        for k,f in enumerate(fv.faces):
            gradphi[f[0],:] -= r1[k,:]
            gradphi[f[1],:] += r1[k,:]
    
    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        r1 = -((dphia**2).sum(axis=1))[...,np.newaxis] * tl
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            L = a[I].sum()
            c = (dphi[I,:]*tl[I,:]).sum()/L
            r1[I,:] = r1[I,:]/L - (((dphi[I,:]**2).sum(axis=1)/a[I]).sum()/L**2) * (tl[I,:])
            r1[I,:] -= 2*(c/L)*((dphia[I,:]*nl[I,:]).sum(axis=1)[...,np.newaxis]*nl[I,:]-(c)*tl[I,:])
        for k,f in enumerate(fv.faces):
            gradx[f[0],:] -= r1[k,:]
            gradx[f[1],:] += r1[k,:]
            
    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        print 'Incorrect option in diffH1Alpha'
        
def diffH1AlphaInvariant(fv, phi, variables='both'):
    phi1 = phi[fv.faces[:,0],:]
    phi2 = phi[fv.faces[:,1],:]
    a2 = (fv.linel**2).sum(axis=1)
    a = np.sqrt(a2)
    tl = fv.linel/a[...,np.newaxis]
    nl = np.zeros(tl.shape)
    nl[:,0] = -tl[:,1]
    nl[:,1] = tl[:,0]
    dphi = phi2-phi1
    dphia = dphi/a[...,np.newaxis]
    nc = 1+fv.component.max()
    if variables == 'both' or variables == 'phi':
        r1 = 2 * dphia
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            c1 = 2*(dphi[I,:]*tl[I,:]).sum()
            c2 = 2*(dphi[I,:]*nl[I,:]).sum()
            L = a[I].sum()
            r1[I,:] = r1[I,:]/L - (c1/L**2)*tl[I,:] - (c2/L**2)*nl[I,:]
        gradphi = np.zeros(phi.shape)
        for k,f in enumerate(fv.faces):
            gradphi[f[0],:] -= r1[k,:]
            gradphi[f[1],:] += r1[k,:]
    
    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        r1 = -((dphia**2).sum(axis=1))[...,np.newaxis] * tl
        for comp in range(nc):
            I = np.nonzero(fv.component==comp)[0]
            L = a[I].sum()
            c1 = (dphi[I,:]*tl[I,:]).sum()/L
            c2 = (dphi[I,:]*nl[I,:]).sum()/L
            r1[I,:] = r1[I,:]/L - (((dphi[I,:]**2).sum(axis=1)/a[I]).sum()/L**2) * (tl[I,:])
            r1[I,:] -= 2*(c1/L)*((dphia[I,:]*nl[I,:]).sum(axis=1)[...,np.newaxis]*nl[I,:]-(c1)*tl[I,:])
            r1[I,:] += 2*(c2/L)*((dphia[I,:]*tl[I,:]).sum(axis=1)[...,np.newaxis]*nl[I,:]+(c2)*tl[I,:])
            #r1[I,:] += 2*(c/L)*((dphia[I,:]*tl[I,:]).sum(axis=1)[...,np.newaxis]*nl[I,:])
            #r1[I,:] += ((c/L)**2)*tl[I,:]
        for k,f in enumerate(fv.faces):
            gradx[f[0],:] -= r1[k,:]
            gradx[f[1],:] += r1[k,:]
            
    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        print 'Incorrect option in diffH1AlphaInvariant'

