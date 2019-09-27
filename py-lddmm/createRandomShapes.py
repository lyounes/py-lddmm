from base.surfaces import *
import numpy as np
import scipy.linalg as la
import vtk
from base import diffeo
from vtk.util import numpy_support

def compute():
    fv0 = Surface(filename='/Users/younes/Development/Data/sculptris/shape1.obj')
    fv0.updateVertices(10*fv0.vertices)

    center = fv0.vertices.sum(axis=0)/fv0.vertices.shape[0]
    
    a = np.sign(fv0.vertices[:,0] - center[0] - 5) + 1
    a = a[:,np.newaxis]
    print(fv0.normGrad(a))
    for k in range(500):
        a += 0.0001 * fv0.laplacian(a, weighted=False)
    print(fv0.normGrad(a))

    fv0.saveVTK('/Users/younes/Development/Data/sculptris/Dataset/template_surf.vtk')
    img = fv0.compute3DVolumeImage(xmin=-50, xmax=50, origin = np.array([0,0,0]))
    diffeo.gridScalars(data=img).saveImg(
        '/Users/younes/Development/Data/sculptris/Dataset/template_img.vtk')

    print(fv0.surfArea())
    npx = np.arange(-50, 51, 1, dtype=int)
    ln = len(npx)
    npgrid = np.meshgrid(npx, npx, npx)
    x = npgrid[0].ravel()
    y = npgrid[1].ravel()
    z = npgrid[2].ravel()

    points = vtk.vtkPoints()
    for i in range(len(x)):
        points.InsertNextPoint(x[i],y[i],z[i])

    # #grid = numpy_support.numpy_to_vtk(npgrid.ravel(), deep=True) # ,deep=0 ,array_type=None)
    # grid = vtk.vtkRectilinearGrid()
    # grid.SetDimensions(len(npx) ,len(npx) ,len(npx))
    # grid.SetXCoordinates(x)
    # grid.SetYCoordinates(x)
    # grid.SetZCoordinates(x)

    
    for l in range(100):
        n = -.25*np.random.uniform(size=(fv0.vertices.shape[0],1))
        for k in range(100):
            n += 0.0001 * fv0.laplacian(n, weighted=False)
        fv1 = Surface(surf=fv0)
        for k in range(1000):
             fv1.updateVertices(fv1.vertices + 0.003 * (n) * fv1.meanCurvatureVector())
        fv1.smooth()
        A = 0.25*np.random.randn(3 ,3)
        R = la.expm((A - A.T) / 2)
        b = 10*np.random.randn(1 ,3)
        fv1.updateVertices(np.dot(fv1.vertices,R) + b)

        print(fv1.surfArea())

        img = fv1.compute3DVolumeImage(xmin=-50, xmax=50, origin = np.array([0,0,0]))
        #
        # select = vtk.vtkSelectEnclosedPoints()
        # select.SetSurfaceData(fv1.toPolyData())
        # vpoints = vtk.vtkPolyData()
        #
        # whiteImage = vtk.vtkImageData()
        # spacing = 1.
        # whiteImage.SetSpacing(spacing,spacing,spacing)
        # whiteImage.SetDimensions(ln,ln,ln)
        # whiteImage.SetExtent(0, ln-1, 0, ln-1, 0, ln-1)
        # bounds = vpoints.GetBounds()
        # origin = bounds[(0,2,4)] + spacing/2
        # whiteImage.SetOrigin(origin[0], origin[1], origin[2])
        # whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1)
        # count = whiteImage.GetNumberOfPoints()
        # inval = 255
        # outval =0
        # for i in range(count):
        #     whiteImage.GetPointData().GetScalars().SetTuple1(i,inval)
        #
        # pol2stenc = vtk.vtkPolyDataToImageStencil()
        # pol2stenc.SetInputData(vpoints)
        # pol2stenc.SetOutputOrigin(origin)
        # pol2stenc.SetOutputSpacing(spacing)
        # pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        # pol2stenc.Update()
        #
        # img2stenc = vtk.vtkImageStencil()
        # img2stenc.SetInputData(whiteImage)
        # img2stenc.SetStencilConnection(pol2stenc.GetOutputPort())
        # img2stenc.ReverseStencilOff()
        # img2stenc.SetBackgroundValue(outval)
        # img2stenc.Update()
        #
        # dataToStencil2.SetInputData(fv1.toPolyData())
        # dataToStencil2.Update()
        # stencil2 = vtk.vtkImageStencil()
        # stencil2.SetInputData(dataToStencil2.GetOutput())
        # stencil2.Update()
        # select.SetInputData(vpoints)
        # select.Update()
        # img = np.zeros((ln,ln,ln))
        # ii = 0
        # for i in range(ln):
        #     for j in range(ln):
        #         for k in range(ln):
        #             img[i,j,k] = select.IsInside(ii)
        #             ii += 1

        diffeo.gridScalars(data=img).saveImg('/Users/younes/Development/Data/sculptris/Dataset/subject_img'+str(l+1)+'.vtk')




if __name__=="__main__":
    compute()