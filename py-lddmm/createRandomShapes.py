from base.surfaces import *

def compute():
    fv0 = Surface(filename='/Users/younes/Development/Data/sculptris/shape1.obj')
    fv0.updateVertices(10*fv0.vertices)

    center = fv0.vertices.sum(axis=0)/fv0.vertices.shape[0]
    
    a = np.sign(fv0.vertices[:,0] - center[0] - 5) + 1
    a = a[:,np.newaxis]
    print fv0.normGrad(a) 
    for k in range(500):
        a += 0.0001 * fv0.laplacian(a, weighted=False)
    print fv0.normGrad(a)

    #fv0.saveVTK('/Users/younes/Data/sculptris/Atrophy/baseline.vtk', scal_name='atrophy', scalars=np.squeeze(a))
    
    print fv0.surfArea()
    
    for l in range(10):
        n = -.25*np.random.uniform(size=(fv0.vertices.shape[0],1))
        for k in range(100):
            n += 0.0001 * fv0.laplacian(n, weighted=False)
        fv1 = Surface(surf=fv0)
        for k in range(1000):
             fv1.updateVertices(fv1.vertices + 0.005 * (n) * fv1.meanCurvatureVector())
        fv1.smooth()
        print fv1.surfArea()
        fv1.saveVTK('/Users/younes/Development/Data/sculptris/Dataset/surface'+str(l+1)+'.vtk')

if __name__=="__main__":
    compute()