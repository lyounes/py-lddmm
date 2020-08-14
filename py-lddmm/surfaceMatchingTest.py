import numpy as np
from base import surfaces
from base import loggingUtils
from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.surfaceMatching import SurfaceMatching, SurfaceMatchingParam
import matplotlib.pyplot as plt
plt.ion()
def compute(model):

    loggingUtils.setup_default_logging('', stdOutput = True)
    sigmaKernel = 0.5
    sigmaDist = 5.
    sigmaError = 1.
    internalWeight = 200.
    internalCost = 'h1'
    if model=='Balls':
        M=100
        [x,y,z] = np.mgrid[0:2*M, 0:2*M, 0:2*M]/float(M)
        y = y-1
        z = z-1
        s2 = np.sqrt(2)

        I1 = .06 - ((x-.50)**2 + 0.5*y**4 + z**2)  
        fv1 = Surface()
        fv1.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        #return fv1
        
        u = (z + y)/s2
        v = (z - y)/s2
        I1 = np.maximum(0.05 - (x-.7)**2 - 0.5*y**2 - z**2, 0.02 - (x-.50)**2 - 0.5*y**2 - z**2)  
        #I1 = .05 - np.minimum((x-.7)**2 + 0.5*y**2 + z**2, (x-.30)**2 + 0.5*y**2 + z**2)  
        #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 
        fv2 = Surface()
        fv2.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        #fv1.saveVTK('/cis/home/younes/MorphingData/fv1.vtk')
        #fv2.saveVTK('/cis/home/younes/MorphingData/fv2.vtk')
        ftemp = fv1 
        ftarg = fv2
    elif model=='Hearts':
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        ay = np.fabs(y-1)
        az = np.fabs(z-1)
        ax = np.fabs(x-0.5)
        s2 = np.sqrt(2)
        c1 = np.sqrt(0.06)
        c2 = np.sqrt(0.045)
        c3 = 0.1

        I1 = np.minimum(c1**2 - (ax**2 + 0.5*ay**2 + az**2), np.minimum((ax**2 + 0.5*ay**2 + az**2)-c2**2, 1+c3-y)) 
        fv1 = Surface()
        fv1.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)


        #return fv1

        #s1 = 1.375
        #s2 = 2
        s1 = 1.1
        s2 = 1.2
        p = 1.75
        I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        fv2 = Surface()
        fv2.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv2.vertices[:,1] += 15 - 15/s1

        s1 *= 1.1
        s2 *= 1.2
        I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        fv3 = Surface()
        fv3.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv3.vertices[:,1] += 15 - 15/s1

        
        fv1.saveVTK('/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Fibers/fv2.vtk')
        fv3.saveVTK('/Users/younes/Development/Results/Fibers/fv3.vtk')
        ftemp = fv1
        ftarg = fv3
    elif model=='Hippo1':
        path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
        #sub1 = '0186193_1_6'
        #sub2 = '1449400_1_L'
        sub2 = 'LU027_R_sumNCBC20100628'
        fv1 = surfaces.Surface(filename =path + '5_population_template_qc/newTemplate.byu')
        v1 = fv1.surfVolume()
        #f0.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
        fv2 = surfaces.Surface(filename =path + '2_qc_flipped_registered/' + sub2 + '_registered.byu')
        v2 = fv2.surfVolume()
        if (v2*v1 < 0):
            fv2.faces = fv2.faces[:, [0,2,1]]
        ftemp = fv1 
        ftarg = fv2
    elif model=='Hippo2':
        #path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
        path = '/Volumes/CIS/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
        #sub1 = '0186193_1_6'
        #sub2 = '1449400_1_L'
        sub2 = 'LU027_R_sumNCBC20100628'
        fv1 = surfaces.Surface(filename =path + '5_population_template_qc/newTemplate.byu')
        v1 = fv1.surfVolume()
        #f0.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
        fv2 = surfaces.Surface(filename =path + '2_qc_flipped_registered/' + sub2 + '_registered.byu')
        v2 = fv2.surfVolume()
        if (v2*v1 < 0):
            fv2.faces = fv2.faces[:, [0,2,1]]
        ftemp = fv1 
        ftarg = fv2
    elif model=='biocardMulti':
        path = '/cis/home/younes/MorphingData/biocard/'
        #path = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
        #path2 = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/amygdala/6_mappings_baseline_template_all/0_template_to_all/' ;
        #path = '/cis/home/younes/MorphingData/Biocard/'
        sub1 = '0186193_1_6'
        #sub2 = '1449400_1_L'
        sub2 = '1229175_2_4'
        ftemp = []
        ftemp.append(surfaces.Surface(filename =path + 'Atlas_hippo_L_separate.byu'))
        ftemp.append(surfaces.Surface(filename =path + 'Atlas_amyg_L_separate.byu'))
        ftemp.append(surfaces.Surface(filename =path + 'Atlas_ent_L_up_separate.byu'))
        ftarg = []
        ftarg.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_hippo_L_qc_pass1_daniel2_reg.vtk'))
        ftarg.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_amyg_L_qc_pass1_daniel2_reg.vtk'))
        ftarg.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_ec_L_qc_pass1_daniel2_reg.vtk'))
    elif model=='KCrane':
        ftemp = surfaces.Surface(filename='/Users/younes/Development/Data/KCrane/blub/blub_triangulated_reduced.obj')
        #ftemp.Simplify(1000, deciPro=True)
        ftarg = surfaces.Surface(filename='/Users/younes/Development/Data/KCrane/spot/spot_triangulated_reduced.obj')
        #ftarg.Simplify(1000, deciPro=True)
        R0, T0 = rigidRegistration(surfaces = (ftarg.vertices, ftemp.vertices),  rotWeight=0., verb=False, temperature=10., annealing=True)
        ftarg.updateVertices(np.dot(ftarg.vertices, R0.T) + T0)
        sigmaKernel = 0.5
        sigmaDist = 5.
        sigmaError = 0.01
        internalWeight = 10.
    elif model=='snake':
        M=100
        [x,y,z] = np.mgrid[0:2*M, 0:2*M, 0:2*M]/float(M)
        x = x-1
        y = y-1
        z = z-1
        t = np.arange(-0.5, 0.5, 0.01)

        r = .3
        c = .95
        delta = 0.05
        h = 0.25
        f1 = np.zeros((t.shape[0],3))
        f1[:,0] = r*np.cos(2*np.pi*c*t) -r
        f1[:,1] = r*np.sin(2*np.pi*c*t)
        fig = plt.figure(4)
        f1[:,2] = h*t
        fig.clf()
        ax = fig.gca(projection='3d')
        ax.plot(f1[:,0], f1[:,1], f1[:,2])

        f2 = np.zeros((t.shape[0],3))
        f2[:,0] = r*np.cos(2*np.pi*c*t)-r
        f2[:,1] = r*np.sin(2*np.pi*c*t)
        f2[:,2] = -h*t
        ax.plot(f2[:,0], f2[:,1], f2[:,2])
        ax.axis('equal')
        plt.pause((0.1))

        dst = (x[..., np.newaxis] - f1[:,0])**2 + (y[..., np.newaxis] - f1[:,1])**2 + (z[..., np.newaxis] - f1[:,2])**2
        dst = np.min(dst, axis=3)
        ftarg = Surface()
        ftarg.Isosurface((dst < delta**2), value=0.5)
        dst = (x[..., np.newaxis] - f2[:,0])**2 + (y[..., np.newaxis] - f2[:,1])**2 + (z[..., np.newaxis] - f2[:,2])**2
        dst = np.min(dst, axis=3)
        ftemp = Surface()
        ftemp.Isosurface((dst < delta**2), value=0.5)
        sigmaKernel = np.array([1,5,10])
        sigmaDist = 10.
        sigmaError = .1
        internalWeight = 5.
        internalCost = None
    else:
        return

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = sigmaKernel)

    sm = SurfaceMatchingParam(timeStep=0.1, algorithm='cg', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                              errorType='varifold', internalCost=internalCost)
    f = SurfaceMatching(Template=ftemp, Target=ftarg, outputDir='/Users/younes/Development/Results/'+model+'LDDMM5p0H5000',param=sm,
                        testGradient=True,
                        #subsampleTargetSize = 500,
                         internalWeight=internalWeight, maxIter=1000, affine= 'none', rotWeight=.01, transWeight = .01, scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


if __name__=="__main__":
    compute('Hippo1')
