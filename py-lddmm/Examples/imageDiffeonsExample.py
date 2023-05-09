from sys import path as sys_path
sys_path.append('..')
from scipy.ndimage import gaussian_filter
from base.gridscalars import GridScalars
from base.kernelFunctions import *
from base.gaussianDiffeonsImageMatching import ImageMatchingDiffeons
from base import loggingUtils

def compute(createImages=True):
    if createImages:
        [x,y] = np.mgrid[0:100, 0:100]/50.
        x = x-1
        y = y-1

        I1 = gaussian_filter(255*np.array(.06 - ((x)**2 + 1.5*y**2) > 0), 1)
        im1 = GridScalars(grid = I1, dim=2)

        #return fv1
        
        I2 = gaussian_filter(255*np.array(.05 - np.minimum((x-.2)**2 + 1.5*y**2, (x+.20)**2 + 1.5*y**2) > 0), 1)
        im2 = GridScalars(grid = I2, dim=2)
        #I1 = .06 - ((x-.50)**2 + 0.75*y**2 + z**2)  
        #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 

        # im1.saveImg('../Output/Diffeons/Images/im1.png', normalize=True)
        # im2.saveImg('/Users/younes/Development/Results/Diffeons/Images/im2.png', normalize=True)
    else:
        # if True:
        path = '/Users/younes/IMAGES/'
        #im1 = gridScalars(fileName = path+'database/camel07.pgm', dim=2)
        #im2 = gridScalars(fileName = path+'database/camel08.pgm', dim=2)
        path = '/Volumes/younes/IMAGES/'
        # #im1 = gridScalars(fileName = path+'database/camel07.pgm', dim=2)
        # #im2 = gridScalars(fileName = path+'database/camel08.pgm', dim=2)
        # #im1 = gridScalars(fileName = path+'yalefaces/subject01.normal.gif', dim=2)
        # #im2 = gridScalars(fileName = path+'yalefaces/subject01.happy.gif', dim=2)
        im1 = GridScalars(grid = path+'heart/heart01.tif', dim=2)
        im2 = GridScalars(grid = path+'heart/heart09.tif', dim=2)
        im1.data = gaussian_filter(im1.data, .5)
        im2.data = gaussian_filter(im2.data, .5)
        #im1 = gridScalars(fileName = path+'image_0031.jpg', dim=2)
        #im2 = gridScalars(fileName = path+'image_0043.jpg', dim=2)
        #im2.saveImg('/Users/younes/Development/Results/Diffeons/Images/imTest.png', normalize=True)
        print(im2.data.max())
        # else:
        #     #f1.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub2+'_amyg_L.byu'))
        #     im1 = GridScalars(grid='/Users/younes/Development/Results/Diffeons/Images/im1.png', dim=2)
        #     im2  = GridScalars(grid='/Users/younes/Development/Results/Diffeons/Images/im2.png', dim=2)

        #return fv1, fv2

    ## Object kernel

    options = {
        'timeStep': 0.05,
        'sigmaKernel': 5.,
        'sigmaError':10.,
        'outputDir': '../Output/ImageDiffeons',
        'algorithm': 'bfgs',
        'mode': 'normal',
        'subsampleTemplate': 1,
        'zeroVar': False,
        'targetMargin': 0,
        'templateMargin':0,
        'DecimationTarget':5,
        'maxIter': 10000,
        'affine': 'none',
        'rotWeight': 1.,
        'transWeight': 1.,
        'scaleWeight': 10.,
        'affineWeight:': 100.
    }
    f=ImageMatchingDiffeons(Template=im1, Target=im2, options=options)
    f.optimizeMatching()
    return f
if __name__=="__main__":
    loggingUtils.setup_default_logging('', stdOutput=True)
    compute()
