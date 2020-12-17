from base.surfaceExamples import Sphere, Heart
from base.surfaceSection import Hyperplane, SurfaceSection
from base.surfaceMatching import SurfaceMatchingParam
from base.surfaceToSectionsMatching import SurfaceToSectionsMatching
from base import loggingUtils
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base.kernelFunctions import Kernel
plt.ion()


loggingUtils.setup_default_logging('', stdOutput=True)
sigmaKernel = 5
sigmaDist = 5.
sigmaError = 1
internalWeight = 1.
regweight = 1.
internalCost = 'h1'

fv0 = Heart(zoom=100)
fv1 = Heart(p=1.75, scales=(1.1, 1.5), zoom = 100)
m = fv1.vertices[:,1].min()
M = fv1.vertices[:,1].max()
h = Hyperplane()
target = ()
for t in range(1,10):
    ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(0, 1, 0), offset = m + 0.1*t*(M-m)))
    target += (ss,)

m = fv1.vertices[:,2].mean()
ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(0, 0, 1), offset = m))
target += (ss,)
m = fv1.vertices[:,0].mean()
ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(1, 0, 0), offset = m))
target += (ss,)

fig = plt.figure(13)
ax = Axes3D(fig)
lim1 = fv0.addToPlot(ax, ec='k', fc='r', al=0.1)
ax.set_xlim(lim1[0][0], lim1[0][1])
ax.set_ylim(lim1[1][0], lim1[1][1])
ax.set_zlim(lim1[2][0], lim1[2][1])
colors = ('b', 'm', 'g', 'r', 'y', 'k')

for k,ss in enumerate(target):
    ss.curve.addToPlot(ax, ec=colors[k%6], lw=5)
fig.canvas.flush_events()
# plt.pause(1000)
#
# exit()

K1 = Kernel(name='laplacian', sigma=sigmaKernel)

sm = SurfaceMatchingParam(timeStep=0.1, algorithm='cg', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                          errorType='current', internalCost=internalCost)
f = SurfaceToSectionsMatching(Template=fv0, Target= target,
                    outputDir='/Users/younes/Development/Results/Sections', param=sm,
                    testGradient=False, regWeight=regweight,
                    # subsampleTargetSize = 500,
                    internalWeight=internalWeight, maxIter=1000, affine='translation', rotWeight=10., transWeight=10.,
                    scaleWeight=100., affineWeight=100.)

f.optimizeMatching()
plt.ioff()
plt.show()

