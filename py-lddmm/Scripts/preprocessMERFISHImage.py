import ntpath
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
import multiprocessing as mp
from multiprocessing import Pool
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import cv2
from skimage.segmentation import watershed
from base.meshes import buildImageFromFullListHR, buildMeshFromCentersCounts


testRun = False
homedir = '/Users/younes/Development/Data/Merfish'
if not os.path.exists(homedir + '/1_meshdata'):
    os.mkdir(homedir + '/1_meshdata')
mouses = glob(homedir + '/0_origdata/mouse2')

def f(file1, radius=30.):
    # file1 = arg[0]
    # outdir = arg[1]
    # print(outdir)
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    print('reading ' + file1)
    df = pd.read_csv(file1)
    print('done')
    x_ = df['global_x'].to_numpy()
    y_ = df['global_y'].to_numpy()

    ugenes, inv = np.unique(df['gene'], return_inverse=True)
    img1 = buildImageFromFullListHR(x_, y_, inv, radius=radius)
    return img1

if __name__ == '__main__':
    file1 = homedir + '/202202170851_60988201_VMSC01001/detected_transcripts.csv'
    print('building image')
    img, info = f(file1, 15.)
    img2 = img.sum(axis=2)
    minx = info[0]
    miny = info[1]
    spacing = info[2]
    cm = img2.max()
    imgout = ((255 / cm) * img2).astype(np.uint8)
    print('running watershed')
    seg = (1+watershed(-img2)) * (img2>0)
    nlab = seg.max() + 1
    print(f'number of labels: {nlab}')
    print(seg.shape)
    segout = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for j in range(3):
        col = np.random.choice(256, nlab)
        print(col.shape, col[seg].shape)
        segout[:, :, j] = col[seg]
    iio.imwrite(homedir + "/img.png", imgout)
    print(segout.shape)
    cv2.imwrite(homedir + "/seg.png", segout.astype(np.int8))



    centers = np.zeros((nlab, 2))
    cts = np.zeros((nlab, img.shape[2]), dtype=int)
    nb = np.zeros(nlab, dtype=int)
    x0 = np.linspace(minx+spacing/2, minx + img.shape[0]*spacing - spacing/2, img.shape[0])
    y0 = np.linspace(miny+spacing/2, miny + img.shape[1]*spacing - spacing/2, img.shape[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            centers[seg[i,j], 0] += x0[i]
            centers[seg[i,j], 1] += y0[j]
            nb[seg[i,j]] += 1
            cts[seg[i,j],:] += img[i,j,:].astype(int)
    centers = centers[nb>0, :]
    cts = cts[nb > 0, :]
    centers /= nb[nb>0, None]
    print(f'number of cells: {centers.shape[0]}')
    df = pd.DataFrame(data = {'centers_x':centers[:,0], 'centers_y':centers[:,1]})
    df.to_csv(homedir + '/centers.csv')
    df = pd.DataFrame(data=cts)
    df.to_csv(homedir + '/counts.csv')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=100, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/mesh.vtk')

    #plt.imshow(imgout)
    #plt.show()
