import csv
from base import diffeo
from base import surfaces
from os import path






# Pipeline for longitudinal shape analysis
# Input: .csv file with location of original files assumed to be either image segmentations or surfaces.
# Input: visit orders should also be listed in file, as well as left/right info

# Step 1: (if needed): isosurface the image files and flip right files; save resulting surfaces in separate directory
# Step 2: Rigid alignment to the first baseline; save resulting surfaces in separate directory
# Step 3: Template estimation from baselines; save template in separate directory
# Step 4: Registration of all surfaces to template; save relevant information in vtk files in separate directory.


class Pipeline:
    def __init__(self, fileIn, dir1, dir2, dir3, dir4, locField = 'location'):
        self.data = []
        with open(fileIn) as csvfile:
            data = csv.DictReader(csvfile, delimiter=',')
            for row in data:
                self.data.append(row)
        self.dir1 = dir1
        self.dir2 = dir2
        self.dir3 = dir3
        self.dir4 = dir4
        self.locField = locField

    def Step1_Isosurface(self, zeroPad=False, axun=False, withBug = False, smooth=False, targetSize=1000):
        sf = surfaces.Surface()
        for record in self.data:
            v = diffeo.gridScalars(fileName=record[self.locField], force_axun=axun, withBug = withBug)
            if zeroPad:
                v.zeroPad(1)
            t = 0.5 * (v.data.max() + v.data.min())
            # print v.resol
            if smooth:
                sf.Isosurface(v.data ,value=t ,target=targetSize ,scales=v.resol ,smooth=.75)
            else:
                sf.Isosurface(v.data ,value=t ,target=targetSize ,scales=v.resol ,smooth=-1)

            sf.edgeRecover()
            # print sf.surfVolume()
            u = path.split(record[self.locField])
            [nm ,ext] = path.splitext(u[1])
            sf.savebyu(self.dir1 + '/' + nm + '.byu')

    #def Step2_Rigid(self):
