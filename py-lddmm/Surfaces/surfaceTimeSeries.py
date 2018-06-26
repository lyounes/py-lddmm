import logging
import numpy.linalg as la
from Surfaces import surfaces
from PointSets.pointSets import *
from Common import conjugateGradient as cg, pointEvolution as evol
from Common.affineBasis import *

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'

class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []


## Main class for surface matching
#        Template: surface class (from surface.py); if not specified, opens fileTemp
#        Target: surface class (from surface.py); if not specified, opens fileTarg
#        param: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        affineWeight: multiplicative constant on affine regularization
#        rotWeight: multiplicative constant on affine regularization (supercedes affineWeight)
#        scaleWeight: multiplicative constant on scale regularization (supercedes affineWeight)
#        transWeight: multiplicative constant on translation regularization (supercedes affineWeight)
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'similitude', 'euclidean', 'translation' or 'none'
#        maxIter: max iterations in conjugate gradient
class SurfaceMatching(object):
    def __init__(self, Template=None, Targets=None, fileTempl=None, fileTarg=None, param=None, times = None,
                 maxIter=1000, regWeight = 1.0, affineWeight = 1.0, verb=True, affine = 'none',
                  rotWeight = None, scaleWeight = None, transWeight = None,
                  rescaleTemplate=False, subsampleTargetSize=-1, testGradient=True,  saveFile = 'evolution', outputDir = '.'):
        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        if Template==None:
            if fileTempl==None:
                logging.error('Please provide a template surface')
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileTempl)
        else:
            self.fv0 = surfaces.Surface(surf=Template)
        if Targets==None:
            if fileTarg==None:
                logging.error('Please provide a list of target surfaces')
                return
            else:
                self.fv1 = []
                if self.param.errorType == 'L2Norm':
                    for f in fileTarg:
                        fv1 = surfaces.Surface()
                        fv1.readFromImage(f)
                        self.fv1.append(fv1)
                else:
                    for f in fileTarg:
                        self.fv1.append(surfaces.Surface(filename=f))
        else:
            self.fv1 = []
            if self.param.errorType == 'L2Norm':
                for s in Targets:
                    fv1 = surfaces.Surface()
                    fv1.readFromImage(s)
                    self.fv1.append(fv1)
            else:
                for s in Targets:
                    self.fv1.append(surfaces.Surface(surf=s))


        if rescaleTemplate:
            f0 = np.fabs(self.fv0.surfVolume())
            f1 = np.fabs(self.fv1[0].surfVolume()) 
            self.fv0.updateVertices(self.fv0.vertices * (f1/f0)**(1./3))
            m0 = np.mean(self.fv0.vertices, axis = 0)
            m1 = np.mean(self.fv1[0].vertices, axis = 0)
            self.fv0.updateVertices(self.fv0.vertices + (m1-m0))

        self.volumeWeight = 10.0 
        self.nTarg = len(self.fv1)
        self.saveRate = 10
        self.iter = 0
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)
        self.dim = self.fv0.vertices.shape[1]
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        
        self.affine = affine
        if self.affine=='euclidean' or self.affine=='translation':
            self.saveCorrected = True
        else:
            self.saveCorrected = False

        self.affB = AffineBasis(self.dim, affine)
        self.affineDim = self.affB.affineDim
        self.affineBasis = self.affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(self.affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[self.affB.rotComp] = rotWeight
        if (len(self.affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[self.affB.simComp] = scaleWeight
        if (len(self.affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[self.affB.transComp] = transWeight

        self.coeffAff_ = np.ones(self.affineDim)
        if (len(self.affB.rotComp) > 0):
            self.coeffAff_[self.affB.rotComp] = 100.

        self.fv0Fine = surfaces.Surface(surf=self.fv0)
        if (subsampleTargetSize > 0):
            self.fv0.Simplify(subsampleTargetSize)
            print 'simplified template', self.fv0.vertices.shape[0]
        v0 = self.fv0.surfVolume()
        #print 'v0', v0
        if self.param.errorType == 'L2Norm' and v0 < 0:
            #print 'flip'
            self.fv0.flipFaces()
            v0 = -v0 ;
        for s in self.fv1:
            v1 = s.surfVolume()
            #print 'v1', v1
            if (v0*v1 < 0):
                #print 'flip1'
                s.flipFaces()
        self.x0 = self.fv0.vertices
        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.npt = self.x0.shape[0]
        if times is None:
            times = 1+np.array(range(self.nTarg))
        self.Tsize = int(round(times[-1]/self.param.timeStep))        
        self.jumpIndex = np.int_(np.round(times/self.param.timeStep))
        self.isjump = np.zeros(self.Tsize+1, dtype=bool)
        for k in self.jumpIndex:
            self.isjump[k] = True
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.a0 = np.zeros([self.x0.shape[0], self.x0.shape[1]])
        self.at = np.tile(self.a0, [self.Tsize, 1, 1])

        self.regweight = np.ones(self.Tsize)
        self.regweight[range(self.jumpIndex[0])] = regWeight

        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.atTry = np.zeros([self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])

        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        for k,s in enumerate(self.fv1):
            s.saveVTK(self.outputDir+'/Target'+str(k)+'.vtk')
        self.affBurnIn = 20
        self.coeffAff1 = 1.
        self.coeffAff2 = 100.
        self.coeffAff = self.coeffAff1 * self.coeffAff_
        if self.affineDim > 0:
            self.affineBurnIn = True
        else:
            self.affineBurnIn = False
        z= self.fv0.surfVolume()
        if (z < 0):
            self.fv0ori = -1
        else:
            self.fv0ori = 1

        # z= self.fv1.surfVolume()
        # if (z < 0):
        #     self.fv1ori = -1
        # else:
        #     self.fv1ori = 1


    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef):
        obj = 0
        for k,s in enumerate(_fvDef):
            if self.param.errorType == 'L2Norm':
                obj += surfaces.L2Norm(s, self.fv1[k].vfld) / (self.param.sigmaError ** 2)
            else:
                obj += self.param.fun_obj(s, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
            #print 'surf', s.surfVolume(), self.fv1[k].surfVolume(), self.volumeWeight
            obj += self.volumeWeight*(s.surfVolume()-self.fv1[k].surfVolume())**2
        return obj

    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian=False):
        x0 = self.x0
            
        param = self.param
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        #print a0.shape
        if withJacobian:
            (xt, Jt)  = evol.landmarkDirectEvolutionEuler(x0, at, param.KparDiff, withJacobian=True,affine=A)
        else:
            xt  = evol.landmarkDirectEvolutionEuler(x0, at, param.KparDiff, affine=A)
        #print xt[-1, :, :]
        #print obj
        obj = 0
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :, :])
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            ra = param.KparDiff.applyK(z, a)
            self.v[t, :] = ra
            obj = obj + self.regweight[t]*timeStep*np.multiply(a, (ra)).sum()
            if self.affineDim > 0:
                obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            return obj, xt, Jt
        elif withTrajectory:
            return obj, xt
        else:
            return obj


    def objectiveFun(self):
        if self.obj is None:
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.obj0 = 0
            for k in range(self.nTarg):
                if self.param.errorType == 'L2Norm':
                    self.obj0 += surfaces.L2Norm0(self.fv1[k]) / (self.param.sigmaError ** 2)
                else:   
                    self.obj0 += self.param.fun_obj0(self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                #foo = surfaces.Surface(surf=self.fvDef[k])
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))
                #foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return (self.at, self.Afft)
    
    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft
        foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTry += foo[0]

        ff = [] 
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(foo[1][self.jumpIndex[k], :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500

        if (objRef is None) | (objTry < objRef):
            self.atTry = atTry
            self.AfftTry = AfftTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry



    def endPointGradient(self):
        px = []
        for k in range(self.nTarg):
            if self.param.errorType == 'L2Norm':
                targGradient = -surfaces.L2NormGradient(self.fvDef[k], self.fv1[k].vfld) / (self.param.sigmaError ** 2)
            else:
                targGradient = -self.param.fun_objGrad(self.fvDef[k], self.fv1[k], self.param.KparDist)/(self.param.sigmaError**2)
            targGradient -= (2./3) * self.volumeWeight*(self.fvDef[k].surfVolume() - self.fv1[k].surfVolume()) * self.fvDef[k].computeAreaWeightedVertexNormals()
            px.append(targGradient)
        #print "px", (px[0]**2).sum()
        return px 


    def getGradient(self, coeff=1.0):
        px1 = self.endPointGradient()
            
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]

        foo = evol.timeSeriesGradient(self.x0, self.at, px1,
                                        self.param.KparDiff,
                                        self.regweight, affine=A, isjump = self.isjump)
        # times = (1+np.array(range(self.nTarg)))*self.Tsize1)
        grd = Direction()
        grd.diff = foo[0] / (coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineBurnIn:
            grd.diff *= 0 
        if self.affineDim > 0 and self.iter < self.affBurnIn:
            dA = foo[1]
            db = foo[2]
            grd.aff = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft)
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
               grd.aff[t] -=  dAff.reshape(grd.aff[t].shape)
            grd.aff /= (self.coeffAff*coeff*self.Tsize)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.aff = dir1.aff + beta * dir2.aff
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)
        dir.aff = np.copy(dir0.aff)

        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
        if self.affineBurnIn:
            dirfoo.diff *= 0 
        if self.iter < self.affBurnIn:
            dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        else:
            dirfoo.aff = np.zeros([self.Tsize, self.affineDim])
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        #gg = g1.at
        uu = g1.aff
        for t in range(self.Tsize):
            gg = self.param.KparDiff.applyK(self.xt[t,...], g1.diff[t,...])
            ll = 0
            for ll,gr in enumerate(g2):
                ggOld = gr.diff[t,...]
                res[ll]  += (ggOld*gg).sum()
        
        if not uu is None:
            for ll,gr in enumerate(g2):
                res[ll] += (uu * gr.aff *self.coeffAff).sum()

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        #print self.at

    def endOfIteration(self):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2 * self.coeffAff_
            self.affineBurnIn = False
        if (self.iter % self.saveRate == 0):
            logging.info('Saving surfaces...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
                    
            (xt, ft, Jt)  = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                           withPointSet = self.fv0Fine.vertices, withJacobian=True)

            if self.saveCorrected:
                f = surfaces.Surface(surf=self.fv0Fine)
                X = self.affB.integrateFlow(self.Afft)
                displ = np.zeros(self.x0.shape[0])
                dt = 1.0 /self.Tsize ;
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yyt = np.dot(xt[t,...] - X[1][t, ...], U.T)
                    zt = np.dot(ft[t,...] - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        at = np.dot(self.at[t,...], U.T)
                        vt = self.param.KparDiff.applyK(yyt, at, firstVar=zt)
                    f.updateVertices(zt)
                    vf = surfaces.vtkFields() ;
                    vf.scalars.append('Jacobian') ;
                    vf.scalars.append(np.exp(Jt[t, :])-1)
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity') ;
                    vf.vectors.append(vt)
                    nu = self.fv0ori*f.computeVertexNormals()
                    if t >= self.jumpIndex[0]:
                        displ += dt * (vt*nu).sum(axis=1)
                    f.saveVTK2(self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.vtk', vf)

                for k,fv in enumerate(self.fv1):
                    f = surfaces.Surface(surf=fv)
                    U = la.inv(X[0][self.jumpIndex[k]])
                    yyt = np.dot(f.vertices - X[1][self.jumpIndex[k], ...], U.T)
                    f.updateVertices(yyt)
                    f.saveVTK(self.outputDir +'/Target'+str(k)+'Corrected.vtk')
            
            fvDef = surfaces.Surface(surf=self.fv0Fine)
            AV0 = fvDef.computeVertexArea()
            nu = self.fv0ori*self.fv0Fine.computeVertexNormals()
            #v = self.v[0,...]
            displ = np.zeros(self.npt)
            dt = 1.0 /self.Tsize
            v = self.param.KparDiff.applyK(ft[0,...], self.at[0,...], firstVar=self.xt[0,...])
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(ft[kk, :, :]))
                AV = fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0])-1
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jt[kk, :])-1)
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(np.exp(Jt[kk, :])/(AV+1)-1)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if kk < self.Tsize:
                    nu = self.fv0ori*fvDef.computeVertexNormals()
                    v = self.param.KparDiff.applyK(ft[kk,...], self.at[kk,...], firstVar=self.xt[kk,...])
                    #v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                if kk >= self.jumpIndex[0]:
                    displ += dt * (v*nu).sum(axis=1)
                vf.vectors.append('velocity')
                vf.vectors.append(self.v[kkm,:])
                fvDef.saveVTK2(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', vf)
        else:
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))


    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        logging.info('Gradient lower bound: %f'  %(self.gradEps))
        #print 'x0:', self.x0
        #print 'y0:', self.y0
        self.cgBurnIn = self.affBurnIn
        
        cg.cg(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
        #return self.at, self.xt

