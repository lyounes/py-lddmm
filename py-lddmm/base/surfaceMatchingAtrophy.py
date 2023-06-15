import numpy.linalg as la
import logging
from .surfaces import Surface, vtkFields
#import pointEvolution_fort as evol_omp
from . conjugateGradient import cg
from .bfgs import bfgs
import pointEvolution as evol
from . surfaceMatching import SurfaceMatching, Control, State
from .affineBasis import *


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDiffOut: background kernel: if not specified, use typeKernel with width sigmaKernelOut
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class SurfaceMatchingParamAtrophy(surfaceMatching.SurfaceMatchingParam):
#     def __init__(self, timeStep = .1, KparDiff = None, KparDist = None, KparDiffOut = None, sigmaKernel = 6.5,
#                  sigmaKernelOut=6.5, sigmaDist=2.5, sigmaError=1.0, typeKernel='gauss', errorType='varifold'):
#         surfaceMatching.SurfaceMatchingParam.__init__(self, timeStep = timeStep, KparDiff = KparDiff, KparDist=KparDist,
#                                                       sigmaKernel =  sigmaKernel, sigmaDist = sigmaDist,
#                                                       sigmaError = sigmaError,
#                                                       typeKernel = typeKernel, errorType=errorType)
#         self.sigmaKernelOut = sigmaKernelOut
#         if KparDiffOut == None:
#             self.KparDiffOut = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernelOut)
#         else:
#             self.KparDiffOut = KparDiffOut
#

## Main class for surface matching
#        Template: sequence of surface classes (from surface.py) if not specified, opens files in fileTemp
#        Target: sequence of surface classes (from surface.py) if not specified, opens files in fileTarg
#        par: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        regWeightOut: multiplicative constant on background regularization
#        affineWeight: multiplicative constant on affine regularization
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        mu: initial value for quadratic penalty normalization
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'euclidean' or 'none'
#        maxIter_cg: max iterations in conjugate gradient
#        maxIter_al: max interation for augmented lagrangian

class SurfaceMatchingAtrophy(SurfaceMatching):
    def __init__(self, Template=None, Target=None, options=None):
        super().__init__(Template=Template, Target=Target, options=options)
        if self.options['volumeOnly']:
            self.cval = np.zeros(self.Tsize + 1)
            self.cstr = np.zeros(self.Tsize + 1)
            self.lmb = np.zeros(self.Tsize + 1)
        else:
            self.cval = np.zeros([self.Tsize + 1, self.npt])
            self.cstr = np.zeros([self.Tsize + 1, self.npt])
            self.lmb = np.zeros([self.Tsize + 1, self.npt])
        self.nu = np.zeros([self.Tsize + 1, self.npt, self.dim])

        self.mu = self.options['mu']
        self.ds = 1.0

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['volumeOnly'] = False
        options['maxIter_grad'] = 1000
        options['maxIter_al'] = 100
        options['mu'] = 1.0
        options['KparDiffOut'] = None
        self.converged = False
        return options


    def constraintTerm(self, st, control):
        at = control['at']
        Afft = control['Afft']
        xt = st['xt']
        timeStep = 1.0/self.Tsize
        obj=0
        cval = np.zeros(self.cval.shape)
        for t in range(self.Tsize):
            a = at[t]
            x = xt[t]
            nu = np.zeros(x.shape)
            fk = self.fv0.faces
            xDef0 = x[fk[:, 0], :]
            xDef1 = x[fk[:, 1], :]
            xDef2 = x[fk[:, 2], :]
            nf =  np.cross(xDef1-xDef0, xDef2-xDef0)
            for kk,j in enumerate(fk[:,0]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,1]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,2]):
                nu[j, :] += nf[kk,:]
            if not self.options['volumeOnly']:
                nu /= np.sqrt((nu**2).sum(axis=1)).reshape([nu.shape[0], 1])
            nu *= self.fv0ori


            #r = self.options['KparDiff'].applyK(x, a) + np.dot(x, A.T) + b
            r = self.options['KparDiff'].applyK(x, a)
            self.nu[t,...] = nu
            self.v[t,...] = r
            if self.options['volumeOnly']:
                cstr = (nu*r).sum()
            else:
                cstr = np.squeeze((nu*r).sum(axis=1))
            self.cstr[t,...] = np.maximum(cstr, 0)
            cval[t,...] = np.maximum(cstr - self.lmb[t,...]*self.mu, 0)
            obj += 0.5*timeStep * (cval[t,...]**2).sum()/self.mu

        #print 'cstr', obj
        return obj, cval

    def constraintTermGrad(self, st, control):
        at = control['at']
        Afft = control['Afft']
        xt = st['xt']
        lmb = np.zeros(self.cval.shape)
        dxcval = np.zeros(xt.shape)
        dacval = np.zeros(at.shape)
        dAffcval = None
        for t in range(self.Tsize):
            a = at[t]
            x = xt[t]
            fk = self.fv0.faces
            nu = np.zeros(x.shape)
            xDef0 = x[fk[:, 0], :]
            xDef1 = x[fk[:, 1], :]
            xDef2 = x[fk[:, 2], :]
            nf =  np.cross(xDef1-xDef0, xDef2-xDef0)
            for kk,j in enumerate(fk[:,0]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,1]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,2]):
                nu[j, :] += nf[kk,:]
            normNu = np.sqrt((nu**2).sum(axis=1))
            if not self.options['volumeOnly']:
                nu /= normNu.reshape([nu.shape[0], 1])
            nu *= self.fv0ori

            vt = self.options['KparDiff'].applyK(x, a)
            if self.options['volumeOnly']:
                lmb[t] = -np.maximum((nu*vt).sum() -self.lmb[t]*self.mu, 0)/self.mu
                lnu = lmb[t]*nu
                lv = vt * lmb[t]
            else:
                lmb[t, :] = -np.maximum(np.multiply(nu, vt).sum(axis=1) -self.lmb[t,:]*self.mu, 0)/self.mu
                lnu = np.multiply(nu, lmb[t, :].reshape([self.npt, 1]))
                lv = vt * lmb[t,:,np.newaxis]
            #lnu = np.multiply(nu, np.mat(lmb[t, npt:npt1]).T)
            #print lnu.shape
            dxcval[t] = self.options['KparDiff'].applyDiffKT(x, a, lnu)
            dxcval[t] += self.options['KparDiff'].applyDiffKT(x, lnu, a)
            #dxcval[t] += np.dot(lnu, A)
            if not self.euclideanGradient:
                dacval[t] = np.copy(lnu)
            else:
                dacval[t] = self.options['KparDiff'].applyK(x, lnu)
            # dAffcval = []
            # if self.affineDim > 0:
            #     dAffcval[t, :] = (np.dot(self.affineBasis.T, np.vstack([np.dot(lnu.T, x).reshape([dim2,1]), lnu.sum(axis=0).reshape([self.dim,1])]))).flatten()
            if not self.options['volumeOnly']:
                lv /= normNu.reshape([nu.shape[0], 1])
                lv -= np.multiply(nu, np.multiply(nu, lv).sum(axis=1).reshape([nu.shape[0], 1]))
            lvf = lv[fk[:,0]] + lv[fk[:,1]] + lv[fk[:,2]]
            dnu = np.zeros(x.shape)
            foo = np.cross(xDef2-xDef1, lvf)
            for kk,j in enumerate(fk[:,0]):
                dnu[j, :] += foo[kk,:]
            foo = np.cross(xDef0-xDef2, lvf)
            for kk,j in enumerate(fk[:,1]):
                dnu[j, :] += foo[kk,:]
            foo = np.cross(xDef1-xDef0, lvf)
            for kk,j in enumerate(fk[:,2]):
                dnu[j, :] += foo[kk,:]
            dxcval[t] -= self.fv0ori*dnu


        #print 'testg', (lmb**2).sum() 
        return lmb, dxcval, dacval, dAffcval






    def testConstraintTerm(self, st, control):
        at = control['at']
        Afft = control['Afft']
        eps = 0.00000001
        stTry = State()
        ctrTry = Control()
        rx = np.random.randn(self.Tsize+1, self.npt, self.dim)
        ra = np.random.randn(self.Tsize+1, self.npt, self.dim)
        stTry['xt'] = st['xt'] + eps*rx
        ctrTry['at'] = control['at'] + eps*ra
        ux0 = self.constraintTerm(stTry, control)
        ua0 = self.constraintTerm(st, ctrTry)
        stTry['xt'] = st['xt'] - eps*rx
        ctrTry['at'] = control['at'] - eps*ra
        ux1 = self.constraintTerm(stTry, control)
        ua1 = self.constraintTerm(st, ctrTry)
        [l, dx, da, dA] = self.constraintTermGrad(st, control)
        vx = (dx * rx).sum()
        va = (da * ra).sum()
        logging.info('Testing constraints:')
        logging.info('var x: %f %f' %( self.Tsize*(ux1[0]-ux0[0])/(2*eps), -vx))
        logging.info('var a: %f %f' %( self.Tsize*(ua1[0]-ua0[0])/(2*eps), -va))
        # if self.affineDim > 0:
        #     uA = self.constraintTerm(xt, at, AfftTry)
        #     vA = np.multiply(dA, AfftTry-Afft).sum()/eps
        #     logging.info('var affine: %f %f' %(self.Tsize*(uA[0]-u0[0])/(eps), -vA ))

    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian = False):
        f = super().objectiveFunDef(control, var = var, withTrajectory=True, withJacobian=withJacobian)
        cstr = self.constraintTerm(f[1], control)
        obj = f[0]+cstr[0]

        #print f[0], cstr[0]

        if withJacobian or withTrajectory:
            return obj, f[1], cstr[1]
        else:
            return obj

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.options['fun_obj0'](self.fv1, self.options['KparDist']) / (self.options['sigmaError']**2)
            if self.options['symmetric']:
                self.obj0 += self.options['fun_obj0'](self.fv0, self.options['KparDist']) / (self.options['sigmaError']**2)

            (self.obj, self.state, self.cval) = self.objectiveFunDef(self.control, withTrajectory=True)
            self.obj += self.obj0

            self.fvDef.updateVertices(np.squeeze(self.state['xt'][self.Tsize, ...]))
            self.obj += self.options['fun_obj'](self.fvDef, self.fv1, self.options['KparDist']) / (self.options['sigmaError']**2)
            if self.options['symmetric']:
                self.fvInit.updateVertices(np.squeeze(self.control['x0']))
                self.obj += self.options['fun_obj'](self.fvInit, self.fv0, self.options['KparDist']) / (self.options['sigmaError']**2)

        return self.obj

    # def getVariable(self):
    #     return [self.at, self.Afft]

    def updateTry(self, dr, eps, objRef=None):
        controlTry = Control()
        controlTry['at']  = self.control['at'] - eps * dr['at']
        if self.affineDim > 0:
            controlTry['Afft'] = self.control['Afft'] - eps * dr['Afft']
        else:
            controlTry['Afft'] = None

        fv0 = Surface(surf=self.fv0)
        if self.options['symmetric']:
            controlTry['x0'] = self.control['x0'] - eps * dr['x0']
            fv0.updateVertices(controlTry['x0'])
        objTry, st, cval = self.objectiveFunDef(controlTry, var = {'fv0': fv0}, withTrajectory=True)

        ff = Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(st['xt'][self.Tsize, ...]))
        objTry += self.options['fun_obj'](ff, self.fv1, self.options['KparDist']) / (self.options['sigmaError']**2)
        if self.options['symmetric']:
            ffI = Surface(surf=self.fvInit)
            ffI.updateVertices(controlTry['x0'])
            objTry += self.options['fun_obj'](ffI, self.fv0, self.options['KparDist']) / (self.options['sigmaError']**2)
        objTry += self.obj0

        if np.isnan(objTry):
            logging.warning('Warning: nan in updateTry')
            return 1e500


        if (objRef == None) | (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry
            self.cval = cval

        return objTry


    def covectorEvolution(self, control, px1):
        at = control['at']
        Afft = control['Afft']
        M = self.Tsize
        timeStep = 1.0/M
        if self.affineDim > 0:
            dim2 = self.dim ** 2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
            else:
                A = None

        st = self.solveStateEquation(control=control, init_state=self.control['x0'])
        xt = st['xt']
        # xt = evol.landmarkDirectEvolutionEuler(self.control['x0'], at, self.options['KparDiff'], affine = A)
        #xt = xJ
        pxt = np.zeros([M+1, self.npt, self.dim])
        pxt[M, :, :] = px1
        
        foo = self.constraintTermGrad(xt, control)
        lmb = foo[0]
        dxcval = foo[1]
        dacval = foo[2]
        dAffcval = foo[3]
        
        pxt[M, :, :] += timeStep * dxcval[M]
        
        for t in range(M):
            px = pxt[M-t, :, :]
            z = xt[M-t-1, :, :]
            a = at[M-t-1, :, :]
            zpx = np.copy(dxcval[M-t-1])
            zpx += self.options['KparDiff'].applyDiffKT(z, px, a, regweight=self.options['regWeight'], lddmm=True)
            if self.affineDim > 0:
                pxt[M-t-1, :, :] = np.dot(px, self.affB.getExponential(timeStep*A[0][M-t-1])) + timeStep * zpx
                #zpx += np.dot(px, A[0][M-t-1])
            else:
                pxt[M-t-1, :, :] = px + timeStep * zpx
            

        return pxt, xt, dacval, dAffcval


    def HamiltonianGradient(self, control, px1, getCovector = False):
        at = control['at']
        Afft = control['Afft']
        (pxt, xt, dacval, dAffcval) = self.covectorEvolution(control, px1)

        if not self.euclideanGradient:
            dat = 2*self.options['regWeight']*at - pxt[1:pxt.shape[0],...] - dacval
        else:
            dat = -dacval
            for t in range(self.Tsize):
                dat[t] += self.options['KparDiff'].applyK(xt[t], 2*self.options['regWeight']*at[t] - pxt[t+1])
        if self.affineDim > 0:
            timeStep = 1.0/self.Tsize
            dAfft = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), Afft) 
            #dAfft = 2*np.multiply(self.affineWeight, Afft) - dAffcval
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t]) 
                A = AB[0:self.dim**2].reshape([self.dim, self.dim])
                #A[1][t] = AB[dim2:dim2+self.dim]
                #dA = np.dot(pxt[t+1].T, xt[t]).reshape([self.dim**2, 1])
                dA = self.affB.gradExponential(timeStep*A, pxt[t+1], xt[t]).reshape([self.dim**2, 1])
                db = pxt[t+1].sum(axis=0).reshape([self.dim,1]) 
                dAff = np.dot(self.affineBasis.T, np.vstack([dA, db]))
                dAfft[t] -=  dAff.reshape(dAfft[t].shape)
            #dAfft = np.divide(dAfft, self.affineWeight.reshape([1, self.affineDim]))
        else:
            dAfft = None
 
        if getCovector == False:
            return dat, dAfft, xt
        else:
            return dat, dAfft, xt, pxt

    # def endPointGradient(self):
    #     px1 = -self.options['fun_obj']Grad(self.fvDef, self.fv1, self.options['KparDist']) / self.options['sigmaError']**2
    #     return px1

    # def addProd(self, dir1, dir2, beta):
    #     dir = surfaceMatching.Direction()
    #     dir.diff = dir1.diff + beta * dir2.diff
    #     dir.initx = dir1.initx + beta * dir2.initx
    #     if self.affineDim > 0:
    #         dir.aff = dir1.aff + beta * dir2.aff
    #     return dir

    # def copyDir(self, dir0):
    #     dir = surfaceMatching.Direction()
    #     dir.diff = np.copy(dir0.diff)
    #     dir.aff = np.copy(dir0.aff)
    #     return dir

        
    # def dotProduct_Riemannian(self, g1, g2):
    #     res = np.zeros(len(g2))
    #     for t in range(self.Tsize):
    #         z = self.state['xt'][t, :, :]
    #         gg = g1['at'][t, :, :]
    #         u = self.options['KparDiff'].applyK(z, gg)
    #         #if self.affineDim > 0:
    #             #uu = np.multiply(g1.aff[t], self.affineWeight)
    #         ll = 0
    #         for gr in g2:
    #             ggOld = gr.['diff[t, :, :])
    #             res[ll]  +=  np.multiply(ggOld,u).sum()
    #             if self.affineDim > 0:
    #                 res[ll] += np.multiply(g1.aff[t], gr.aff[t]).sum() * self.coeffAff
    #                 #res[ll] += np.multiply(uu, gr.aff[t]).sum()
    #             ll = ll + 1
    #     if self.options['symmetric']:
    #         for ll,gr in enumerate(g2):
    #             res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx
    #
    #     return res
    #
    # def standardDotProduct(self, g1, g2):
    #     res = np.zeros(len(g2))
    #     dim2 = self.dim**2
    #     for ll,gr in enumerate(g2):
    #         res[ll]=0
    #         res[ll] += np.multiply(g1.diff, gr.diff).sum()
    #         if self.affineDim > 0:
    #             #uu = np.multiply(g1.aff, self.affineWeight.reshape([1, self.affineDim]))
    #             #res[ll] += np.multiply(uu, gr.aff).sum() * self.coeffAff
    #             res[ll] += np.multiply(g1.aff, gr.aff).sum() * self.coeffAff
    #             #+np.multiply(g1[1][k][:, dim2:dim2+self.dim], gr[1][k][:, dim2:dim2+self.dim]).sum())
    #
    #     if self.options['symmetric']:
    #         for ll,gr in enumerate(g2):
    #             res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx
    #
    #     return res



    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control= self.control
            endPoint = self.fvDef
            #A = self.affB.getTransforms(self.Afft)
        else:
            control = Control()
            if update[0]['Afft'] is not None:
                control['Afft'] = self.control['Afft'] - update[1]*update[0]['Afft']
                A = self.affB.getTransforms(control['Afft'])
            else:
                A = None
            control['at'] = self.control['at'] - update[1] *update[0]['at']
            if self.options['symmetric']:
                control['x0'] = self.control['x0'] - update[1]*update[0]['x0']
            st = self.solveStateEquation(control=control, init_state=control['x0'])
            xt = st['xt']
            # xt = evol.landmarkDirectEvolutionEuler(control['x0'], control['at']*self.ds,
            #                                        self.options['KparDiff'], affine=A)
            endPoint = Surface(surf=self.fv0)
            endPoint.updateVertices(xt[-1, :, :])

        px1 = -self.endPointGradient(endPoint=endPoint)
        #px1.append(np.zeros([self.npoints, self.dim]))
        foo = self.HamiltonianGradient(control, px1, getCovector=True)
        grd = Control()
        grd['at'] = foo[0] / (coeff*self.Tsize)
        if self.affineDim > 0:
            grd['Afft'] = foo[1] / (self.coeffAff*coeff*self.Tsize)
        if self.options['symmetric']:
            grd['x0'] = (self.initPointGradient() - foo[3][0,...])/(self.coeffInitx * coeff)
        return grd

    def randomDir(self):
        dirfoo = Control()
        dirfoo['at'] = np.random.randn(self.Tsize, self.npt, self.dim)
        if self.options['symmetric']:
            dirfoo['x0'] = np.random.randn(self.npt, self.dim)
        else:
            dirfoo['x0'] = np.zeros((self.npt, self.dim))
        if self.affineDim > 0:
            dirfoo['aff'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    # def acceptVarTry(self):
    #     self.obj = self.objTry
    #     self.at = np.copy(self.atTry)
    #     self.Afft = np.copy(self.AfftTry)

    def endOfIteration(self, forceSave=False):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0) :
            (obj1, self.state, self.cval) = self.objectiveFunDef(self.control, withJacobian=True)
            logging.info('mean constraint %f max constraint %f' %(np.sqrt((self.cstr**2).sum()/self.cval.size), np.fabs(self.cstr).max()))
            logging.info('saving data')
            self.fvInit.updateVertices(self.state['x0'])

            if self.options['affine']=='euclidean' or self.options['affine']=='translation':
                f = Surface(surf=self.fvInit)
                X = self.affB.integrateFlow(self.control['Afft'])
                displ = np.zeros(self.npt)
                dt = 1.0 /self.Tsize 
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yt = np.dot(self.state['xt'][t,...] - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        at = np.dot(self.control['at'][t,...], U.T)
                        vt = self.options['KparDiff'].applyK(yt, at)
                    f.updateVertices(yt)
                    vf = vtkFields()
                    vf.scalars.append('Jacobian') 
                    vf.scalars.append(np.exp(self.state['Jt'][t, :]))
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity') 
                    vf.vectors.append(vt)
                    f.saveVTK2(self.outputDir +'/'+self.options['saveFile']+'Corrected'+str(t)+'.vtk', vf)
                    nu = self.fv0ori*f.computeVertexNormals()
                    displ += dt * (vt*nu).sum(axis=1)
                f = Surface(surf=self.fv1)
                #logging.info('rotation?: %f' %(np.fabs(np.dot(U, U.T)- np.eye(3)).sum()))
                yt = np.dot(f.vertices - X[1][-1, ...], U.T)
                f.updateVertices(yt)
                f.saveVTK(self.outputDir +'/TargetCorrected.vtk')
                
                
            #self.testConstraintTerm(self.xt, self.at, self.Afft)
            nn = 0 
            AV0 = self.fvInit.computeVertexArea()
            nu = self.fv0ori*self.fvInit.computeVertexNormals()
            v = self.v[0,...]
            displ = np.zeros(self.npt)
            dt = 1.0 /self.Tsize 
            n1 = self.state['xt'].shape[1]
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(self.state['xt'][kk, :, :])
                AV = self.fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0])-1
                vf = vtkFields()
                vf.scalars.append('Jacobian') 
                vf.scalars.append(np.exp(self.state['Jt'][kk, :]))
                vf.scalars.append('Jacobian_T') 
                vf.scalars.append(AV[:,0])
                vf.scalars.append('Jacobian_N') 
                vf.scalars.append(np.exp(self.state['Jt'][kk, :])/(AV[:,0]+1)-1)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if kk < self.Tsize:
                    nu = self.fv0ori*self.fvDef.computeVertexNormals()
                    v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                if not self.options['volumeOnly']:
                    vf.scalars.append('constraint') 
                    vf.scalars.append(self.cstr[kkm,:])
                vf.vectors.append('velocity') 
                vf.vectors.append(self.v[kkm,:])
                vf.vectors.append('normals') 
                vf.vectors.append(self.nu[kkm,:])
                self.fvDef.saveVTK2(self.outputDir +'/'+self.options['saveFile']+str(kk)+'.vtk', vf)
                displ += dt * (v*nu).sum(axis=1)
        else:
            (obj1, self.state, self.cval) = self.objectiveFunDef(self.control, withJacobian=True)
            logging.info('mean constraint %f max constraint %f' %(np.sqrt((self.cstr**2).sum()/self.cval.size), np.fabs(self.cstr).max()))
            self.fvDef.updateVertices(np.squeeze(self.state['xt'][-1, :, :]))
            self.fvInit.updateVertices(self.control['x0'])

    def optimizeMatching(self):
        self.coeffZ = 10.
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = np.sqrt(grd2) / 100
        self.coeffAff = self.coeffAff1
        self.muEps = 1.0
        it = 0
        while (self.muEps > 0.001) & (it<self.options['maxIter_al'])  :
            logging.info('Starting Minimization: gradEps = %f muEps = %f mu = %f' %(self.gradEps, self.muEps,self.mu))
            #self.coeffZ = max(1.0, self.mu)
            if self.options['algorithm'] == 'cg':
                cg(self, verb = self.options['verb'], maxIter = self.options['maxIte_gradr'],
                      TestGradient=self.options['testGradient'], epsInit=.01,
                      Wolfe=self.options['Wolfe'])
            elif self.options['algorithm'] == 'bfgs':
                bfgs(self, verb = self.options['verb'], maxIter = self.options['maxIter_grad'],
                      TestGradient=self.options['testGradient'], epsInit=1.,
                      Wolfe=self.options['Wolfe'], lineSearch=self.options['lineSearch'], memory=50)
            self.coeffAff = self.coeffAff2
            for t in range(self.Tsize+1):
                self.lmb[t, ...] = -self.cval[t, ...]/self.mu
            logging.info('mean lambdas %f' %(np.fabs(self.lmb).sum() / self.lmb.size))
            if self.converged:
                self.gradEps *= .75
                if (((self.cstr**2).sum()/self.cval.size) > self.muEps**2):
                    self.mu *= 0.5
                else:
                    self.muEps = self.muEps /2
            else:
                self.mu *= 0.9
            self.obj = None
            it = it+1
            
            #return self.fvDef

