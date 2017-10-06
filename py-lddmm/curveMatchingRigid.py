import os
import glob
import matplotlib
matplotlib.use("TKAgg")
import numpy as np
import scipy.linalg as la
import matchingParam
import curves
import grid
import conjugateGradient as cg
from affineBasis import *
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from kernelFunctions import Kernel
import logging
import loggingUtils
from tqdm import *

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class CurveMatchingRigidParam(matchingParam.MatchingParam):
    def __init__(self, timeStep = .1, KparDiff = None, KparDist = None, sigmaKernel = 6.5, sigmaDist=2.5, sigmaError=1.0, 
                 errorType = 'measure', typeKernel='gauss', internalCost=None):
        matchingParam.MatchingParam.__init__(self, timeStep=timeStep, KparDiff = KparDiff, KparDist = KparDist, sigmaKernel = sigmaKernel, sigmaDist=sigmaDist,
					     sigmaError=sigmaError, errorType = errorType, typeKernel=typeKernel)
          
        self.internalCost = internalCost
                                         
        if errorType == 'current':
            print 'Running Current Matching'
            self.fun_obj0 = curves.currentNorm0
            self.fun_obj = curves.currentNormDef
            self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            print 'Running Measure Matching'
            self.fun_obj0 = curves.measureNorm0
            self.fun_obj = curves.measureNormDef
            self.fun_objGrad = curves.measureNormGradient
        elif errorType=='varifold':
            self.fun_obj0 = curves.varifoldNorm0
            self.fun_obj = curves.varifoldNormDef
            self.fun_objGrad = curves.varifoldNormGradient
        elif errorType=='varifoldComponent':
            self.fun_obj0 = curves.varifoldNormComponent0
            self.fun_obj = curves.varifoldNormComponentDef
            self.fun_objGrad = curves.varifoldNormComponentGradient
        else:
            print 'Unknown error Type: ', self.errorType

class Direction:
    def __init__(self):
        self.skew = []
        self.trans = []


## Main class for curve matching
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
class CurveMatchingRigid:
    def __init__(self, Template=None, Target=None, Clamped=None, fileTempl=None, fileTarg=None, param=None, maxIter=1000, regWeight = 1.0,
                 verb=True, gradLB = 0.001, saveRate=10, saveTrajectories=False, parpot = None,
                 testGradient=False, saveFile = 'evolution', outputDir = '.', pplot=True):
        if Template is None:
            if fileTempl is None:
                #print 'Please provide a template curve'
                return
            else:
                self.fv0 = curves.Curve(filename=fileTempl)
        else:
            self.fv0 = curves.Curve(curve=Template)
        if Target is None:
            if fileTarg is None:
                print 'Please provide a target curve'
                return
            else:
                self.fv1 = curves.Curve(filename=fileTarg)
        else:
            self.fv1 = curves.Curve(curve=Target)



        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]

        if not(Clamped is None):
            self.fvc = curves.Curve(curve=Clamped)
            self.xc = self.fvc.vertices
        else:
            self.fvc = curves.Curve()
            self.xc = np.zeros([0, self.dim])

        self.nptc = self.fv0.vertices.shape[0] + self.xc.shape[0]
        if not(self.dim == 2):
            print 'This program runs in 2D only'
            return
            
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)
        for f in glob.glob(outputDir+'/*.vtk'):
            os.remove(f)
        self.fvDef = curves.Curve(curve=self.fv0)
        self.iter = 0
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.minEig = 1e-8
        self.parpot = parpot
        self.rpot = 3.

        if param==None:
            self.param = CurveMatchingRigidParam()
        else:
            self.param = param
        

        self.x0 = self.fv0.vertices
        self.ncomponent = self.fv0.component.max() + 1
        self.component = np.zeros(self.x0.shape[0], dtype=int)
        for k in range(self.fv0.faces.shape[0]):
            self.component[self.fv0.faces[k,0]] = self.fv0.component[k]
            self.component[self.fv0.faces[k,1]] = self.fv0.component[k]

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.ncomponent])
        self.taut = np.zeros([self.Tsize, self.ncomponent, 2])
        self.atTry = np.zeros([self.Tsize, self.ncomponent])
        self.tautTry = np.zeros([self.Tsize, self.ncomponent, 2])
        self.pxt = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        self.gradLB = gradLB
        self.saveRate = saveRate 
        self.saveTrajectories = saveTrajectories
        self.pplot = pplot
        if self.pplot:
            fig=plt.figure(1)
            fig.clf()
            ax = fig.gca()
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf,:],0], self.fv1.vertices[self.fv1.faces[kf,:],1], color=[0,0,1])
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
            plt.axis('equal')
            plt.pause(0.1)



    def dataTerm(self, _fvDef):
        obj = self.param.fun_obj(_fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, at, taut, withTrajectory = False, x0 = None):
        if x0 == None:
            x0 = self.fv0.vertices
        param = self.param
        timeStep = 1.0/self.Tsize
        xt  = self.directEvolutionEuler(x0, at, taut)
        obj=0
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            zc = np.concatenate([z,self.xc])
            a = np.squeeze(at[t, :])
            tau = np.squeeze(taut[t, :, :])
            Jz = np.zeros(z.shape)
            Jz[:,0] = z[:,1]
            Jz[:,1] = -z[:,0]
            v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            v = np.concatenate([v,np.zeros([self.xc.shape[0], self.dim])])
            K = param.KparDiff.getK(zc)
            #mu = self.solveK(z,v)
            #muv = (mu*v).sum()
            eK = la.eigh(K)
            J = np.nonzero(eK[0]>self.minEig)[0]
            w = np.dot(v.T,eK[1][:,J]).T
            newmuv = ((w*w)/eK[0][J, np.newaxis]).sum()
            obj += self.regweight * timeStep * newmuv
        obj /= 2
        if withTrajectory:
            return obj, xt
        else:
            return obj

    def objectiveFunPotential(self, xt):
        if self.parpot is None:
            return 0
        obj = 0
        timeStep = 1.0/self.Tsize
        for k in range(self.ncomponent):
            Ik = np.nonzero(self.component == k)[0]
            xk = xt[:,Ik,:]
            for l in range(k + 1, self.ncomponent):
                Il = np.nonzero(self.component == l)[0]
                xl = xt[:,Il,:]
                for t in range(xt.shape[0]):
                    delta = xk[t, :, np.newaxis, :] - xl[t, np.newaxis, :, :]
                    d = np.sqrt(((delta) ** 2).sum(axis=2))
                    obj += (d ** (-self.rpot)).sum()
            if len(self.xc>0):
                for t in range(xt.shape[0]):
                    d = np.sqrt(((xk[t, :, np.newaxis, :] - self.xc[np.newaxis, :, :]) ** 2).sum(axis=2))
                    obj += (d ** (-self.rpot)).sum()
        return self.parpot * obj * timeStep

    def gradPotential(self, z):
        grad = np.zeros(z.shape)
        if self.parpot is None:
            return grad
        for k in range(self.ncomponent):
            Ik = np.nonzero(self.component == k)[0]
            for l in range(self.ncomponent):
                if l != k:
                    Il = np.nonzero(self.component == l)[0]
                    delta = z[Ik, np.newaxis, :] - z[np.newaxis, Il, :]
                    d = np.sqrt(((delta) ** 2).sum(axis=2))[:,:,np.newaxis]
                    grad[Ik,:] += (delta/(d ** (self.rpot+2))).sum(axis=1)
            if len(self.xc)>0:
                delta = z[Ik, np.newaxis, :] - self.xc[np.newaxis, :, :]
                d = np.sqrt(((delta) ** 2).sum(axis=2))[:,:,np.newaxis]
                grad[Ik,:] += (delta/(d ** (self.rpot+2))).sum(axis=1)
        return -self.rpot*self.parpot * grad

    def testGradPotential(self,z):
        dz = np.random.normal(0,1,z.shape)
        eps = 1e-8
        z2 = z + eps * dz
        obj0 = self.objectiveFunPotential(z[np.newaxis,...])
        obj = self.objectiveFunPotential(z2[np.newaxis,...])
        grad = self.gradPotential(z)
        print 'testGradPotential:', self.Tsize*(obj-obj0)/eps, (grad*dz).sum()

    def  _objectiveFun(self, at, taut, withTrajectory = False):
        (obj, xt) = self.objectiveFunDef(at, taut, withTrajectory=True)
        self.fvDef.updateVertices(np.squeeze(xt[-1, :, :]))
        obj0 = self.dataTerm(self.fvDef)
        obj2 = self.objectiveFunPotential(xt)

        if withTrajectory:
            return obj+obj0+obj2, xt
        else:
            return obj+obj0+obj2

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.taut, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef) + self.objectiveFunPotential(self.xt)
            print self.obj0, self.obj

        return self.obj

    def getVariable(self):
        return [self.at, self.taut]

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.skew
        tautTry = self.taut - eps * dir.trans
        foo = self.objectiveFunDef(atTry, tautTry, withTrajectory=True)
        objTry += foo[0]
        objTry += self.objectiveFunPotential(foo[1])

        ff = curves.Curve(curve=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.tautTry = tautTry

        return objTry

    def directEvolutionEuler(self, x0, at, taut):
        xt = np.zeros([self.Tsize+1, x0.shape[0], x0.shape[1]])
        xt[0,:,:] = np.copy(x0)
        timeStep = 1.0/self.Tsize
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :])
            tau = np.squeeze(taut[t, :, :])
            Jz = np.zeros(z.shape)
            ca = np.cos(timeStep*a)
            sa = np.sin(timeStep*a)
            Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
            Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            #v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            xt[t+1, :, :] = Jz + timeStep * tau[self.component,:]
        return xt

    def solveK(self,z, v):
        zc = np.concatenate([z,self.xc])
        vc = np.concatenate([v,np.zeros(self.xc.shape)])
        K = self.param.KparDiff.getK(zc)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad value in SolveK')
        J = np.nonzero(eK[0] > self.minEig)[0]
        w = np.dot(vc.T, eK[1][:, J]).T / eK[0][J, np.newaxis]
        mu = np.dot(eK[1][:, J], w)
        mu = mu[0:self.npt,:]
        return mu

    def solveMKM(self, z, p):
        zc = np.concatenate([z,self.xc])
        #vc = np.concatenate([v,np.zeros(self.xc.shape)])
        K = self.param.KparDiff.getK(zc)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad Value in solveMKM')
        J = np.nonzero(eK[0] > self.minEig)[0]
        Ki = np.dot(eK[1][:, J], eK[1][:, J].T/eK[0][J,np.newaxis])
        #check = np.dot(K,Ki)
        M = np.zeros([self.nptc*self.dim, self.ncomponent*(1+self.dim)])
        k1 = 0
        for k in range(self.ncomponent):
            I = np.nonzero(self.component==k)[0]
            zk = z[I,:]
            for i in range(zk.shape[0]):
                    u = zk[i,:]
                    Ju = np.array([u[1],-u[0]])
                    mm = np.concatenate([Ju[:,np.newaxis], np.eye(self.dim)], axis=1)
                    M[k1+self.dim*i:k1+self.dim*(i+1), k*(1+self.dim):(k+1)*(1+self.dim)] = mm
            k1 += self.dim*zk.shape[0]

        MKM = np.dot(M.T, np.dot(np.kron(Ki,np.eye(self.dim)),M))
        M = M[0:self.npt*self.dim,:]
        try:
            theta = la.solve(MKM, np.dot(M.T,np.ravel(p)))
        except Exception:
            raise Exception('Bad Value in solveMKM')
        return theta

    def solveMKM2(self, z, rho):
        zc = np.concatenate([z,self.xc])
        K = self.param.KparDiff.getK(zc)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad Value in solveMKM2')
        J = np.nonzero(eK[0] > self.minEig)[0]
        Ki = np.dot(eK[1][:, J], eK[1][:, J].T/eK[0][J, np.newaxis])
        M = np.zeros([self.nptc*self.dim, self.ncomponent*(1+self.dim)])
        k1 = 0
        for k in range(self.ncomponent):
            I = np.nonzero(self.component==k)[0]
            zk = z[I,:]
            for i in range(zk.shape[0]):
                    u = zk[i,:]
                    Ju = np.array([u[1],-u[0]])
                    mm = np.concatenate([Ju[:,np.newaxis], np.eye(self.dim)], axis=1)
                    M[k1+self.dim*i:k1+self.dim*(i+1), k*(1+self.dim):(k+1)*(1+self.dim)] = mm
            k1 += self.dim*zk.shape[0]

        MKM = np.dot(M.T, np.dot(np.kron(Ki,np.eye(self.dim)),M))
        try:
            theta = la.solve(MKM, rho)
        except Exception:
            raise Exception('Bad Value in solveMKM2')
        return theta




    def geodesicEquation2(self, Tsize, a0, tau0, pplot = False):
        fv0 = self.fv0
        x0 = fv0.vertices
        xt = np.zeros([Tsize+1, x0.shape[0], x0.shape[1]])
        xt[0,:,:] = np.copy(x0)
        ncomponent = fv0.component.max() + 1
        dim = x0.shape[1]
        component = np.zeros(x0.shape[0], dtype=int)
        for k in range(fv0.faces.shape[0]):
            component[fv0.faces[k,0]] = fv0.component[k]
            component[fv0.faces[k,1]] = fv0.component[k]
        at = np.zeros([Tsize+1, ncomponent])
        at[0,:] = np.copy(a0)
        taut = np.zeros([Tsize+1, ncomponent, dim])
        taut[0,:,:] = np.copy(tau0)
        rho = np.zeros(ncomponent*(dim + 1))
        Jz = np.zeros(x0.shape)
        Jz[:, 0] = x0[:, 1]
        Jz[:, 1] = -x0[:, 0]
        v = a0[component, np.newaxis] * Jz + tau0[component, :]
        mu0 = self.solveK(x0,v)
        for k in range(ncomponent):
            J = np.nonzero(component==k)[0]
            u = np.zeros([len(J),dim])
            u[:,0] = x0[J,1]
            u[:,1] = -x0[J,0]
            rho[k * (dim + 1)] = (mu0[J, :] * u).sum()
            rho[k * (dim + 1)+1:(k+1)*(dim+1)] = mu0[J, :].sum(axis=0)

        fvDef = curves.Curve(curve=fv0)
        if self.pplot:
            fig = plt.figure(2)
            fig.clf()
            ax = fig.gca()
            for kf in range(fvDef.faces.shape[0]):
                ax.plot(fvDef.vertices[fvDef.faces[kf, :], 0], fvDef.vertices[fvDef.faces[kf, :], 1], color=[1, 0, 0],
                        marker='*')
            plt.axis('equal')
            plt.pause(0.1)

        timeStep = 1.0/Tsize
        for t in tqdm(range(Tsize)):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :])
            tau = np.squeeze(taut[t, :, :])
            ca = np.cos(timeStep*a)
            sa = np.sin(timeStep*a)
            Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
            Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            #v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            xt[t+1, :, :] = Jz + timeStep * tau[self.component,:]
            Jz[:,0] = z[:,1]
            Jz[:,1] = -z[:,0]
            v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            #xt[t+1, :, :] = xt[t, :, :] + timeStep * v
            try:
                mu = self.solveK(z,v)
            except Exception:
                print 'solved until t=',t*timeStep
                return xt[0:t,...], at[0:t,...], taut[0:t,...]

            a1 = self.regweight*mu[np.newaxis,...]
            a2 = mu[np.newaxis,...]
            zpx = self.param.KparDiff.applyDiffKT(z, a1, a2)
            Jz[:, 0] = mu[:, 1]
            Jz[:, 1] = -mu[:, 0]
            dv = a[self.component, np.newaxis] * Jz
            zpot = self.gradPotential(z)
            zpx += dv - zpot
            drho = np.zeros(ncomponent * (dim + 1))
            for k in range(ncomponent):
                J = np.nonzero(component == k)[0]
                u = np.zeros([len(J), dim])
                u[:,0] = z[J, 1]
                u[:,1] = -z[J, 0]
                drho[k * (dim + 1)] = (zpx[J, :] * u).sum()
                drho[k * (dim + 1) + 1:(k + 1) * (dim + 1)] = zpx[J, :].sum(axis=0)
            for k in range(ncomponent):
                pt = rho[k * (dim + 1) + 1:(k + 1) * (dim + 1)]
                drho[k * (dim + 1)] += -pt[0]*tau[k,1] + pt[1]*tau[k,0]
                drho[k * (dim + 1)+1] += -a[k] * pt[1]
                drho[k * (dim + 1)+2] += a[k] * pt[0]
            rho -= timeStep * drho
            theta = self.solveMKM2(z,rho)
            at[t+1,:] = theta[range(0,len(theta),self.dim+1)]
            taut[t+1, :, 0] = theta[range(1, len(theta), self.dim + 1)]
            taut[t+1, :, 1] = theta[range(2, len(theta), self.dim + 1)]
            fvDef.updateVertices(xt[t+1,:,:])
            if pplot:
                fig=plt.figure(2)
                fig.clf()
                ax = fig.gca()
                if len(self.xc) > 0:
                    for kf in range(self.fvc.faces.shape[0]):
                        ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
                                self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[0, 0, 0])
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                            color=[0, 0, 1])
                for kf in range(fvDef.faces.shape[0]):
                    ax.plot(fvDef.vertices[fvDef.faces[kf,:],0], fvDef.vertices[fvDef.faces[kf,:],1], color=[1,0,0])
                for k in range(fvDef.component.max() + 1):
                    I = np.nonzero(fvDef.component == k)[0]
                    xDef = fvDef.vertices[I, :]
                    ax.plot(np.array([np.mean(xDef[:, 0]), xDef[0, 0]]),
                            np.array([np.mean(xDef[:, 1]), xDef[0, 1]]),
                            color=[0, 0, 1])
                    ax.plot(np.mean(xt[0:t, I, 0], axis=1), np.mean(xt[0:t, I, 1], axis=1))
                plt.title('t={0:.4f}'.format(t*timeStep))
                plt.axis('equal')
                plt.pause(0.001)
        if self.pplot:
            fig=plt.figure(2)
            fig.clf()
            ax = fig.gca()
            for kf in range(fvDef.faces.shape[0]):
                ax.plot(fvDef.vertices[fvDef.faces[kf,:],0], fvDef.vertices[fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
            plt.axis('equal')
            plt.pause(0.1)
        return xt, at, taut

    def geodesicEquation(self, Tsize, a0, tau0, pplot = False, symplectic=False):
        fv0 = self.fv0
        x0 = fv0.vertices
        xt = np.zeros([Tsize+1, x0.shape[0], x0.shape[1]])
        xt[0,:,:] = np.copy(x0)
        ncomponent = fv0.component.max() + 1
        component = np.zeros(x0.shape[0], dtype=int)
        for k in range(fv0.faces.shape[0]):
            component[fv0.faces[k,0]] = fv0.component[k]
            component[fv0.faces[k,1]] = fv0.component[k]
        at = np.zeros([Tsize+1, ncomponent])
        at[0,:] = np.copy(a0)
        taut = np.zeros([Tsize+1, ncomponent, x0.shape[1]])
        taut[0,:,:] = np.copy(tau0)
        pxt = np.zeros([Tsize+1, x0.shape[0], x0.shape[1]])
        Jz = np.zeros(x0.shape)
        Jz[:, 0] = x0[:, 1]
        Jz[:, 1] = -x0[:, 0]
        v = a0[self.component, np.newaxis] * Jz + tau0[self.component, :]
        mu0 = self.solveK(x0,v)
        pxt[0,:,:] = mu0
        fvDef = curves.Curve(curve=fv0)
        if self.pplot:
            fig = plt.figure(2)
            fig.clf()
            ax = fig.gca()
            for kf in range(fvDef.faces.shape[0]):
                ax.plot(fvDef.vertices[fvDef.faces[kf, :], 0], fvDef.vertices[fvDef.faces[kf, :], 1], color=[1, 0, 0],
                        marker='*')
            plt.axis('equal')
            plt.pause(0.1)

        timeStep = 1.0/Tsize
        for t in range(Tsize):
            z = np.squeeze(xt[t, :, :])
            px = np.squeeze(pxt[t, :, :])
            a = np.squeeze(at[t, :])
            tau = np.squeeze(taut[t, :, :])
            ca = np.cos(timeStep*a)
            sa = np.sin(timeStep*a)
            if symplectic:
                z2 = z + timeStep * tau[self.component,:]
                Jz[:,0] = ca[self.component]*z2[:,0] + sa[self.component]*z2[:,1]
                Jz[:,1] = -sa[self.component]*z2[:,0] + ca[self.component]*z2[:,1]
                znew = np.copy(Jz)
                z = np.copy(znew)
            else:
                Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
                Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
                #v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
                znew = Jz + timeStep * tau[self.component,:]
            xt[t+1, :, :] = znew
            #xt[t+1, :, :] = xt[t, :, :] + timeStep * v
            Jz[:,0] = z[:,1]
            Jz[:,1] = -z[:,0]
            v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            try:
                mu = self.solveK(z,v)
            except Exception:
                print 'solved until t=',t*timeStep
                return xt[0:t,...], pxt[0:t,...], at[0:t,...], taut[0:t,...]

            a1 = self.regweight*mu[np.newaxis,...]
            a2 = mu[np.newaxis,...]
            zpx = self.param.KparDiff.applyDiffKT(z, a1, a2)
            # pm = px-mu
            # Jz[:,0] = -pm[:,1]
            # Jz[:,1] = pm[:,0]
            # dv = a[self.component, np.newaxis] * Jz
            # zpx += dv
            Jz[:,0] = mu[:,1]
            Jz[:,1] = -mu[:,0]
            dv = a[self.component, np.newaxis] * Jz
            zpot = self.gradPotential(z)
            zpx += dv - zpot
            px2 = px - timeStep * zpx
            #ca = np.cos(timeStep*a)
            #sa = np.sin(timeStep*a)
            Jz[:,0] = ca[self.component]*px2[:,0] + sa[self.component]*px2[:,1]
            Jz[:,1] = -sa[self.component]*px2[:,0] + ca[self.component]*px2[:,1]
            #pxt[M-t-1, :, :] = Jz + timeStep * zpx
            pxt[t+1, :, :] = Jz
            try:
                theta = self.solveMKM(z,px2)
            except Exception:
                return xt[0:t,...], pxt[0:t,...], at[0:t,...], taut[0:t,...]
            at[t+1,:] = theta[range(0,len(theta),self.dim+1)]
            taut[t+1, :, 0] = theta[range(1, len(theta), self.dim + 1)]
            taut[t+1, :, 1] = theta[range(2, len(theta), self.dim + 1)]
            fvDef.updateVertices(xt[t+1,:,:])
            if pplot:
                fig=plt.figure(2)
                fig.clf()
                ax = fig.gca()
                if len(self.xc) > 0:
                    for kf in range(self.fvc.faces.shape[0]):
                        ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
                                self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[0, 0, 0])
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                            color=[0, 0, 1])
                for kf in range(fvDef.faces.shape[0]):
                    ax.plot(fvDef.vertices[fvDef.faces[kf,:],0], fvDef.vertices[fvDef.faces[kf,:],1], color=[1,0,0])
                for k in range(fvDef.component.max() + 1):
                    I = np.nonzero(fvDef.component == k)[0]
                    xDef = fvDef.vertices[I, :]
                    ax.plot(np.array([np.mean(xDef[:, 0]), xDef[0, 0]]),
                            np.array([np.mean(xDef[:, 1]), xDef[0, 1]]),
                            color=[0, 0, 1])
                    ax.plot(np.mean(xt[0:t, I, 0], axis=1), np.mean(xt[0:t, I, 1], axis=1))
                plt.title('t={0:.4f}'.format(t*timeStep))
                plt.axis('equal')
                plt.pause(0.001)
        if self.pplot:
            fig = plt.figure(2)
            fig.clf()
            ax = fig.gca()
            if len(self.xc)>0:
                for kf in range(self.fvc.faces.shape[0]):
                    ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0], self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[0, 0, 0])
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                        color=[0, 0, 1])
            for kf in range(fvDef.faces.shape[0]):
                ax.plot(fvDef.vertices[fvDef.faces[kf, :], 0], fvDef.vertices[fvDef.faces[kf, :], 1], color=[1, 0, 0])
            plt.axis('equal')
            plt.pause(0.001)
        return xt, pxt, at, taut

    def __geodesicEquation__(self, Tsize, p0):
        fv0 = self.fv0
        x0 = fv0.vertices
        xt = np.zeros([Tsize+1, x0.shape[0], x0.shape[1]])
        xt[0,:,:] = np.copy(x0)
        ncomponent = fv0.component.max() + 1
        component = np.zeros(x0.shape[0], dtype=int)
        for k in range(fv0.faces.shape[0]):
            component[fv0.faces[k,0]] = fv0.component[k]
            component[fv0.faces[k,1]] = fv0.component[k]
        Jz = np.zeros(x0.shape)
        at = np.zeros([Tsize+1, ncomponent])
        taut = np.zeros([Tsize+1, ncomponent, x0.shape[1]])
        pxt = np.zeros([Tsize+1, x0.shape[0], x0.shape[1]])
        pxt[0,:,:] = p0
        theta = self.solveMKM(x0, p0)
        at[0, :] = theta[range(0, len(theta), self.dim + 1)]
        taut[0, :, 0] = theta[range(1, len(theta), self.dim + 1)]
        taut[0, :, 1] = theta[range(2, len(theta), self.dim + 1)]
        fvDef = curves.Curve(curve=fv0)

        timeStep = 1.0/Tsize
        for t in range(Tsize):
            z = np.squeeze(xt[t, :, :])
            px = np.squeeze(pxt[t, :, :])
            a = np.squeeze(at[t, :])
            tau = np.squeeze(taut[t, :, :])
            Jz[:,0] = z[:,1]
            Jz[:,1] = -z[:,0]
            v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            ca = np.cos(timeStep*a)
            sa = np.sin(timeStep*a)
            Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
            Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            xt[t+1, :, :] = Jz + timeStep * tau[self.component,:]
            #xt[t+1, :, :] = xt[t, :, :] + timeStep * v
            try:
                mu = self.solveK(z,v)
            except Exception:
                return xt, pxt, at, taut
            a1 = self.regweight*mu[np.newaxis,...]
            a2 = mu[np.newaxis,...]
            zpx = self.param.KparDiff.applyDiffKT(z, a1, a2)
            Jz[:,0] = mu[:,1]
            Jz[:,1] = -mu[:,0]
            dv = a[self.component, np.newaxis] * Jz
            zpx += dv
            px2 = px - timeStep * zpx
            Jz[:,0] = ca[self.component]*px2[:,0] + sa[self.component]*px2[:,1]
            Jz[:,1] = -sa[self.component]*px2[:,0] + ca[self.component]*px2[:,1]
            #pxt[M-t-1, :, :] = Jz + timeStep * zpx
            pxt[t+1, :, :] = px2
            # pm = px-mu
            # Jz[:,0] = -pm[:,1]
            # Jz[:,1] = pm[:,0]
            # dv = a[self.component, np.newaxis] * Jz
            # zpx += dv
            # px2 = px - timeStep * zpx
            # pxt[t+1, :, :] = px2
            try:
                theta = self.solveMKM(z,px2)
            except Exception:
                return xt, pxt, at, taut
            at[t+1,:] = theta[range(0,len(theta),self.dim+1)]
            taut[t+1, :, 0] = theta[range(1, len(theta), self.dim + 1)]
            taut[t+1, :, 1] = theta[range(2, len(theta), self.dim + 1)]
            fvDef.updateVertices(xt[t+1,:,:])
        if self.pplot:
            fig=plt.figure(3)
            fig.clf()
            ax = fig.gca()
            if len(self.xc)>0:
                for kf in range(self.fvc.faces.shape[0]):
                    ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0], self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[0, 0, 0])
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                        color=[0, 0, 1])
            for kf in range(fvDef.faces.shape[0]):
                ax.plot(fvDef.vertices[fvDef.faces[kf,:],0], fvDef.vertices[fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
            plt.axis('equal')
            plt.pause(0.001)
        return xt, pxt, at, taut

    def hamiltonianCovector(self, px1, affine = None):
        x0 = self.x0
        at = self.at
        taut = self.taut
        KparDiff = self.param.KparDiff
        N = x0.shape[0]
        dim = x0.shape[1]
        M = at.shape[0]
        timeStep = 1.0/M
        xt = self.directEvolutionEuler(x0, at, taut)
        px1 -= self.gradPotential(xt[-1,:,:])*timeStep

        pxt = np.zeros([M+1, N, dim])
        pxt[M, :, :] = px1
        foo = curves.Curve(curve=self.fv0)
        for t in range(M):
            px = np.squeeze(pxt[M-t, :, :])
            z = np.squeeze(xt[M-t-1, :, :])
            a = np.squeeze(at[M-t-1, :])
            tau = np.squeeze(taut[M-t-1, :, :])
            Jz = np.zeros(z.shape)
            Jz[:,0] = z[:,1]
            Jz[:,1] = -z[:,0]
            v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            #K = self.param.KparDiff.getK(z)
            mu = self.solveK(z,v)
            foo.updateVertices(z)
            a1 = self.regweight*mu[np.newaxis,...]
            a2 = mu[np.newaxis,...]
            zpx = self.param.KparDiff.applyDiffKT(z, a1, a2)
            #pm = px-mu
            Jz[:,0] = mu[:,1]
            Jz[:,1] = -mu[:,0]
            dv = a[self.component, np.newaxis] * Jz
            #self.testGradPotential(z)
            zpx += dv - self.gradPotential(z)
            ca = np.cos(timeStep*a)
            sa = np.sin(timeStep*a)
            Jz[:,0] = ca[self.component]*px[:,0] - sa[self.component]*px[:,1]
            Jz[:,1] = sa[self.component]*px[:,0] + ca[self.component]*px[:,1]
            pxt[M-t-1, :, :] = Jz + timeStep * zpx
        return pxt, xt

    def hamiltonianGradient(self, px1):
        foo = curves.Curve(curve=self.fv0)
        timeStep = 1.0/self.Tsize
        (pxt, xt) = self.hamiltonianCovector(px1)
        at = self.at        
        dat = np.zeros(at.shape)
        taut = self.taut
        dtaut = np.zeros(taut.shape)
        for k in range(at.shape[0]):
            z = np.squeeze(xt[k,...])
            zc = np.concatenate([z,self.xc])
            foo.updateVertices(z)
            a = np.squeeze(at[k, :])
            tau = np.squeeze(taut[k, :, :])
            px = np.squeeze(pxt[k+1, :, :])
            Jz = np.zeros(z.shape)
            Jz[:,0] = z[:,1]
            Jz[:,1] = -z[:,0]
            v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            vc = np.concatenate([v,np.zeros(self.xc.shape)])
            K = self.param.KparDiff.getK(zc)
            #mu = la.solve(K,v)
            eK = la.eigh(K)
            J = np.nonzero(eK[0]>self.minEig)[0]
            w = np.dot(vc.T,eK[1][:,J]).T/eK[0][J, np.newaxis]
            mu = np.dot(eK[1][:,J],w)[0:self.npt,:]
            p1 = mu * Jz
            ca = np.cos(timeStep*a)
            sa = np.sin(timeStep*a)
            Jz[:,0] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            Jz[:,1] = -ca[self.component]*z[:,0] + -sa[self.component]*z[:,1]
            p1 -= px*Jz
            for j in range(self.ncomponent):
                I = np.nonzero(self.component == j)
                I = I[0]
                dat[k, j] = p1[I,:].sum()
                dtaut[k,j,:] = (mu-px)[I,:].sum(axis=0)

        return dat, dtaut, xt, pxt


    def endPointGradient(self):
        px = self.param.fun_objGrad(self.fvDef, self.fv1, self.param.KparDist)
        return px / self.param.sigmaError**2


    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        foo = self.hamiltonianGradient(px1)
        grd = Direction()
        grd.skew = foo[0]/(coeff*self.Tsize)
        grd.trans = foo[1]/(coeff*self.Tsize)
        self.pxt = foo[3]
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.skew = dir1.skew + beta * dir2.skew
        dir.trans = dir1.trans + beta * dir2.trans
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.skew = np.copy(dir0.skew)
        dir.trans = np.copy(dir0.trans)

        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.skew = np.random.randn(self.Tsize, self.ncomponent)
        dirfoo.trans = np.random.randn(self.Tsize, self.ncomponent, self.dim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        #dim2 = self.dim**2
        for t in range(self.Tsize):
            gs = np.squeeze(g1.skew[t, :])
            gt = np.squeeze(g1.trans[t, :, :])
            for ll,gr in enumerate(g2):
                ggs = np.squeeze(gr.skew[t, :])
                ggt = np.squeeze(gr.trans[t, :, :])
                res[ll] += (gs*ggs).sum() + (gt*ggt).sum()

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.taut = np.copy(self.tautTry)

    def endOfIteration(self):
        (obj1, self.xt) = self.objectiveFunDef(self.at, self.taut, withTrajectory=True)
        self.iter += 1

        if self.saveRate > 0 and self.iter%self.saveRate==0:
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
            self.geodesicEquation(self.Tsize, self.at[0, :], self.taut[0, :, :])
            #self.__geodesicEquation__(self.Tsize, self.pxt[0, :,:])
        else:
            self.fvDef.updateVertices(np.squeeze(self.xt[self.Tsize, :, :]))

        if self.pplot:
            fig=plt.figure(1)
            fig.clf()
            ax = fig.gca()
            if len(self.xc)>0:
                for kf in range(self.fvc.faces.shape[0]):
                    ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0], self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[1, 0, 0])
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf,:],0], self.fv1.vertices[self.fv1.faces[kf,:],1], color=[0,0,1])
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
            plt.axis('equal')
            plt.pause(0.1)
                

    def endOptim(self):
        if self.saveRate==0 or self.iter%self.saveRate > 0:
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
        self.defCost = self.obj - self.obj0 - self.dataTerm(self.fvDef)   


    def optimizeMatching(self):
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(self.gradLB, np.sqrt(grd2) / 100000)
        print 'Gradient bound:', self.gradEps
        kk = 0
        while os.path.isfile(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk'):
            os.remove(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
            kk += 1
        cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt

    def run(self):
        plt.ion()
        N = 50
        t = np.arange(0., 2 * np.pi, 2*np.pi/N)
        r = 0.25
        x = np.zeros((len(t), 2))
        x[:,0] = r*np.cos(t)
        x[:,1] = r*np.sin(t)
        fv0 = [curves.Curve(pointSet=x+np.array([0.5,0.5])), curves.Curve(pointSet=x+np.array([0, 1]))]
        fv1 = [curves.Curve(pointSet=x+np.array([0, 1])), curves.Curve(pointSet=x+np.array([0, 1]))]

        sigma = .1
        K1 = Kernel(name='laplacian', sigma=sigma)
        sigmaDist = 2.0
        sigmaError = 0.01
        dirOut = '/Users/younes'

        if os.path.isfile(dirOut + '/Development/Results/curveMatchingRigid/info.tex'):
            os.remove(dirOut + '/Development/Results/curveMatchingRigid/info.tex')
        loggingUtils.setup_default_logging(dirOut + '/Development/Results/curveMatchingRigid', fileName='info.txt',
                                           stdOutput=True)

        sm = CurveMatchingRigidParam(timeStep=0.01, KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError, errorType='varifold')
        f = CurveMatchingRigid(Template=fv0, Target=fv1, outputDir=dirOut + '/Development/Results/curveRigid', param=sm,
                          testGradient=True, gradLB=1e-5, saveTrajectories=True, regWeight=1., maxIter=10000)
        f.optimizeMatching()
        f.geodesicEquation(f.Tsize, f.at[0,:], f.taut[0,:,:])
        #f.__geodesicEquation__(f.Tsize, f.fv0, f.pxt[0,:,:])

        logging.shutdown()
        plt.ioff()
        plt.show()
        return f

    def __circle(self,N,r):
        t = np.arange(0., 2 * np.pi, 2*np.pi/N)
        x = np.zeros((len(t), 2))
        x[:, 0] = r * np.cos(t)
        x[:, 1] = r * np.sin(t)
        return x

    def __square(self,N,r):
        t = np.arange(0., 1., 4./N)[:,np.newaxis]
        x = np.concatenate([t*[1,0], [1,0] + t*[0,-1], [1,-1] + t*[-1,0], [0,-1]+t*[0,1]])
        x -= [.5,-.5]
        x *= 2*r
        return x


    def shootingScenario(self, scenario = 1, T=5, dt=0.001):
        dirOut = '/Users/younes'
        if os.path.isfile(dirOut + '/Development/Results/curveMatchingRigid/info.tex'):
            os.remove(dirOut + '/Development/Results/curveMatchingRigid/info.tex')
        loggingUtils.setup_default_logging(dirOut + '/Development/Results/curveMatchingRigid', fileName='info.txt',
                                           stdOutput=True)
        sigma = .05
        K1 = Kernel(name='laplacian', sigma=sigma)
        sigmaDist = 2.0
        sigmaError = 0.01

        sm = CurveMatchingRigidParam(timeStep=dt / T, KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                                     errorType='varifold')
        if scenario == 1:
            x = self.__circle(50, 0.25)
            # x = self.__square(50, 0.25)
            fv0 = [curves.Curve(pointSet=x + np.array([-1, 0])),
                   curves.Curve(pointSet=x + np.array([.9, 1.])),
                   curves.Curve(pointSet=x + np.array([.8, 0]))]
            x = self.__square(200, 2)
            # fvc = curves.Curve(pointSet=x + np.array([.3, 1]))
            fvc = curves.Curve(pointSet=x)
            a0 = np.array([.05, 0, 0])
            tau0 = np.array([[1, -.05], [0, 0], [0, 0]])

            f = CurveMatchingRigid(Template=fv0, Target=fv0, Clamped=fvc,
                                   outputDir=dirOut + '/Development/Results/curveRigid', param=sm,
                                   testGradient=True, gradLB=1e-5, saveTrajectories=True,
                                   regWeight=1., maxIter=10000)
            return f, T*a0, T*tau0

    def runShoot(self, dt=0.001):
        plt.ion()
        S = self.shootingScenario(1,dt=dt, T=20)
        f = S[0]
        a0 = S[1]
        tau0 = S[2]
        f.parpot = -.05
        geod = f.geodesicEquation2(f.Tsize, a0, tau0,pplot=False)
        # f.__geodesicEquation__(f.Tsize, f.fv0, f.pxt[0,:,:])
        fig = plt.figure(2)
        fvDef = curves.Curve(curve=f.fv0)
        xt = geod[0]
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        dirMov = '/Users/younes/OneDrive - Johns Hopkins University/TALKS/MECHANICAL/Videos/'
        with writer.saving(fig, dirMov+"threeBallPotential01Try.mp4", 100):
            for t in range(0,xt.shape[0], xt.shape[0]/100):
                fig.clf()
                ax = fig.gca()
                fvDef.updateVertices(xt[t,:,:])
                if len(f.xc)>0:
                    for kf in range(f.fvc.faces.shape[0]):
                        ax.plot(f.fvc.vertices[f.fvc.faces[kf, :], 0], f.fvc.vertices[f.fvc.faces[kf, :], 1], color=[0, 0, 0])
                for kf in range(fvDef.faces.shape[0]):
                    ax.plot(fvDef.vertices[fvDef.faces[kf, :], 0], fvDef.vertices[fvDef.faces[kf, :], 1], color=[1, 0, 0])
                for k in range(fvDef.component.max()+1):
                    I = np.nonzero(fvDef.component==k)[0]
                    xDef = fvDef.vertices[I, :]
                    ax.plot(np.array([np.mean(xDef[:,0]),xDef[0, 0]]),
                            np.array([np.mean(xDef[:,1]), xDef[0, 1]]),
                            color = [0,0,1])
                    ax.plot(np.mean(xt[0:t,I,0], axis=1), np.mean(xt[0:t,I,1], axis=1))
                plt.axis('equal')
                plt.title('t={0:.3f}'.format(t*dt))
                writer.grab_frame()
                plt.pause(0.001)

        logging.shutdown()
        plt.ioff()
        plt.show()
        return f


if __name__ == "__main__":
    CurveMatchingRigid().runShoot()
