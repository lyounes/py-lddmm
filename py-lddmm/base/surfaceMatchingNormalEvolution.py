import numpy as np
import logging
from . import surfaces
from . import pointSets
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, bfgs
from . import surfaceMatching
from .affineBasis import getExponential, gradExponential
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDiffOut: background kernel: if not specified, use typeKernel with width sigmaKernelOut
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class SurfaceMatchingParam(surfaceMatching.SurfaceMatchingParam):
#     def __init__(self,  timeStep=.1, algorithm='bfgs', Wolfe=True, KparDiff=None, KparDist=None, KparDiffOut=None,
#                  sigmaError = 1., errorType='varifold', internalCost=None):
#         surfaceMatching.SurfaceMatchingParam.__init__(self, timeStep=timeStep, KparDiff=KparDiff,
#                                                       KparDist=KparDist, algorithm = algorithm, Wolfe=Wolfe,
#                                                       sigmaError=sigmaError,
#                                                       errorType=errorType, internalCost=internalCost)
#         if KparDiffOut is None:
#             self.KparDiffOut = self.KparDiff
#             self.sigmaKernelOut = self.sigmaKernel
#         elif type(KparDiffOut) in (list,tuple):
#             self.typeKernelOut = KparDiffOut[0]
#             self.sigmaKernelOut = KparDiffOut[1]
#             if self.typeKernelOut == 'laplacian' and len(KparDiff) > 2:
#                 self.orderKernelOut = KparDist[2]
#             self.KparDist = kfun.Kernel(name = self.typeKernelOut, sigma = self.sigmaKernelOut, order=self.orderKernelOut)
#         else:
#             self.KparDiffOut = KparDiffOut
#
#         if KparDiffOut is None:
#             self.KparDiffOut = kfun.Kernel(name=self.typeKernel, sigma=self.sigmaKernelOut)
#         else:
#             self.KparDiffOut = KparDiffOut
#

## Main class for surface matching
#        Template: sequence of surface classes (from surface.py); if not specified, opens files in fileTemp
#        Target: sequence of surface classes (from surface.py); if not specified, opens files in fileTarg
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

class SurfaceMatching(surfaceMatching.SurfaceMatching):
    def __init__(self, Template=None, Target=None, options=None):
        super(SurfaceMatching, self).__init__(Template=Template, Target=Target, options=options)
        if self.options['affine'] != 'none':
             logging.warning('Warning: Affine transformations should not be used with this function.')
                 # param=param, maxIter=maxIter_cg,
                 #                              regWeight=regWeight,
                 #                              internalWeight=internalWeight, affineWeight=affineWeight,
                 #                              mode=mode, subsampleTargetSize=-1, rotWeight=rotWeight,
                 #                              scaleWeight=scaleWeight, transWeight=transWeight,
                 #                              symmetric=symmetric, pplot = pplot, testGradient=testGradient,
                 #                              saveFile=saveFile, saveTrajectories=saveTrajectories, affine=affine,
                 #                              outputDir=outputDir)

        self.cval = np.zeros([self.Tsize + 1, self.npt])
        self.cstr = np.zeros([self.Tsize + 1, self.npt])
        self.lmb = np.zeros([self.Tsize + 1, self.npt])
        self.nu = np.zeros([self.Tsize + 1, self.npt, self.dim])

        self.mu = self.options['mu']
        self.ds = 1.0
        self.meanc = 0
        self.converged = False

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['maxIter_grad'] = 1000
        options['maxIter_al'] = 100
        options['mu'] = 1.0
        options['KparDiffOut'] = None
        return options

    def set_parameters(self):
        super().set_parameters()
        if type(self.options['KparDiffOut']) in (list,tuple):
            typeKernel = self.options['KparDiffOut'][0]
            sigmaKernel = self.options['KparDiffOut'][1]
            if typeKernel == 'laplacian' and len(self.options['KparDiffOut']) > 2:
                orderKernel = self.options['KparDiffOut'][2]
            else:
                orderKernel = 0
            self.options['KparDiff'] = kfun.Kernel(name = typeKernel, sigma = sigmaKernel, order=orderKernel)

        if self.options['KparDiffOut'] is None:
            self.options['KparDiffOut'] = self.options['KparDiff']

    def constraintTerm(self, state, control):
        at = control['at']
        xt = state['xt']
        Afft = control['Afft']
        timeStep = 1.0 / self.Tsize
        obj = 0
        cval = np.zeros(self.cval.shape)
        for t in range(self.Tsize + 1):
            x = xt[t]
            #if t < self.Tsize:

            nu = np.zeros(x.shape)
            fk = self.fv0.faces
            xDef0 = x[fk[:, 0], :]
            xDef1 = x[fk[:, 1], :]
            xDef2 = x[fk[:, 2], :]
            nf = np.cross(xDef1 - xDef0, xDef2 - xDef0)
            for kk, j in enumerate(fk[:, 0]):
                nu[j, :] += nf[kk, :]
            for kk, j in enumerate(fk[:, 1]):
                nu[j, :] += nf[kk, :]
            for kk, j in enumerate(fk[:, 2]):
                nu[j, :] += nf[kk, :]
            nu /= np.sqrt((nu ** 2).sum(axis=1)).reshape([nu.shape[0], 1])
            nu *= self.fv0ori
            self.nu[t, ...] = nu

            if t < self.Tsize:
                a = at[t]
                r = self.options['KparDiff'].applyK(x, a)
                self.v[t, ...] = r
                # cval[t,...] = ((r*r).sum(axis=1) - ((nu*r).sum(axis=1))**2)/2
                cval[t, ...] = (np.sqrt((r * r).sum(axis=1)) - (nu * r).sum(axis=1)) / 2
                obj += timeStep * ((-self.lmb[t, ...] * cval[t, ...]).sum() + (cval[t, ...] ** 2).sum() / (2 * self.mu))

        # print 'cstr', obj
        return obj, cval

    def constraintTermGrad(self, state, control):
        at = control['at']
        xt = state['xt']
        Afft = control['Afft']
        lmb = np.zeros(self.cval.shape)
        dxcval = np.zeros(xt.shape)
        dacval = np.zeros(at.shape)
        if Afft is not None:
            dAffcval = np.zeros(Afft.shape)
        else:
            dAffcval = None
        # for t in (0, self.Tsize-1):
        for t in range(self.Tsize):
            a = at[t]
            x = xt[t]
            fk = self.fv0.faces
            nu = np.zeros(x.shape)
            xDef0 = x[fk[:, 0], :]
            xDef1 = x[fk[:, 1], :]
            xDef2 = x[fk[:, 2], :]
            nf = np.cross(xDef1 - xDef0, xDef2 - xDef0)
            for kk, j in enumerate(fk[:, 0]):
                nu[j, :] += nf[kk, :]
            for kk, j in enumerate(fk[:, 1]):
                nu[j, :] += nf[kk, :]
            for kk, j in enumerate(fk[:, 2]):
                nu[j, :] += nf[kk, :]
            normNu = np.sqrt((nu ** 2).sum(axis=1))
            nu /= normNu.reshape([nu.shape[0], 1])
            nu *= self.fv0ori
            vt = self.options['KparDiff'].applyK(x, a)
            normvt = np.sqrt((vt * vt).sum(axis=1))
            vnu = (nu * vt).sum(axis=1)
            lmb[t, :] = self.lmb[t, :] - (normvt - vnu) / (2 * self.mu)
            lnu = - nu * lmb[t, :, np.newaxis] / 2  # np.multiply(nu, lmb[t, :].reshape([self.npt, 1]))
            lv = vt * lmb[t, :, np.newaxis] / 2
            lnu += lv / (np.maximum(normvt[:, np.newaxis], 1e-6))
            # lv = lv * vnu[:,np.newaxis]
            dxcval[t] = self.options['KparDiff'].applyDiffKT(x, a, lnu)
            dxcval[t] += self.options['KparDiff'].applyDiffKT(x, lnu, a)
            if self.euclideanGradient:
                dacval[t] = self.options['KparDiff'].applyK(x, lnu)
            else:
                dacval[t] = np.copy(lnu)
            dAffcval = []
            # if self.affineDim > 0:
            #     dAffcval[t, :] = (np.dot(self.affineBasis.T, np.vstack([np.dot(lnu.T, x).reshape([dim2,1]), lnu.sum(axis=0).reshape([self.dim,1])]))).flatten()
            lv /= normNu.reshape([nu.shape[0], 1])
            lv -= np.multiply(nu, np.multiply(nu, lv).sum(axis=1).reshape([nu.shape[0], 1]))
            lvf = lv[fk[:, 0]] + lv[fk[:, 1]] + lv[fk[:, 2]]
            dnu = np.zeros(x.shape)
            foo = np.cross(xDef2 - xDef1, lvf)
            for kk, j in enumerate(fk[:, 0]):
                dnu[j, :] += foo[kk, :]
            foo = np.cross(xDef0 - xDef2, lvf)
            for kk, j in enumerate(fk[:, 1]):
                dnu[j, :] += foo[kk, :]
            foo = np.cross(xDef1 - xDef0, lvf)
            for kk, j in enumerate(fk[:, 2]):
                dnu[j, :] += foo[kk, :]
            # dxcval[t] -= self.fv0ori*dnu
            dxcval[t] += self.fv0ori * dnu

        # print 'testg', (lmb**2).sum()
        return lmb, dxcval, dacval, dAffcval

    def testConstraintTerm(self, state, control):
        at = control['at']
        Afft = control['Afft']
        xt = state['xt']
        eps = 0.00000001
        xtTry = xt + eps * np.random.randn(self.Tsize + 1, self.npt, self.dim)
        atTry = at + eps * np.random.randn(self.Tsize, self.npt, self.dim)
        # if self.affineDim > 0:
        #     AfftTry = Afft + eps*np.random.randn(self.Tsize, self.affineDim)


        u0 = self.constraintTerm(xt, control)
        ux = self.constraintTerm(xtTry, control)
        ua = self.constraintTerm(xt, {'at': atTry, 'Afft': Afft})
        [l, dx, da, dA] = self.constraintTermGrad(xt, control)
        vx = np.multiply(dx, xtTry - xt).sum() / eps
        va = np.multiply(da, atTry - at).sum() / eps
        logging.info('Testing constraints:')
        logging.info('var x: %f %f' % ((ux[0] - u0[0]) / (eps), -vx))
        logging.info('var a: %f %f' % ((ua[0] - u0[0]) / (eps), -va))
        # if self.affineDim > 0:
        #     uA = self.constraintTerm(xt, at, AfftTry)
        #     vA = np.multiply(dA, AfftTry-Afft).sum()/eps
        #     logging.info('var affine: %f %f' %(self.Tsize*(uA[0]-u0[0])/(eps), -vA ))

    def objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False):
        obj_, f = super(SurfaceMatching, self).objectiveFunDef(control, withTrajectory=True, withJacobian=withJacobian,
                                                         var = var)
        cstr = self.constraintTerm(f, control)
        obj = obj_ + cstr[0]

        # print f[0], cstr[0]

        if withJacobian or withTrajectory:
            return obj, f, cstr[1]
        else:
            return obj

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError'] ** 2)
            if self.options['symmetric']:
                self.obj0 += self.fun_obj0(self.fv0) / (self.options['sigmaError'] ** 2)

            (self.obj, self.state, self.cval) = self.objectiveFunDef(self.control, withTrajectory=True)
            self.obj += self.obj0

            self.fvDef.updateVertices(np.squeeze(self.state['xt'][self.Tsize, ...]))
            if self.fv1:
                self.obj += self.fun_obj(self.fvDef, self.fv1) / (self.options['sigmaError'] ** 2)
            else:
                self.obj += self.fun_obj(self.fvDef) / (self.options['sigmaError'] ** 2)
            if self.options['symmetric']:
                self.fvInit.updateVertices(np.squeeze(self.control['x0']))
                self.obj += self.fun_obj(self.fvInit, self.fv0, self.options['KparDist']) / (
                self.options['sigmaError'] ** 2)

        return self.obj

    # def getVariable(self):
    #     return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        controlTry = surfaceMatching.Control()
        for k in dir:
            if dir[k] is not None:
                controlTry[k] = self.control[k] - eps * dir[k]
        # atTry = self.at - eps * dir['diff']
        # if self.affineDim > 0:
        #     AfftTry = self.Afft - eps * dir['aff']
        # else:
        #     AfftTry = []
        # if self.options['symmetric']:
        #     x0Try = self.x0 - eps * dir['initx']
        # else:
        #     x0Try = self.x0
        ff = surfaces.Surface(surf=self.fv0)
        ff.updateVertices(controlTry['x0'])
        obj_, stateTry, cvalTry = self.objectiveFunDef(controlTry, var = {'fv0':ff}, withTrajectory=True)
        objTry = 0

        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(stateTry['xt'][self.Tsize, ...]))
        if self.fv1:
            objTry += self.fun_obj(ff, self.fv1) / (self.options['sigmaError'] ** 2)
        else:
            objTry += self.fun_obj(ff) / (self.options['sigmaError'] ** 2)
        if self.options['symmetric']:
            ffI = surfaces.Surface(surf=self.fvInit)
            ffI.updateVertices(controlTry['x0'])
            objTry += self.fun_obj(ffI, self.fv0) / (self.options['sigmaError'] ** 2)
        objTry += obj_ + self.obj0

        if np.isnan(objTry):
            logging.warning('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = controlTry
            self.cval = cvalTry
            self.objTry = objTry

        return objTry

    def covectorEvolution(self, control, px1):
        at = control['at']
        Afft = control['Afft']
        x0 = control['x0']
        M = self.Tsize
        timeStep = 1.0 / M
        dim2 = self.dim ** 2
        if Afft is not None:
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2 + self.dim]
        else:
            A = None
        xt = evol.landmarkDirectEvolutionEuler(x0, at*self.ds, self.options['KparDiff'], affine=A)
        # xt = xJ
        pxt = np.zeros([M + 1, self.npt, self.dim])
        pxt[M, :, :] = px1

        foo = self.constraintTermGrad({'xt':xt}, control)
        # lmb = foo[0]
        dxcval = foo[1]
        dacval = foo[2]
        dAffcval = foo[3]

        pxt[M, :, :] += dxcval[M] * timeStep
        foo = surfaces.Surface(surf=self.fv0)

        for t in range(M):
            px = np.squeeze(pxt[M - t, :, :])
            z = np.squeeze(xt[M - t - 1, :, :])
            a = np.squeeze(at[M - t - 1, :, :])
            zpx = np.copy(dxcval[M - t - 1])
            foo.updateVertices(z)
            v = self.options['KparDiff'].applyK(z, a)*self.ds
            if self.internalCost:
                grd = self.internalCostGrad(foo, v)
                Lv = grd[0]
                DLv = self.options['internalWeight'] * self.options['regWeight'] * grd[1]
                zpx += self.options['KparDiff'].applyDiffKT(z, px, a*self.ds, regweight=self.options['regWeight'],
                            lddmm=True, extra_term=-self.options['internalWeight'] * self.options['regWeight']*Lv) - DLv
            else:
                zpx += self.options['KparDiff'].applyDiffKT(z, px, a*self.ds, regweight=self.options['regWeight'], lddmm=True)
            if self.affineDim > 0:
                pxt[M - t - 1, :, :] = np.dot(px, getExponential(timeStep * A[0][M - t - 1])) + timeStep * zpx
            else:
                pxt[M - t - 1, :, :] = px + timeStep * zpx

        return pxt, xt, dacval, dAffcval

    def HamiltonianGradient(self, control, px1, getCovector=False):
        at = control['at']
        Afft = control['Afft']
        (pxt, xt, dacval, dAffcval) = self.covectorEvolution(control, px1)

        foo = surfaces.Surface(surf=self.fv0)
        if not self.euclideanGradient:
            dat = - dacval
            for t in range(self.Tsize):
                z = np.squeeze(xt[t, ...])
                foo.updateVertices(z)
                a = np.squeeze(at[t, :, :])
                px = np.squeeze(pxt[t + 1, :, :])
                v = self.options['KparDiff'].applyK(z, a)*self.ds
                dat[t, :, :] += 2 * self.options['regWeight'] * a *self.ds**2 - px * self.ds
                if self.internalCost:
                    Lv = self.internalCostGrad(foo, v, variables='phi')
                    dat[t, :, :] += self.options['regWeight'] * self.options['internalWeight'] * Lv * self.ds
        else:
            dat = -dacval
            for t in range(self.Tsize):
                z = np.squeeze(xt[t, ...])
                foo.updateVertices(z)
                a = np.squeeze(at[t, :, :])
                px = np.squeeze(pxt[t + 1, :, :])
                v = self.options['KparDiff'].applyK(z, a)
                if self.internalCost:
                    Lv = self.internalCostGrad(foo, v, variables='phi')
                    dat[t, :, :] += self.options['KparDiff'].applyK(z, 2 * self.options['regWeight'] * a - px +
                                                            self.options['regWeight'] * self.options['internalWeight'] * Lv)
                else:
                    dat[t, :, :] += self.options['KparDiff'].applyK(z, 2 * self.options['regWeight'] * a - px)
        if self.affineDim > 0:
            timeStep = 1.0 / self.Tsize
            dAfft = 2 * np.multiply(self.affineWeight.reshape([1, self.affineDim]), Afft)
            # dAfft = 2*np.multiply(self.affineWeight, Afft) - dAffcval
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A = AB[0:self.dim ** 2].reshape([self.dim, self.dim])
                dA = gradExponential(timeStep * A, pxt[t + 1], xt[t]).reshape([self.dim ** 2, 1])
                db = pxt[t + 1].sum(axis=0).reshape([self.dim, 1])
                dAff = np.dot(self.affineBasis.T, np.vstack([dA, db]))
                dAfft[t] -= dAff.reshape(dAfft[t].shape)
        else:
            dAfft = None

        if getCovector == False:
            return dat, dAfft, xt
        else:
            return dat, dAfft, xt, pxt


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            endPoint = self.fvDef
            #A = self.affB.getTransforms(self.Afft)
        else:
            control = surfaceMatching.Control()
            for k in update[0]:
                if update[0][k] is not None:
                    control[k] = self.control[k] - update[1] * update[0][k]
            A = self.affB.getTransforms(control['Afft'])
            xt = evol.landmarkDirectEvolutionEuler(control['x0'], control['at']*self.ds, self.options['KparDiff'], affine=A)
            endPoint = surfaces.Surface(surf=self.fv0)
            endPoint.updateVertices(xt[-1, :, :])

        px1 = -self.endPointGradient(endPoint=endPoint)
        foo = self.HamiltonianGradient(control, px1, getCovector=True)
        grd = surfaceMatching.Control()
        grd['at'] = foo[0]/(coeff*self.Tsize)

        if self.affineDim > 0:
            grd['Afft'] = foo[1] / (self.coeffAff * coeff * self.Tsize)
        if self.options['symmetric']:
            grd['x0'] = (self.initPointGradient() - foo[3][0, ...]) / (self.coeffInitx * coeff)
        else:
            grd['x0'] = np.zeros((self.npt, self.dim))
        return grd

    def saveEvolution(self, fv0, state, fileName='evolution', velocity = None, orientation= None,
                      constraint = None, normals=None):
        xt = state['xt']
        Jacobian = state['xt']
        if velocity is None:
            velocity = self.v
        if orientation is None:
            orientation = self.fv0ori
        if constraint is None:
            constraint = self.cstr
        if normals is None:
            normals = self.nu
        fvDef = surfaces.Surface(surf=fv0)
        AV0 = fvDef.computeVertexArea()
        nu = orientation * fv0.computeVertexNormals()
        v = velocity[0, ...]
        npt = fv0.vertices.shape[0]
        displ = np.zeros(npt)
        area_displ = np.zeros((self.Tsize + 1, npt))
        dt = 1.0 / self.Tsize
        for kk in range(self.Tsize + 1):
            fvDef.updateVertices(np.squeeze(xt[kk, :, :]))
            AV = fvDef.computeVertexArea()
            AV = (AV[0] / AV0[0])
            vf = surfaces.vtkFields()
            if Jacobian is not None:
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jacobian[kk, :, 0]))
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(np.exp(Jacobian[kk, :, 0]) / AV)
                vf.scalars.append('displacement')
            vf.scalars.append(displ)
            if kk < self.Tsize:
                nu = orientation * fvDef.computeVertexNormals()
                v = velocity[kk, ...]
                kkm = kk
            else:
                kkm = kk - 1
            vf.vectors.append('velocity')
            vf.vectors.append(velocity[kkm, :])
            if kk > 0:
                area_displ[kk, :] = area_displ[kk - 1, :] + dt * np.fabs((AV) * np.sqrt((v * v).sum(axis=1)))[np.newaxis, :]
            #fvDef.saveVTK2(self.outputDir + '/' + fileName + str(kk) + '.vtk', vf)
            displ += dt * (v * nu).sum(axis=1)
            vf.scalars.append('area_displacement')
            vf.scalars.append(area_displ[kk,:])
            vf.scalars.append('constraint')
            vf.scalars.append(constraint[kkm, :])
            vf.vectors.append('normals')
            vf.vectors.append(normals[kkm, :])
            fvDef.saveVTK2(self.outputDir + '/' + self.options['saveFile'] + str(kk) + '.vtk', vf)
            #
        adisp = area_displ / np.maximum(area_displ[-1, :][np.newaxis,:], 1e-10)
        fvDef = surfaces.Surface(surf=fv0)
        fvDef.saveVTK(self.outputDir + '/' + self.options['saveFile'] +'_bok0' + '.vtk')
        x = np.zeros((npt, self.dim))
        for kk in range(1, self.Tsize+1):
            Inext = ((adisp - kk/self.Tsize)>-1e-10).argmax(axis=0)
            for jj in range(npt):
                r = (adisp[Inext[jj], jj] - kk/self.Tsize)/np.maximum(adisp[Inext[jj], jj] - adisp[Inext[jj]-1, jj], 1e-10)
                x[jj] = r*self.state['xt'][Inext[jj]-1,jj,:] + (1-r)*self.state['xt'][Inext[jj],jj,:]
            fvDef.updateVertices(x)
            fvDef.saveVTK(self.outputDir + '/' + self.options['saveFile'] + '_bok' + str(kk) + '.vtk')
            # self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')

    def startOfIteration(self):
        if self.reset:
            self.options['KparDiff'].pk_dtype = 'float64'
            self.options['KparDist'].pk_dtype = 'float64'


    def endOfIteration(self, forceSave = False):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        if (forceSave or self.iter % self.saveRate == 0):
            (obj1, self.state, self.cval) = self.objectiveFunDef(self.control, withJacobian=True)
            self.meanc = np.sqrt((self.cval ** 2).sum() / 2)
            logging.info('mean constraint %f max constraint %f' % (self.meanc, np.fabs(self.cval).max()))
            logging.info('saving data')

            self.fvInit.updateVertices(self.control['x0'])
            if self.options['saveTrajectories']:
                pointSets.saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curves.vtk', self.state['xt'])

            if self.options['affine'] == 'euclidean' or self.options['affine'] == 'translation':
                self.saveCorrectedEvolution(self.fvInit, self.state, self.control, fileName=self.options['saveFile'])
            self.saveEvolution(self.fvInit, self.state, fileName=self.options['saveFile'])

            # # # nn = 0 ;
            # AV0 = self.fvInit.computeVertexArea()
            # nu = self.fv0ori * self.fvInit.computeVertexNormals()
            # v = self.v[0, ...]
            # # displ = np.zeros(self.npt)
            # area_dis0pl = np.zeros((self.Tsize+1, self.npt))
            # dt = 1.0 / self.Tsize
            # # # n1 = self.xt.shape[1] ;
            # # for kk in range(self.Tsize):
            # #     self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
            # #     nu = self.fv0ori * self.fvDef.computeVertexNormals()
            # #     v = self.v[kk, ...]
            # #     displ += dt * (v * nu).sum(axis=1)
            # for kk in range(self.Tsize + 1):
            # # #     self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
            #     AV = self.fvDef.computeVertexArea()
            #     AV = (AV[0] / AV0[0]) - 1
            # #     vf = surfaces.vtkFields()
            # #     vf.scalars.append('Jacobian')
            # #     vf.scalars.append(np.exp(Jt[kk, :, 0]))
            # #     vf.scalars.append('Jacobian_T')
            # #     vf.scalars.append(AV)
            # #     vf.scalars.append('Jacobian_N')
            # #     vf.scalars.append(np.exp(Jt[kk, :, 0]) / (AV + 1) - 1)
            # #     vf.scalars.append('displacement')
            # #     vf.scalars.append(displ)
            #     if kk < self.Tsize:
            #         nu = self.fv0ori * self.fvDef.computeVertexNormals()
            #         v = self.v[kk, ...]
            #         kkm = kk
            #     else:
            #         kkm = kk - 1
            #     if kk > 0:
            #         area_displ[kk,:] = area_displ[kk-1,:] + dt * ((AV + 1) * (v * nu).sum(axis=1))[np.newaxis,:]
            # # #     vf.scalars.append('area_displacement')
            # # #     vf.scalars.append(area_displ[kk,:])
            # # #     vf.scalars.append('constraint')
            # # #     vf.scalars.append(self.cstr[kkm, :])
            # # #     vf.vectors.append('velocity')
            # # #     vf.vectors.append(self.v[kkm, :])
            # # #     vf.vectors.append('normals')
            # # #     vf.vectors.append(self.nu[kkm, :])
            # # #     self.fvDef.saveVTK2(self.outputDir + '/' + self.saveFile + str(kk) + '.vtk', vf)
            # #
            # adisp = area_displ / area_displ[-1, :][np.newaxis,:]
            # fvDef = surfaces.Surface(surf=self.fv0)
            # fvDef.saveVTK(self.outputDir + '/' + self.saveFile +'_bok0' + '.vtk')
            # x = np.zeros((self.npt, self.dim))
            # for kk in range(1, self.Tsize+1):
            #     Inext = ((adisp - float(kk)/(self.Tsize))>-1e-10).argmax(axis=0)
            #     for jj in range(self.npt):
            #         r = (adisp[Inext[jj], jj] - float(kk)/(self.Tsize))/(adisp[Inext[jj], jj] - adisp[Inext[jj]-1, jj])
            #         x[jj] = r*self.xt[Inext[jj]-1,jj,:] + (1-r)*self.xt[Inext[jj],jj,:]
            #     fvDef.updateVertices(x)
            #     fvDef.saveVTK(self.outputDir + '/' + self.saveFile + '_bok' + str(kk) + '.vtk')
        else:
            (obj1, self.state, self.cval) = self.objectiveFunDef(self.control, withJacobian=True)
            self.meanc = np.sqrt((self.cval ** 2).sum() / 2)
            logging.info('mean constraint %f max constraint %f' % (self.meanc, np.fabs(self.cval).max()))
            # logging.info('mean constraint %f max constraint %f' %(np.sqrt((self.cstr**2).sum()/self.cval.size), np.fabs(self.cstr).max()))
            self.fvDef.updateVertices(np.squeeze(self.state['xt'][-1, :, :]))
            self.fvInit.updateVertices(self.control['x0'])
            #self.testConstraintTerm(self.xt, self.at, self.Afft)
        if self.pplot:
            fig = plt.figure(4)
            # fig.clf()
            ax = Axes3D(fig)
            lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
            lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
            ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
            ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
            ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
            fig.canvas.flush_events()
        self.options['KparDiff'].pk_dtype = self.Kdiff_dtype
        self.options['KparDist'].pk_dtype = self.Kdist_dtype

    def optimizeMatching(self):
        self.coeffZ = 10.
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = np.sqrt(grd2) / 10000
        self.coeffAff = self.coeffAff1
        self.muEps = 1.0
        it = 0
        while (self.muEps > 0.001) & (it < self.options['maxIter_al']):
            logging.info('Starting Minimization: Iteration = %d gradEps = %f muEps = %f mu = %f' % (it, self.gradEps, self.muEps, self.mu))
            # self.coeffZ = max(1.0, self.mu)
            if self.options['algorithm'] == 'bfgs':
                bfgs.bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter_grad'],
                          TestGradient=self.options['testGradient'], epsInit=1.,
                          lineSearch=self.options['lineSearch'], memory=25)
            else:
                cg.cg(self, verb=self.options['verb'], maxIter=self.options['maxIter_grad'],
                      TestGradient=self.options['testGradient'], epsInit=0.01,
                      lineSearch=self.options['lineSearch'])
            self.coeffAff = self.coeffAff2
            for t in range(self.Tsize + 1):
                self.lmb[t, ...] = -self.cval[t, ...] / self.mu
            logging.info('mean lambdas %f' % (np.fabs(self.lmb).sum() / self.lmb.size))
            if self.converged:
                self.gradEps *= .5
                if it > 10 and self.meanc > self.muEps:
                    self.mu *= 0.75
                #            else:
                #                self.mu *= 0.9
            if self.muEps > self.meanc:
                self.muEps = min(0.75*self.muEps, 0.9 * self.meanc)
            self.obj = None
            self.reset = True
            it = it + 1

            # return self.fvDef

