import numpy as np
from numba import jit, prange
from . import gaussianDiffeons as gd
import numpy.linalg as LA
from . import affineBasis


##### First-order evolution

# Solves dx/dt = K(x,x) a(t) + A(t) x + b(t) with x(0) = x0
# affine: affine component [A, b] or None
# if withJacobian =True: return Jacobian determinant
# if withNormal = nu0, returns nu(t) evolving as dnu/dt = -Dv^{T} nu
# def landmarkDirectEvolutionEuler_py(x0, at, KparDiff, affine = None, withJacobian=False, withNormals=None, withPointSet=None):
#     N = x0.shape[0]
#     dim = x0.shape[-1]
#     M = at.shape[0] + 1
#     timeStep = 1.0/(M-1)
#     xt = np.zeros([M, N, dim])
#     xt[0, ...] = x0
#     simpleOutput = True
#     if not (withNormals is None):
#         simpleOutput = False
#         nt = np.zeros([M, N, dim])
#         nt[0, ...] = withNormals
#     if not(affine is None):
#         A = affine[0]
#         b = affine[1]
#     if not (withPointSet is None):
#         simpleOutput = False
#         K = withPointSet.shape[0]
#         yt = np.zeros([M,K,dim])
#         yt[0,...] = withPointSet
#         if withJacobian:
#             simpleOutput = False
#             Jt = np.zeros([M, K])
#     else:
#         if withJacobian:
#             simpleOutput = False
#             Jt = np.zeros([M, N])
#
#     for k in range(M-1):
#         z = np.squeeze(xt[k, ...])
#         a = np.squeeze(at[k, ...])
#         if not(affine is None):
#             Rk = affineBasis.getExponential(timeStep * A[k])
#             xt[k+1, ...] = np.dot(z, Rk.T) + timeStep * (KparDiff.applyK(z, a) + b[k])
#         else:
#             xt[k+1, ...] = z + timeStep * KparDiff.applyK(z, a)
#         # if not (affine is None):
#         #     xt[k+1, :, :] += timeStep * (np.dot(z, A[k].T) + b[k])
#         if not (withPointSet is None):
#             zy = np.squeeze(yt[k, :, :])
#             if not(affine is None):
#                 yt[k+1, ...] = np.dot(zy, Rk.T) + timeStep * (KparDiff.applyK(z, a, firstVar=zy) + b[k])
#             else:
#                 yt[k+1, :, :] = zy + timeStep * KparDiff.applyK(z, a, firstVar=zy)
#             # if not (affine is None):
#             #     yt[k+1, :, :] += timeStep * (np.dot(zy, A[k].T) + b[k])
#             if withJacobian:
#                 Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a, firstVar=zy)
#                 if not (affine is None):
#                     Jt[k+1, :] += timeStep * (np.trace(A[k]))
#         else:
#             if withJacobian:
#                 Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a)
#                 if not (affine is None):
#                     Jt[k+1, :] += timeStep * (np.trace(A[k]))
#
#         if not (withNormals is None):
#             zn = np.squeeze(nt[k, :, :])
#             nt[k+1, :, :] = zn - timeStep * KparDiff.applyDiffKT(z, zn[np.newaxis,...], a[np.newaxis,...])
#             if not (affine is None):
#                 nt[k+1, :, :] -= timeStep * np.dot(zn, A[k])
#     if simpleOutput:
#         return xt
#     else:
#         output = [xt]
#         if not (withPointSet is None):
#             output.append(yt)
#         if not (withNormals is None):
#             output.append(nt)
#         if withJacobian:
#             output.append(Jt)
#         return output
#

def landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=None,
                                 withJacobian=False, withNormals=None, withPointSet=None):
    if not (affine is None or len(affine[0]) == 0):
        withaff = True
        A = affine[0]
        b = affine[1]
    else:
        withaff = False
        A = np.zeros((1,1,1)) #np.zeros((T,dim,dim))
        b = np.zeros((1,1)) #np.zeros((T,dim))

    N = x0.shape[0]
    dim = x0.shape[1]
    T = at.shape[0]
    timeStep = 1.0 / (T)
    xt = np.zeros((T+1, N, dim))
    xt[0, :,:] = x0

    Jt = np.zeros((T + 1, N, 1))
    simpleOutput = True
    if withPointSet is not None:
        simpleOutput = False
        K = withPointSet.shape[0]
        y0 = withPointSet
        yt = np.zeros((T + 1, K, dim))
        yt[0, :,:] = y0
        if withJacobian:
            simpleOutput = False

    if withNormals is not None:
        simpleOutput = False
        nt = np.zeros((T+1, N, dim))
        nt[0, :,:] = withNormals

    if withJacobian:
        simpleOutput = False

    for t in range(T):
        if withaff:
            Rk = affineBasis.getExponential(timeStep * A[t,:,:])
            xt[t+1,:,:] = xt[t,:,:] @ Rk.T + timeStep * b[t,None,:]
        else:
            xt[t+1, :,:] = xt[t, :,:]
        xt[t+1,:,:] += timeStep*KparDiff.applyK(xt[t,:,:], at[t,:,:])

        if withPointSet is not None:
            if withaff:
                yt[t+1,:,:] = yt[t, :,:] @ Rk.T + timeStep * b[t,None, :]
            else:
                yt[t+1,:,:] = yt[t, :,:]
            yt[t + 1, :,:] += timeStep * KparDiff.applyK(xt[t, :,:], at[t, :,:], firstVar=yt[t, :,:])

        if withJacobian:
            Jt[t+1,:,:] = Jt[t,:,:] + timeStep * KparDiff.applyDivergence(xt[t,:,:], at[t,:,:])
            if withaff:
                Jt[t+1, :,:] += timeStep * (np.trace(A[t]))

        if withNormals is not None:
            nt[t+1, :,:] = nt[t, :,:] - timeStep * KparDiff.applyDiffKT(xt[t,:,:], nt[t, :, :], at[t, :, :])
            if withaff:
                nt[t + 1, :, :] -= timeStep * (nt[t, :, :] @ A[t])

    if simpleOutput:
        return xt
    else:
        output = [xt]
        if not (withPointSet is None):
            output.append(yt)
        if not (withNormals is None):
            output.append(nt)
        if withJacobian: #not (Jt is None):
            output.append(Jt)
        return output



def landmarkHamiltonianCovector(x0, at, px1, Kpardiff, regweight, affine=None, extraTerm = None):
    if not (affine is None or len(affine[0]) == 0):
        A = affine[0]
    else:
        A = None

    N = x0.shape[0]
    dim = x0.shape[1]
    M = at.shape[0]
    timeStep = 1.0 / (M)

    xt = landmarkDirectEvolutionEuler(x0, at, Kpardiff, affine=affine)

    pxt = np.zeros((M + 1, N, dim))
    pxt[M, :, :] = px1

    for t in range(M):
        px = pxt[M - t, :, :]
        z = xt[M - t - 1, :, :]
        a = at[M - t - 1, :, :]
        if extraTerm is not None:
            grd = extraTerm['grad'](z, Kpardiff.applyK(z,a))
            Lv = -extraTerm['coeff'] * grd[0]
            DLv = extraTerm['coeff'] * grd[1]
            zpx = Kpardiff.applyDiffKT(z, px, a, regweight=regweight, lddmm=True,
                                       extra_term=Lv) - DLv
        else:
            zpx = Kpardiff.applyDiffKT(z, px, a, regweight=regweight, lddmm=True)

        if affine is not None:
            pxt[M - t - 1, :, :] = px @ affineBasis.getExponential(timeStep * A[M - t - 1, :, :]) + timeStep * zpx
        else:
            pxt[M - t - 1, :, :] = px + timeStep * zpx
    return pxt, xt


# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def landmarkHamiltonianGradient(x0, at, px1, KparDiff, regweight, getCovector = False, affine = None, extraTerm = None):
    (pxt, xt) = landmarkHamiltonianCovector(x0, at, px1, KparDiff, regweight, affine=affine, extraTerm=extraTerm)
    dat = np.zeros(at.shape)
    timeStep = 1.0/at.shape[0]
    if affine is not None:
        A = affine[0]
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for k in range(at.shape[0]):
        a = at[k, :, :]
        x = xt[k, :, :]
        px = pxt[k+1, :, :]
        #print 'testgr', (2*a-px).sum()
        dat[k, :, :] = (2*regweight*a-px)
        if extraTerm is not None:
            Lv = extraTerm['grad'](x, KparDiff.applyK(x,a), variables='phi')
            #Lv = -foo.laplacian(v)
            dat[k, :, :] += extraTerm['coeff'] * Lv
        if not (affine is None):
            dA[k] = affineBasis.gradExponential(A[k] * timeStep, px, x) #.reshape([self.dim**2, 1])/timeStep
            db[k] = px.sum(axis=0) #.reshape([self.dim,1])

    if affine is None:
        if getCovector == False:
            return dat, xt
        else:
            return dat, xt, pxt
    else:
        if getCovector == False:
            return dat, dA, db, xt
        else:
            return dat, dA, db, xt, pxt


########### Semi-reduced equations: ct are control points in the evolution

def landmarkSemiReducedEvolutionEuler(x0, ct, at, KparDiff, affine=None,
                                 withJacobian=False, withNormals=None, withPointSet=None):
    if not (affine is None or len(affine[0]) == 0):
        withaff = True
        A = affine[0]
        b = affine[1]
    else:
        withaff = False
        A = np.zeros((1,1,1)) #np.zeros((T,dim,dim))
        b = np.zeros((1,1)) #np.zeros((T,dim))

    N = x0.shape[0]
    dim = x0.shape[1]
    T = at.shape[0]
    timeStep = 1.0 / (T)
    xt = np.zeros((T+1, N, dim))
    xt[0, :,:] = x0

    Jt = np.zeros((T + 1, N, 1))
    simpleOutput = True
    if withPointSet is not None:
        simpleOutput = False
        K = withPointSet.shape[0]
        y0 = withPointSet
        yt = np.zeros((T + 1, K, dim))
        yt[0, :, :] = y0
        if withJacobian:
            simpleOutput = False

    if withNormals is not None:
        simpleOutput = False
        nt = np.zeros((T+1, N, dim))
        nt[0, :, :] = withNormals

    if withJacobian:
        simpleOutput = False

    for t in range(T):
        if withaff:
            Rk = affineBasis.getExponential(timeStep * A[t,:,:])
            xt[t+1, :, :] = np.dot(xt[t,:,:], Rk.T) + timeStep * b[t,None, :]
        else:
            xt[t+1, :,:] = xt[t, :,:]
        xt[t+1,:,:] += timeStep*KparDiff.applyK(ct[t,:,:], at[t,:,:], firstVar=xt[t,:,:])

        if withPointSet is not None:
            if withaff:
                yt[t+1,:,:] = np.dot(yt[t, :,:], Rk.T) + timeStep * b[t, None, :]
            else:
                yt[t+1,:,:] = yt[t, :,:]
            yt[t + 1, :,:] += timeStep * KparDiff.applyK(ct[t, :,:], at[t, :,:], firstVar=yt[t, :,:])

        if withJacobian:
            Jt[t+1,:,:] = Jt[t,:,:] + timeStep * KparDiff.applyDivergence(ct[t,:,:], at[t,:,:])
            if withaff:
                Jt[t+1, :,:] += timeStep * (np.trace(A[t]))

        if withNormals is not None:
            nt[t+1, :,:] = nt[t, :,:] - timeStep * KparDiff.applyDiffKT(ct[t,:,:], nt[t, :, :], at[t, :, :])
            if withaff:
                nt[t + 1, :, :] -= timeStep * np.dot(nt[t, :, :], A[t])

    if simpleOutput:
        return xt
    else:
        output = [xt]
        if not (withPointSet is None):
            output.append(yt)
        if not (withNormals is None):
            output.append(nt)
        if withJacobian: #not (Jt is None):
            output.append(Jt)
        return output


def landmarkSemiReducedHamiltonianCovector(x0, ct, at, px1, Kpardiff, affine=None, forwardTraj = None,
                                           stateSubset = None, controlSubset = None, stateProb = 1.,
                                           controlProb = 1., weightSubset = 1.):
    if not (affine is None or len(affine[0]) == 0):
        A = affine[0]
    else:
        A = np.zeros((1, 1, 1))

    if np.isscalar(stateProb):
        stateProb = stateProb * np.ones(x0.shape[0])

    N = x0.shape[0]
    dim = x0.shape[1]
    T = at.shape[0]
    timeStep = 1.0 / T

    if stateSubset is not None:
        x0_ = x0[stateSubset]
    else:
        stateSubset = np.arange(N)
        x0_ = x0

    J = np.intersect1d(stateSubset, controlSubset)

    if forwardTraj is None:
        xt = landmarkSemiReducedEvolutionEuler(x0_, ct, at, Kpardiff, affine=affine)
    else:
        xt = forwardTraj

    pxt = np.zeros((T, N, dim))
    pxt[T-1, stateSubset, :] = px1

    for t in range(1,T):
        px = np.squeeze(pxt[T - t, stateSubset, :])
        z = np.squeeze(xt[T - t, :, :])
        c = np.squeeze(ct[T - t, :, :])
        a = np.squeeze(at[T - t, :, :])
        zpx = np.zeros((N, dim))
        zpx[stateSubset, :] = Kpardiff.applyDiffKT(c, px, a, firstVar=z)
        zpx[stateSubset, :] -= 2* (weightSubset/stateProb[stateSubset, None]) * z
        zpx[J, :] += 2*(weightSubset / (stateProb[J, None]*controlProb)) * c[J, :]
        if not (affine is None):
            pxt[T - t - 1, stateSubset, :] = np.dot(px, affineBasis.getExponential(timeStep * A[T - t, :, :])) \
                                             + timeStep * zpx[stateSubset, :]
        else:
            pxt[T - t - 1, stateSubset, :] = px + timeStep * zpx[stateSubset, :]
    return pxt, xt



# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def landmarkSemiReducedHamiltonianGradient(x0, ct, at, px1, KparDiff, regweight, getCovector = False, affine = None,
                                           controlSubset = None, controlProb = 1., stateSubset = None,
                                           stateProb = 1., weightSubset = 1., forwardTraj = None):
    if controlSubset is None:
        controlSubset = np.arange(at.shape[1])
        stateSubset = np.arange(x0.shape[0])

    if np.isscalar(stateProb):
        stateProb = stateProb * np.ones(x0.shape[0])

    (pxt, xt) = landmarkSemiReducedHamiltonianCovector(x0, ct, at, px1, KparDiff, affine=affine,
                                                       controlSubset = controlSubset, stateSubset=stateSubset,
                                                       stateProb=stateProb, controlProb=controlProb,
                                                       weightSubset=weightSubset,
                                                       forwardTraj=forwardTraj)
    #pprob = controlProb * controlProb
    M = at.shape[1]
    pprob = controlProb * (M*controlProb - 1)/(M-1)
    dprob = 1/controlProb - 1/pprob
    I0 = controlSubset
    #I1 = controlSubset[1]
    J = np.intersect1d(I0, stateSubset)

    dat = np.zeros(at.shape)
    dct = np.zeros(at.shape)
    timeStep = 1.0/at.shape[0]
    if not (affine is None):
        A = affine[0]
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for t in range(at.shape[0]):
        a0 = at[t, I0, :]
        #a1 = at[t, I1, :]
        c = ct[t, :, :]
        x = np.zeros(x0.shape)
        x[stateSubset, :] = xt[t, :, :]
        px = pxt[t, stateSubset, :]
        #print 'testgr', (2*a-px).sum()
        dat[t, I0, :] = 2 * regweight*KparDiff.applyK(c[I0,:], a0) / pprob
        dat[t, I0, :] += 2*dprob * a0
        #dat[t, I0, :] += regweight*KparDiff.applyK(c[I1,:], a1, firstVar=c[I0, :]) / pprob
        dat[t, :, :] -= KparDiff.applyK(xt[t, :, :], px, firstVar=c)
        # print(f'px {np.fabs(px).max()} {np.fabs(dat[k,:,:]).max()}')
        if t > 0:
            dct[t, I0, :] = 2*regweight*KparDiff.applyDiffKT(c[I0, :], a0, a0) / pprob
            #dct[t, I1, :] += regweight * KparDiff.applyDiffKT(c[I0, :], a1, a0, firstVar=c[I1, :]) / pprob
            dct[t, :, :] -= KparDiff.applyDiffKT(xt[t, :, :], at[t, :, :], px,  firstVar=c)
            dct[t, controlSubset, :] += 2*weightSubset*c[controlSubset, :]/controlProb
            dct[t, J, :] -= 2 * (weightSubset / (stateProb[J, None]*controlProb)) * x[J, :]

        if not (affine is None):
            dA[t] = affineBasis.gradExponential(A[t] * timeStep, pxt[t, :, :], xt[t, :, :]) #.reshape([self.dim**2, 1])/timeStep
            db[t] = pxt[t, :, :].sum(axis=0) #.reshape([self.dim,1])

    if affine is None:
        if getCovector == False:
            return dct, dat, xt
        else:
            return dct, dat, xt, pxt
    else:
        if getCovector == False:
            return dct, dat, dA, db, xt
        else:
            return dct, dat, dA, db, xt, pxt




################## Time series
def timeSeriesCovector(x0, at, px1, KparDiff, regweight, affine = None, isjump = None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = at.shape[0]
    nTarg = len(px1)
    if isjump is None:
        Tsize1 = M/nTarg
        isjump = np.array(M+1, dtype=bool)
        for k in range(nTarg):
            isjump[(k+1)*Tsize1] = True
    timeStep = 1.0/M
    xt = landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine)
    pxt = np.zeros([M+1, N, dim])
    pxt[M, :, :] = px1[nTarg-1]
    jk = nTarg-2
    if not(affine is None):
        A = affine[0]

    for t in range(M):
        px = np.squeeze(pxt[M-t, :, :])
        z = np.squeeze(xt[M-t-1, :, :])
        a = np.squeeze(at[M-t-1, :, :])
        # dgzz = kfun.kernelMatrix(KparDiff, z, diff=True)
        # if (isfield(KparDiff, 'zs') && size(z, 2) == 3)
        #     z(:,3) = z(:,3) / KparDiff.zs ;
        # end
        a1 = np.concatenate((px[np.newaxis,:,:], a[np.newaxis,:,:], -2*regweight[M-t-1]*a[np.newaxis,:,:]))
        a2 = np.concatenate((a[np.newaxis,:,:], px[np.newaxis,:,:], a[np.newaxis,:,:]))
        #a1 = [px, a, -2*regweight*a]
        #a2 = [a, px, a]
        #print 'test', px.sum()
        zpx = KparDiff.applyDiffKT(z, a1, a2)
        # if not (affine is None):
        #     zpx += np.dot(px, A[M-t-1])
        # pxt[M-t-1, :, :] = px + timeStep * zpx
        if not (affine is None):
            pxt[M-t-1, :, :] = np.dot(px, affineBasis.getExponential(timeStep * A[M - t - 1])) + timeStep * zpx
        else:
            pxt[M-t-1, :, :] = px + timeStep * zpx
        if (t<M-1) and isjump[M-1-t]:
            pxt[M-t-1, :, :] += px1[jk]
            jk -= 1
        #print 'zpx', np.fabs(zpx).sum(), np.fabs(px).sum(), z.sum()
        #print 'pxt', np.fabs((pxt)[M-t-2]).sum()
        
    return pxt, xt

def timeSeriesGradient(x0, at, px1, KparDiff, regweight, getCovector = False, affine = None, isjump=None):
    (pxt, xt) = timeSeriesCovector(x0, at, px1, KparDiff, regweight, affine=affine, isjump=isjump)
    dat = np.zeros(at.shape)
    if not (affine is None):
        A = affine[0]
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for k in range(at.shape[0]):
        a = np.squeeze(at[k, :, :])
        px = np.squeeze(pxt[k+1, :, :])
        #print 'testgr', (2*a-px).sum()
        dat[k, :, :] = (2*regweight[k]*a-px)
        if not (affine is None):
            dA[k] = affineBasis.gradExponential(A[k] / at.shape[0], pxt[k + 1], xt[k]) #.reshape([self.dim**2, 1])/timeStep
            db[k] = pxt[k+1].sum(axis=0)


    if affine is None:
        if getCovector == False:
            return dat, xt
        else:
            return dat, xt, pxt
    else:
        if getCovector == False:
            return dat, dA, db, xt
        else:
            return dat, dA, db, xt, pxt



################  Second-order equations


def landmarkEPDiff(T, x0, a0, KparDiff, affine = None, withJacobian=False, withNormals=None, withPointSet=None):
    N = x0.shape[0]
    dim = x0.shape[1]
    timeStep = 1.0/T
    at = np.zeros([T, N, dim])
    xt = np.zeros([T+1, N, dim])
    xt[0, :, :] = x0
    at[0, :, :] = a0
    simpleOutput = True
    if not (withNormals is None):
        simpleOutput = False
        nt = np.zeros([T+1, N, dim])
        nt[0, :, :] = withNormals
    if withJacobian:
        simpleOutput = False
        Jt = np.zeros([T+1, N])
    if not(affine is None):
        A = affine[0]
        b = affine[1]
    if not (withPointSet is None):
        simpleOutput = False
        K = withPointSet.shape[0]
        yt = np.zeros([T+1,K,dim])
        yt[0, :, :] = withPointSet

    for k in range(T):
        z = np.squeeze(xt[k, :, :])
        a = np.squeeze(at[k, :, :])
        xt[k+1, :, :] = z + timeStep * KparDiff.applyK(z, a)
        #print 'test', px.sum()
        if k < (T-1):
            at[k+1, :, :] = a - timeStep * KparDiff.applyDiffKT(z, a, a)
        if not (affine is None):
            xt[k+1, :, :] += timeStep * (np.dot(z, A[k].T) + b[k])
        if not (withPointSet is None):
            zy = np.squeeze(yt[k, :, :])
            yt[k+1, :, :] = zy + timeStep * KparDiff.applyK(z, a, firstVar=zy)
            if not (affine is None):
                yt[k+1, :, :] += timeStep * (np.dot(zy, A[k].T) + b[k])

        if not (withNormals is None):
            zn = np.squeeze(nt[k, :, :])
            nt[k+1, :, :] = zn - timeStep * KparDiff.applyDiffKT(z, zn, a)
            if not (affine is None):
                nt[k+1, :, :] += timeStep * np.dot(zn, A[k])
        if withJacobian:
            Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a)
            if not (affine is None):
                Jt[k+1, :] += timeStep * (np.trace(A[k]))
    if simpleOutput:
        return xt, at
    else:
        output = [xt, at]
        if not (withPointSet is None):
            output.append(yt)
        if not (withNormals is None):
            output.append(nt)
        if withJacobian:
            output.append(Jt)
        return output





def secondOrderEvolution(x0, a0, rhot, KparDiff, timeStep, withJacobian=False, withPointSet=None, affine=None):
    T = rhot.shape[0]
    N = x0.shape[0]
    #print M, N
    dim = x0.shape[1]
    at = np.zeros([T+1, N, dim])
    xt = np.zeros([T+1, N, dim])
    xt[0, :, :] = x0
    at[0, :, :] = a0
    simpleOutput = True
    if not(affine is None):
        aff_ = True
        A = affine[0]
        b = affine[1]
    else:
        aff_=False
        A = None
        b = None
    if not (withPointSet is None):
        simpleOutput = False
        K = withPointSet.shape[0]
        zt = np.zeros([T+1,K,dim])
        zt[0, :, :] = withPointSet
        if withJacobian:
            simpleOutput = False
            Jt = np.zeros([T+1, K])
    elif withJacobian:
        simpleOutput = False
        Jt = np.zeros([T+1, N])

    for k in range(T):
        x = xt[k, :, :]
        a = at[k, :, :]
        #print 'evolution v:', np.sqrt((v**2).sum(axis=1)).sum()/v.shape[0]
        rho = rhot[k,:,:]
        zx = KparDiff.applyK(x, a)
        za = -KparDiff.applyDiffKT(x, a, a) + rho
        if aff_:
            #U = np.eye(dim) + timeStep * A[k]
            U = affineBasis.getExponential(timeStep * A[k])
            xt[k+1, :, :] = np.dot(x + timeStep * zx, U.T) + timeStep * b[k]
            Ui = LA.inv(U)
            at[k+1, :, :] = np.dot(a + timeStep * za, Ui)
        else:
            xt[k+1, :, :] = x + timeStep * zx  
            at[k+1, :, :] = a + timeStep * za
        if not (withPointSet is None):
            z = zt[k, :, :]
            zx = KparDiff.applyK(x, a, firstVar=z)
            if aff_:
                zt[k+1, :, :] =  np.dot(z + timeStep * zx, U.T) + timeStep * b[k]
            else:
                zt[k+1, :, :] = z + timeStep * zx  
            if withJacobian:
                Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(x, a, firstVar=z).ravel()
                if aff_:
                    Jt[k+1, :] += timeStep * (np.trace(A[k]))
        elif withJacobian:
            Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a).ravel()
            if aff_:
                Jt[k+1, :] += timeStep * (np.trace(A[k]))
    if simpleOutput:
        return xt, at
    else:
        output = [xt, at]
        if not (withPointSet is None):
            output.append(zt)
        if withJacobian:
            output.append(Jt)
        return output



def secondOrderHamiltonian(x, a, rho, px, pa, KparDiff, affine=None):
    Ht = (px * KparDiff.applyK(x, a)).sum() 
    Ht += (pa*(-KparDiff.applyDiffKT(x, a[np.newaxis,:,:], a[np.newaxis,:,:]) + rho)).sum()
    Ht -= (rho**2).sum()/2
    if not(affine is None):
        A = affine[0]
        b = affine[1]
        Ht += (px * (np.dot(x, A.T) + b)).sum() - (pa * np.dot(a, A)).sum()
    return Ht

    
def secondOrderCovector(x0, a0, rhot, px1, pa1, KparDiff, timeStep, affine = None, isjump = None):
    T = rhot.shape[0]
    nTarg = len(px1)
    if not(affine is None):
        aff_ = True
        A = affine[0]
    else:
        aff_ = False
        
    if isjump is None:
        isjump = np.zeros(T, dtype=bool)
        for t in range(1,T):
            if t%nTarg == 0:
                isjump[t] = True

    N = x0.shape[0]
    dim = x0.shape[1]
    [xt, at] = secondOrderEvolution(x0, a0, rhot, KparDiff, timeStep, affine=affine)
    pxt = np.zeros([T+1, N, dim])
    pxt[T, :, :] = px1[nTarg-1]
    pat = np.zeros([T+1, N, dim])
    pat[T, :, :] = pa1[nTarg-1]
    jIndex = nTarg - 2
    for t in range(T):
        px = np.squeeze(pxt[T-t, :, :])
        pa = np.squeeze(pat[T-t, :, :])
        x = np.squeeze(xt[T-t-1, :, :])
        a = np.squeeze(at[T-t-1, :, :])

        if aff_:
            U = affineBasis.getExponential(timeStep * A[T - t - 1])
            px_ = np.dot(px, U)
            Ui = LA.inv(U)
            pa_ = np.dot(pa,Ui.T)
        else:
            px_ = px
            pa_ = pa

        a1 = np.concatenate((px_[np.newaxis,:,:], a[np.newaxis,:,:]))
        a2 = np.concatenate((a[np.newaxis,:,:], px_[np.newaxis,:,:]))
        zpx = KparDiff.applyDiffKT(x, a1, a2) - KparDiff.applyDDiffK11and12(x, a, a, pa_)
        zpa = KparDiff.applyK(x, px_) - KparDiff.applyDiffK1and2(x, pa_, a)
        pxt[T-t-1, :, :] = px_ + timeStep * zpx
        pat[T-t-1, :, :] = pa_ + timeStep * zpa
        if isjump[T-t-1]:
            pxt[T-t-1, :, :] += px1[jIndex]
            pat[T-t-1, :, :] += pa1[jIndex]
            jIndex -= 1

    return pxt, pat, xt, at

# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def secondOrderGradient(x0, a0, rhot, px1, pa1, KparDiff, timeStep, isjump = None, getCovector = False, affine=None, controlWeight=1.0):
    (pxt, pat, xt, at) = secondOrderCovector(x0, a0, rhot, px1, pa1, KparDiff, timeStep, isjump=isjump,affine=affine)
    if not (affine is None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    Tsize = rhot.shape[0]
    timeStep = 1.0/Tsize
    drhot = np.zeros(rhot.shape)
    if not (affine is None):
        for k in range(Tsize):
            x = np.squeeze(xt[k, :, :])
            a = np.squeeze(at[k, :, :])
            rho = np.squeeze(rhot[k, :, :])
            px = np.squeeze(pxt[k+1, :, :])
            pa = np.squeeze(pat[k+1, :, :])
            zx = x + timeStep * KparDiff.applyK(x, a)
            za = a + timeStep * (-KparDiff.applyDiffKT(x, a[np.newaxis,:,:], a[np.newaxis,:,:]) + rho)
            U = affineBasis.getExponential(timeStep * affine[0][k])
            #U = np.eye(dim) + timeStep * affine[0][k]
            Ui = LA.inv(U)
            pa = np.dot(pa, Ui.T)
            za = np.dot(za, Ui)
#            dA[k,:,:] =  ((px[:,:,np.newaxis]*zx[:,np.newaxis,:]).sum(axis=0)
#                            - (za[:,:,np.newaxis]*pa[:,np.newaxis,:]).sum(axis=0))
            dA[k,...] =  (affineBasis.gradExponential(timeStep * affine[0][k], px, zx)
                          - affineBasis.gradExponential(timeStep * affine[0][k], za, pa))
            drhot[k,...] = rho*controlWeight - pa
        db = pxt[1:Tsize+1,...].sum(axis=1)
        # for k in range(rhot.shape[0]):
        #     #np.dot(pxt[k+1].T, xt[k]) - np.dot(at[k].T, pat[k+1])
        #     #dA[k] = -np.dot(pat[k+1].T, at[k]) + np.dot(xt[k].T, pxt[k+1])
        #     db[k] = pxt[k+1].sum(axis=0)

    #drhot = rhot*controlWeight - pat[1:pat.shape[0],...]
    da0 = KparDiff.applyK(x0, a0) - pat[0,...]

    if affine is None:
        if getCovector == False:
            return da0, drhot, xt, at
        else:
            return da0, drhot, xt, at, pxt, pat
    else:
        if getCovector == False:
            return da0, drhot, dA, db, xt, at
        else:
            return da0, drhot, dA, db, xt, at, pxt, pat

        

def secondOrderFiberEvolution(x0, a0, y0, v0, rhot, KparDiff, withJacobian=False, withPointSet=None):
    T = rhot.shape[0]
    N = x0.shape[0]
    M = y0.shape[0]
    #print M, N
    dim = x0.shape[1]
    timeStep = 1.0/T
    at = np.zeros([T+1, M, dim])
    yt = np.zeros([T+1, M, dim])
    vt = np.zeros([T+1, M, dim])
    xt = np.zeros([T+1, N, dim])
    xt[0, :, :] = x0
    at[0, :, :] = a0
    yt[0, :, :] = y0
    vt[0, :, :] = v0
    simpleOutput = True
    if not (withPointSet is None):
        simpleOutput = False
        K = withPointSet.shape[0]
        zt = np.zeros([T+1,K,dim])
        zt[0, :, :] = withPointSet
        if withJacobian:
            simpleOutput = False
            Jt = np.zeros([T+1, K])
    elif withJacobian:
        simpleOutput = False
        Jt = np.zeros([T+1, M])

    for k in range(T):
        x = np.squeeze(xt[k, :, :])
        y = np.squeeze(yt[k, :, :])
        a = np.squeeze(at[k, :, :])
        v = np.squeeze(vt[k, :, :])
        #print 'evolution v:', np.sqrt((v**2).sum(axis=1)).sum()/v.shape[0]
        rho = np.squeeze(rhot[k,:])
        xt[k+1, :, :] = x + timeStep * KparDiff.applyK(y, a, firstVar=x) 
        yt[k+1, :, :] = y + timeStep * KparDiff.applyK(y, a)
        KparDiff.hold()
        at[k+1, :, :] = a + timeStep * (-KparDiff.applyDiffKT(y, a[np.newaxis,...], a[np.newaxis,...]) + rho[:,np.newaxis] * v) 
        vt[k+1, :, :] = v + timeStep * KparDiff.applyDiffK(y, v, a) 
        KparDiff.release()
        if not (withPointSet is None):
            z = np.squeeze(zt[k, :, :])
            zt[k+1, :, :] = z + timeStep * KparDiff.applyK(y, a, firstVar=z)
            if withJacobian:
                Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(y, a, firstVar=z)
        elif withJacobian:
            Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(y, a)
    if simpleOutput:
        return xt, at, yt, vt
    else:
        output = [xt, at, yt, vt]
        if not (withPointSet is None):
            output.append(zt)
        if withJacobian:
            output.append(Jt)
        return output
    
def secondOrderFiberHamiltonian(x, a, y, v, rho, px, pa, py, pv, KparDiff):

    Ht = ( (px * KparDiff.applyK(y, a, firstVar=x)).sum() 
           + (py*KparDiff.applyK(y, a)).sum())
    KparDiff.hold()
    Ht += ( (pa*(-KparDiff.applyDiffKT(y, a[np.newaxis,...], a[np.newaxis,...]) + rho[:,np.newaxis] * v)).sum()
            + (pv*KparDiff.applyDiffK(y, v, a)).sum()) 
    KparDiff.release()
    Ht -= (rho**2 * (v**2).sum(axis=1)).sum()/2
    return Ht

    
def secondOrderFiberCovector(x0, a0, y0, v0, rhot, px1, pa1, py1, pv1, KparDiff, times= None):
    T = rhot.shape[0]
    nTarg = len(px1)
    Tsize1 = T/nTarg
    if times is None:
        t1 = (float(T)/nTarg) * (range(nTarg)+1)
    N = x0.shape[0]
    M = y0.shape[0]
    dim = x0.shape[1]
    timeStep = 1.0/T
    [xt, at, yt, vt] = secondOrderFiberEvolution(x0, a0, y0, v0, rhot, KparDiff)
    pxt = np.zeros([T, N, dim])
    pxt[T-1, :, :] = px1[nTarg-1]
    pat = np.zeros([T, M, dim])
    pat[T-1, :, :] = pa1[nTarg-1]
    pyt = np.zeros([T, M, dim])
    pyt[T-1, :, :] = py1[nTarg-1]
    pvt = np.zeros([T, M, dim])
    pvt[T-1, :, :] = pv1[nTarg-1]
    for t in range(T-1):
        px = np.squeeze(pxt[T-t-1, :, :])
        pa = np.squeeze(pat[T-t-1, :, :])
        py = np.squeeze(pyt[T-t-1, :, :])
        pv = np.squeeze(pvt[T-t-1, :, :])
        x = np.squeeze(xt[T-t-1, :, :])
        a = np.squeeze(at[T-t-1, :, :])
        y = np.squeeze(yt[T-t-1, :, :])
        v = np.squeeze(vt[T-t-1, :, :])
        rho = np.squeeze(rhot[T-t-1, :])

        zpx = KparDiff.applyDiffKT(y, px[np.newaxis,...], a[np.newaxis,...], firstVar=x)

        zpa = KparDiff.applyK(x, px, firstVar=y)
        KparDiff.hold()
        #print 'zpa1', zpa.sum()
        zpy = KparDiff.applyDiffKT(x, a[np.newaxis,...], px[np.newaxis,...], firstVar=y)
        KparDiff.release()

        a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...]))
        a2 = np.concatenate((a[np.newaxis,...], py[np.newaxis,...]))
        zpy += KparDiff.applyDiffKT(y, a1, a2)
        KparDiff.hold()
        zpy += KparDiff.applyDDiffK11(y, pv, a, v) + KparDiff.applyDDiffK12(y, a, pv, v)
        zpy -= KparDiff.applyDDiffK11(y, a, a, pa) + KparDiff.applyDDiffK12(y, a, a, pa)

        zpv = KparDiff.applyDiffKT(y, pv[np.newaxis,...], a[np.newaxis,...]) + rho[:,np.newaxis]*pa - (rho[:,np.newaxis]**2)*v
        zpa += (KparDiff.applyK(y, py) + KparDiff.applyDiffK2(y, v, pv)
               - KparDiff.applyDiffK(y, pa, a) - KparDiff.applyDiffK2(y, pa, a))
        KparDiff.release()

        pxt[T-t-2, :, :] = px + timeStep * zpx
        pat[T-t-2, :, :] = pa + timeStep * zpa
        pyt[T-t-2, :, :] = py + timeStep * zpy
        pvt[T-t-2, :, :] = pv + timeStep * zpv
        if (t<T-1) and ((T-t-1)%Tsize1 == 0):
#            print T-t-1, (T-t-1)/Tsize1
            pxt[T-t-2, :, :] += px1[(T-t-1)/Tsize1 - 1]
            pat[T-t-2, :, :] += pa1[(T-t-1)/Tsize1 - 1]
            pyt[T-t-2, :, :] += py1[(T-t-1)/Tsize1 - 1]
            pvt[T-t-2, :, :] += pv1[(T-t-1)/Tsize1 - 1]

    return pxt, pat, pyt, pvt, xt, at, yt, vt

# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def secondOrderFiberGradient(x0, a0, y0, v0, rhot, px1, pa1, py1, pv1, KparDiff, times = None, getCovector = False):
    (pxt, pat, pyt, pvt, xt, at, yt, vt) = secondOrderFiberCovector(x0, a0, y0, v0, rhot, px1, pa1, py1, pv1, KparDiff, times=times)
    drhot = np.zeros(rhot.shape)
    for k in range(rhot.shape[0]):
        rho = np.squeeze(rhot[k, :])
        pa = np.squeeze(pat[k, :, :])
        v = np.squeeze(vt[k, :, :])
        drhot[k, :] = rho - (pa*v).sum(axis=1)/(v**2).sum(axis=1)
    if getCovector == False:
        return drhot, xt, at, yt, vt
    else:
        return drhot, xt, at, yt, vt, pxt, pat, pyt, pvt



def landmarkParallelTransport(T, x0, b0, a0, KparDiff):
    timeStep = np.floor(1/T)
    xt, at = landmarkEPDiff(T, x0, a0, KparDiff)
    bt = np.zeros(at.shape)
    bt[0, :, :] = b0
    for t in range(at.shape[0]-1):
        x = xt[t, :, :]
        a = at[t, :, :]
        b = bt[t, :, :]
        xi = KparDiff.applyK(a)
        eta = KparDiff.applyK(b)
        z = KparDiff.applyDiffKT(x, a, b)
        z0 = KparDiff.applyDiffK(x, b, xi) - KparDiff.applyDiffK(x, a, eta)
        z += KparDiff.solve(x, z0)
        bt[t+1, :, :] = b - timeStep * z/2

    return bt