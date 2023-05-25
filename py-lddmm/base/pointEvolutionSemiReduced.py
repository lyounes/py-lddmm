import numpy as np
import numpy.linalg as LA
from . import affineBasis

########### Semi-reduced equations: ct are control points in the evolution

def landmarkSemiReducedEvolutionEuler(x0, ct, at, KparDiff, fidelityWeight = 0.0, affine=None,
                                 withJacobian=False, withNormals=None, withPointSet=None):
    if ct.shape[1] != x0.shape[0]:
        fidelityTerm = False
    else:
        fidelityTerm = True
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
        dx = KparDiff.applyK(ct[t,:,:], at[t,:,:], firstVar=xt[t,:,:])
        if fidelityTerm:
            dx -= fidelityWeight * (xt[t, :, :] - ct[t, :, :])
        xt[t+1,:,:] += timeStep * dx

        if withPointSet is not None:
            if withaff:
                yt[t+1,:,:] = np.dot(yt[t, :,:], Rk.T) + timeStep * b[t, None, :]
            else:
                yt[t+1,:,:] = yt[t, :,:]

            dy = KparDiff.applyK(ct[t, :, :], at[t, :, :], firstVar=yt[t, :, :])
            yt[t + 1, :,:] += timeStep * dy

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


def landmarkSemiReducedHamiltonianCovector(x0, ct, at, px1, Kpardiff, fidelityWeight = 0.,
                                           affine=None, forwardTraj = None,
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
        xt = landmarkSemiReducedEvolutionEuler(x0_, ct, at, Kpardiff, affine=affine, fidelityWeight=fidelityWeight)
    else:
        xt = forwardTraj

    if ct.shape[1] != x0_.shape[0]:
        fidelityTerm = False
    else:
        fidelityTerm = True

    pxt = np.zeros((T, N, dim))
    pxt[T-1, stateSubset, :] = px1

    for t in range(1,T):
        px = pxt[T - t, stateSubset, :]
        z = xt[T - t, :, :]
        c = ct[T - t, :, :]
        a = at[T - t, :, :]
        zpx = np.zeros((N, dim))
        zpx[stateSubset, :] = Kpardiff.applyDiffKT(c, px, a, firstVar=z)
        zpx[stateSubset, :] -= 2* (weightSubset/stateProb[stateSubset, None]) * z
        zpx[J, :] += 2*(weightSubset / (stateProb[J, None]*controlProb)) * c[J, :]
        if fidelityTerm:
            zpx[stateSubset, :] -= fidelityWeight * px
        if not (affine is None):
            pxt[T - t - 1, stateSubset, :] = np.dot(px, affineBasis.getExponential(timeStep * A[T - t, :, :])) \
                                             + timeStep * zpx[stateSubset, :]
        else:
            pxt[T - t - 1, stateSubset, :] = px + timeStep * zpx[stateSubset, :]
    return pxt, xt



# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def landmarkSemiReducedHamiltonianGradient(x0, ct, at, px1, KparDiff, regweight, getCovector = False, affine = None,
                                           fidelityWeight = 0.,
                                           controlSubset = None, controlProb = 1., stateSubset = None,
                                           stateProb = 1., weightSubset = 1., forwardTraj = None):
    if controlSubset is None:
        controlSubset = np.arange(at.shape[1])
        stateSubset = np.arange(x0.shape[0])

    if np.isscalar(stateProb):
        stateProb = stateProb * np.ones(x0.shape[0])

    (pxt, xt) = landmarkSemiReducedHamiltonianCovector(x0, ct, at, px1, KparDiff, affine=affine,
                                                       fidelityWeight = fidelityWeight,
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



