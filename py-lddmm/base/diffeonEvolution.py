import numpy as np
from numba import jit, prange
from . import gaussianDiffeons as gd
import numpy.linalg as LA
from . import affineBasis

@jit(nopython=True, parallel= True)
def zx_(x, c, R, a):
    x_ = x.reshape((np.prod(np.array(x.shape[:-1])), x.shape[-1]))
    dim = x.shape[-1]
    zx = np.zeros(x_.shape)
    Div = np.zeros(x_.shape[0])
    for k in prange(x_.shape[0]):
        for l in range(c.shape[0]):
            dx = x_[k,:] - c[l,:]
            #betax = R[l,:,:]@dx
            betax = np.zeros(dim)
            for kk in range(dim):
                betax[kk] = (R[l, kk, :] * dx).sum()
            dst = (dx * betax).sum()
            fx = np.exp(-dst/2)
            zx[k, :] += fx *a[l,:]
            Div[k] += -fx * (betax * a[l,:]).sum()

    zx = np.reshape(zx, x.shape)
    Div = np.reshape(Div, x.shape[:-1])
    return zx, Div



##################  Diffeons

def gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine=None, withJacobian=False, withPointSet=None,
                                   withNormals=None, withDiffeonSet=None):
    dim = c0.shape[1]
    M = c0.shape[0]
    T = at.shape[0]
    timeStep = 1.0 / T
    ct = np.zeros([T+1, M, dim])
    St = np.zeros([T+1, M, dim, dim])
    ct[0, :, :] = c0
    St[0, :, :, :] = S0
    simpleOutput = True
    sig2 = sigma * sigma

    if (withPointSet is None) and (withDiffeonSet is None):
        withJacobian = False
        withNormals = None

    if withPointSet is not None:
        simpleOutput = False
        x0 = withPointSet
        xt = np.zeros([T+1] + list(x0.shape))
        xt[0, ...] = x0
        if type(withJacobian) == bool:
            if withJacobian:
                Jt = np.zeros(np.insert(x0.shape[0:-1], 0, T+1))
        elif withJacobian is not None:
            # print withJacobian
            J0 = withJacobian
            Jt = np.zeros(np.insert(J0.shape, 0, T+1))
            Jt[0, ...] = J0
            withJacobian = True
        else:
            withJacobian = False
        withPointSet = True

        if not (withNormals is None):
            b0 = withNormals
            bt = np.zeros(np.concatenate([[T], b0.shape]))
            bt[0, ...] = b0

    if not (affine is None):
        A = affine[0]
        b = affine[1]

    if not (withDiffeonSet is None):
        simpleOutput = False
        K = withDiffeonSet[0].shape[0]
        yt = np.zeros([T+1, K, dim])
        Ut = np.zeros([T+1, K, dim, dim])
        yt[0, :, :] = withDiffeonSet[0]
        Ut[0, :, :] = withDiffeonSet[1]
        if withJacobian:
            Jt = np.zeros([T+1, K])
        if not (withNormals is None):
            b0 = withNormals
            bt = np.zeros([T+1, K, dim])
            bt[0, :, :] = b0
            withNormals = True
        withDiffeonSet = True

    sigEye = sig2 * np.eye(dim)
    for t in range(T):
        c = ct[t, :, :]
        S = St[t, :, :, :]
        a = at[t, :, :]

        (R, detR) = gd.multiMatInverse1(sigEye.reshape([1, dim, dim]) + S, isSym=True)

        diff = c[:, None, :] - c[None, :, :]
        betac = (R[None, :, :, :] * diff[:, :, None, :]).sum(axis=3)
        # betac = (R.reshape([1, M, dim, dim]) * diff.reshape([M, M, 1, dim])).sum(axis=3)
        dst = (diff * betac).sum(axis=2)
        fc = np.exp(-dst / 2)
        zc = fc @ a

        Dv = -((fc[:, :, None] * betac)[:, :, None, :] * a[None, :, :, None]).sum(axis=1)
        # Dv = -((fc.reshape([M, M, 1]) * betac).reshape([M, M, 1, dim]) * a.reshape([1, M, dim, 1])).sum(axis=1)
        if not (affine is None):
            Dv = Dv + A[t, None, :, :]
        SDvT = (S[:, :, None, :] * Dv[:, None, :, :]).sum(axis=3)
        # SDvT = (S.reshape([M, dim, 1, dim]) * Dv.reshape([M, 1, dim, dim])).sum(axis=3)
        zS = SDvT.transpose([0, 2, 1]) + SDvT
        zScorr = Dv @ SDvT
        #(SDvT[:, None, :, :] * Dv[:, :, :, None]).sum(axis=2)
        # zScorr = (SDvT.reshape([M, 1, dim, dim]) * Dv.reshape([M, dim, dim, 1])).sum(axis=2)

        ct[t + 1, :, :] = c + timeStep * zc
        if not (affine is None):
            ct[t + 1, :, :] += timeStep * (c @ A[t].T + b[t])

        St[t + 1, :, :, :] = S + timeStep * zS + (timeStep ** 2) * zScorr
        # St[t+1, :, :, :] = S

        if withPointSet:
            x = xt[t, ...]
            zx, Div = zx_(x, c, R, a)
            xt[t + 1, ...] = x + timeStep * zx
            if not (affine is None):
                xt[t + 1, ...] += timeStep * (x @ A[t].T + b[t])
            if withJacobian:
                # print Jt.shape
                # Div = -(fx * (betax * a).sum(axis=-1)).sum(axis=-1)
                Jt[t + 1, ...] = Jt[t, ...] + timeStep * Div
                if not (affine is None):
                    Jt[t + 1, ...] += timeStep * np.trace(A[t])
            if withNormals:
                diffx = x[..., None, :] - c[None, :, :]
                betax = (R * diffx[..., None, :]).sum(axis=-1)
                dst = (betax * diffx).sum(axis=-1)
                fx = np.exp(-dst / 2)
                # zx = fx @ a
                bb = bt[t, ...]
                zb = ((fx * (bb @ a.T))[..., None] * betax).sum(axis=-2)
                zb -= (fx * (betax * a).sum(axis=-1)).sum(axis=-1)[..., None] * bb
                bt[t + 1, :, :] = bb + timeStep * zb

        # if not(withPointSet is None):
        #     x = np.squeeze(xt[t, ...])
        #     diffx = x.[N, 1, dim]) - c.reshape([1, M, dim])
        #     betax = (R.reshape([1, M, dim, dim])*diffx.reshape([N, M, 1, dim])).sum(axis=3)
        #     dst = (betax * diffx).sum(axis=2)
        #     fx = np.exp(-dst/2)
        #     zx = np.dot(fx, a)
        #     xt[t+1, :, :] = x + timeStep * zx
        #     if not (affine is None):
        #         xt[t+1, :, :] += timeStep * (np.dot(x, A[t].T) + b[t])
        #     if withJacobian:
        #         Div = -(fx * (betax * a.reshape(1,M, dim)).sum(axis=2)).sum(axis=1)
        #         Jt[t+1, :] = Jt[t, :] + timeStep * Div
        #         if not (affine is None):
        #             Jt[t+1, :] += timeStep * (np.trace(A[t]))
        #     if not(withNormals is None):
        #         bb = np.squeeze(bt[t, :, :])
        #         zb = ((fx * np.dot(bb, a.T)).reshape([N,M,1]) * betax).sum(axis=1)
        #         zb -= (fx * (betax * a.reshape([1,M,dim])).sum(axis=2)).sum(axis=1).reshape([N,1]) *bb
        #         bt[t+1,:,:] = bb + timeStep*zb

        if withDiffeonSet:
            y = yt[t, :, :]
            U = Ut[t, :, :, :]
            K = y.shape[0]
            # diffy = y.reshape([K, 1, dim]) - c.reshape([1, M, dim])
            diffy = y[:, None, :] - c[None, :, :]
            betay = (R[None, :, :, :] * diffy[:, :, None, :]).sum(axis=3)
            # betay = (R.reshape([1, M, dim, dim]) * diffy.reshape([K, M, 1, dim])).sum(axis=3)
            dst = (diffy * betay).sum(axis=2)
            fy = np.exp(-dst / 2)
            zy = np.dot(fy, a)
            yt[t + 1, :, :] = y + timeStep * zy
            if not (affine is None):
                yt[t + 1, :, :] += timeStep * (y @ A[t].T + b[t])
            Dvy = -((fy[:, :, None] * betay)[:, :, None, :] * a[None, :, :, None]).sum(axis=1)
            if not (affine is None):
                Dvy = Dvy + A[t, None, :, :]
            UDvT = (U[:, :, None, :] * Dvy[:, None, :, :]).sum(axis=3)
            zU = UDvT.transpose([0, 2, 1]) + UDvT
            zUcorr = (UDvT[:, None, :, :] * Dvy[:, :, :, None]).sum(axis=2)
            Ut[t + 1, :, :, :] = U + timeStep * zU + (timeStep ** 2) * zUcorr
            if withJacobian:
                Div = -(fy * (betay * a[None, :, :]).sum(axis=2)).sum(axis=1)
                Jt[t + 1, :] = Jt[t, :] + timeStep * Div
                if not (affine is None):
                    Jt[t + 1, :] += timeStep * (np.trace(A[t]))
            if withNormals:
                bb = bt[t, :, :]
                zb = ((fy * (bb @ a.T))[:, :, None] * betay).sum(axis=1)
                zb -= (fy * (betay * a[None, :, :]).sum(axis=2)).sum(axis=1)[:, None] * bb
                bt[t + 1, :, :] = bb + timeStep * zb

    if simpleOutput:
        return ct, St
    else:
        output = [ct, St]
        if withNormals:
            output.append(bt)
        if withPointSet:
            output.append(xt)
            if withJacobian:
                output.append(Jt)
        if withDiffeonSet:
            output.append(yt)
            output.append(Ut)
            if withJacobian:
                output.append(Jt)
        # if not (withNormals is None):
        #     output.append(nt)
        return output


# backwards covector evolution along trajectory associated to x0, at
def gaussianDiffeonsCovectorPset(c0, S0, x0, at, pc1, pS1, px1, sigma, affine=None, withJacobian=None):
    dim = c0.shape[1]
    M = c0.shape[0]
    T = at.shape[0]
    timeStep = 1.0 / T
    # print c0.shape, x0.shape
    if not (withJacobian is None):
        # print withJacobian
        J0 = withJacobian[0]
        pJ1 = withJacobian[1]
        withJacobian = True
    else:
        J0 = None
        withJacobian = False
    if withJacobian:
        (ct, St, xt, Jt) = gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine=affine, withPointSet=x0,
                                                          withJacobian=J0)
    else:
        (ct, St, xt) = gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine=affine, withPointSet=x0)
    pct = np.zeros([T, M, dim])
    pSt = np.zeros([T, M, dim, dim])
    pxt = np.zeros(np.insert(x0.shape, 0, T))
    pxt[T - 1, ...] = px1
    pct[T - 1, :, :] = pc1
    pSt[T - 1, :, :, :] = pS1
    if withJacobian:
        pJt = np.tile(pJ1, np.insert(np.ones(J0.ndim, dtype=int), 0, T))
        # pJt = np.zeros(np.insert(J0.shape, 0, T))
        # pJt[T-1, ...] = pJ1
    sig2 = sigma * sigma

    if not (affine is None):
        A = affine[0]

    sigEye = sig2 * np.eye(dim)
    for t in range(T - 1):
        px = pxt[T - t - 1, ...]
        pc = pct[T - t - 1, :, :]
        pS = pSt[T - t - 1, :, :]
        x = xt[T - t - 1, ...]
        c = ct[T - t - 1, :, :]
        S = St[T - t - 1, :, :, :]
        a = at[T - t - 1, :, :]

        SS = S[:, None, :, :] + S[None, :, :, :]
        (R, detR) = gd.multiMatInverse1(sigEye[None, :, :] + S, isSym=True)
        (R2, detR2) = gd.multiMatInverse2(sigEye[None, None, :, :] + SS, isSym=True)

        diffx = x[..., np.newaxis, :] - c
        betax = (R * diffx[..., np.newaxis, :]).sum(axis=-1)
        dst = (diffx * betax).sum(axis=-1)
        fx = np.exp(-dst / 2)

        diffc = c[:, None, :] - c[None, :, :]
        betac = (R[None, :, :, :] * diffc[:, :, None, :]).sum(axis=3)
        dst = (diffc * betac).sum(axis=2)
        fc = np.exp(-dst / 2)

        Dv = -((fc[:, :, None] * betac)[:, :, None, :] * a[None, :, :, None]).sum(axis=1)
        IDv = np.eye(dim)[None, :, :] + timeStep * Dv
        SpS = (S[:, :, :, None] * (IDv[:, :, :, None] * pS[:, :, None, :]).sum(axis=1)[:, None, :, :]).sum(axis=2)

        aa = a @ a.T
        pxa = px @ a.T
        pca = pc @ a.T

        betaxSym = betax[..., np.newaxis] * betax[..., np.newaxis, :]
        betaxa = (betax * a).sum(axis=-1)
        betaSym = betac[:, :, :, None] * betac[:, :, None, :]
        betacc = (R2 * diffc[:, :, None, :]).sum(axis=3)
        betaSymcc = betacc[:, :, :, None] * betacc[:, :, None, :]
        dst = (betacc * diffc).sum(axis=2)
        gcc = np.sqrt((detR[:, None] * detR[None, :]) / ((sig2 ** dim) * detR2)) * np.exp(-dst / 2)
        spsa = (SpS[:, None, :, :] * a[None, :, None, :]).sum(axis=3)
        # print np.fabs(betacc + betacc.transpose([1,0,2])).sum()

        # print pxa.shape, fx.shape, betax.shape
        u = (pxa * fx)[..., np.newaxis] * betax
        zpx = u.sum(axis=-2)
        zpc = - u.sum(axis=tuple(range(x0.ndim - 1)))
        u2 = (pca * fc)[..., np.newaxis] * betac
        zpc += u2.sum(axis=1) - u2.sum(axis=0)

        # BmA = betaSym - R.reshape([1, M, dim, dim])
        Ra = (R * a[:, np.newaxis, :]).sum(axis=2)
        BmA = betaSym - R
        u = fc[:, :, None] * (BmA * spsa[:, :, None, :]).sum(axis=3)
        zpc -= 2 * (u.sum(axis=1) - u.sum(axis=0))
        zpc -= 2 * ((gcc * aa)[:, :, None] * betacc).sum(axis=1)

        zpS = - 0.5 * ((fx * pxa)[..., np.newaxis, np.newaxis] * betaxSym).sum(axis=tuple(range(x0.ndim - 1)))
        zpS -= 0.5 * ((fc * pca)[:, :, None, None] * betaSym).sum(axis=0)
        pSDv = (pS[:, :, :, None] * Dv[:, None, :, :]).sum(axis=2)
        zpS += -pSDv - pSDv.transpose((0, 2, 1)) - timeStep * (Dv[:, :, :, None] * pSDv[:, :, None, :]).sum(axis=1)
        u = fc * (spsa * betac).sum(axis=2)
        zpS += (u[:, :, None, None] * betaSym).sum(axis=0)
        u = (fc[:, :, None, None] * spsa[:, :, :, None] * betac[:, :, None, :]).sum(axis=0)
        u = (R[:, :, :, None] * u[:, None, :, :]).sum(axis=2)
        zpS -= u + u.transpose((0, 2, 1))
        zpS += ((gcc * aa)[:, :, None, None] * (betaSymcc - R2 + R[:, None, :, :])).sum(axis=1)

        if withJacobian:
            pJ = pJt[T - t - 1, ...]
            # print betaxa.shape, betax.shape, Ra.shape
            u = (pJ[..., np.newaxis] * fx)[..., np.newaxis] * (betaxa[..., np.newaxis] * betax - Ra)
            zpx -= u.sum(axis=-2)
            zpc += u.sum(axis=tuple(range(x0.ndim - 1)))
            zpS += 0.5 * ((pJ[..., np.newaxis] * fx * betaxa)[..., np.newaxis, np.newaxis] * betaxSym).sum(
                axis=tuple(range(x0.ndim - 1)))
            u = ((pJ[..., np.newaxis] * fx)[..., np.newaxis, np.newaxis] * (
                        Ra[..., np.newaxis] * betax[..., np.newaxis, :])).sum(axis=tuple(range(x0.ndim - 1)))
            zpS -= 0.5 * (u + u.transpose((0, 2, 1)))

        pxt[T - t - 2, ...] = px - timeStep * zpx
        pct[T - t - 2, :, :] = pc - timeStep * zpc
        pSt[T - t - 2, :, :, :] = pS - timeStep * zpS

        if affine is not None:
            At = A[T - t - 1, ...]
            pSA = pS[:, :, :] @ At[None, :, :]
            ApSA = (At[:, :].T)[None, :, :] @ pSA
            pSA += np.transpose(pSA, axes=(0,2,1))
            # pAS = (pStt[:, :, None, :] * idA[None, None, :, :]).sum(axis=3)
            # pAS = (pAS[:, :, None, :] * S[:, None, :, :]).sum(axis=3)
            pxt[T - t - 2, ...] += timeStep * px @ At
            pct[T - t - 2, :, :] += timeStep * pc @ At
            pSt[T-t-2, :, :] += timeStep * pSA + timeStep**2 * ApSA
            # pSt[T - t - 2, :, :, :] -= timeStep * ((At[None, :, :, None] * pStt[:, None, :, :]).sum(axis=2)
            #     + (pStt[:, :, None, :] * At[None, None, :, :]).sum(axis=3))
    # pSt[T - t - 2, :, :, :] -= timeStep * ((At[None, :, :, None] * pStt[:, None, :, :]).sum(axis=2)
    #                                        + (pStt[:, :, None, :] * At[None, None, :, :]).sum(axis=3))
    if withJacobian:
        return pct, pSt, pxt, pJt, ct, St, xt, Jt
    else:
        return pct, pSt, pxt, ct, St, xt


def gaussianDiffeonsCovectorNormals(c0, S0, b0, x0, xS0, at, pc1, pS1, pb1, px1, pxS1, sigma, regweight, affine=None):
    dim = c0.shape[1]
    N = x0.shape[0]
    M = c0.shape[0]
    T = at.shape[0]
    timeStep = 1.0 / T
    (ct, St, bt, xt, xSt) = gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine=affine, withNormals=b0,
                                                           withDiffeonSet=(x0, xS0))
    # print bt
    pbt = np.zeros([T, N, dim])
    pxt = np.zeros([T, N, dim])
    pxSt = np.zeros([T, N, dim, dim])
    pct = np.zeros([T, M, dim])
    pSt = np.zeros([T, M, dim, dim])
    pbt[T - 1, :, :] = pb1
    pxt[T - 1, :, :] = px1
    pxSt[T - 1, :, :, :] = pxS1
    pct[T - 1, :, :] = pc1
    pSt[T - 1, :, :, :] = pS1
    sig2 = sigma * sigma

    if not (affine is None):
        A = affine[0]

    sigEye = sig2 * np.eye(dim)
    for t in range(T - 1):
        pb = pbt[T - t - 1, :, :]
        px = pxt[T - t - 1, :, :]
        pxS = pxSt[T - t - 1, :, :, :]
        pc = pct[T - t - 1, :, :]
        pS = pSt[T - t - 1, :, :]
        x = xt[T - t - 1, :, :]
        xS = xSt[T - t - 1, :, :, :]
        b = bt[T - t - 1, :, :]
        c = ct[T - t - 1, :, :]
        S = St[T - t - 1, :, :, :]
        a = at[T - t - 1, :, :]

        SS = S[:, None, :, :] + S[None, :, :, :]
        (R, detR) = gd.multiMatInverse1(sigEye[None, :, :] + S, isSym=True)
        (R2, detR2) = gd.multiMatInverse2(sigEye[None, None, :, :] + SS, isSym=True)

        diffx = x[:, None, :] - c[None, :, :]
        betax = (R[None, :, :, :] * diffx[:, :, None, :]).sum(axis=3)
        dst = (diffx * betax).sum(axis=2)
        fx = np.exp(-dst / 2)

        diffc = c[:, None, :] - c.reshape[None, :, :]
        betac = (R[None, :, :, :] * diffc[:, :, None, :]).sum(axis=3)
        dst = (diffc * betac).sum(axis=2)
        fc = np.exp(-dst / 2)

        Dv = -((fc[:, :, None] * betac)[:, :, None, :] * a[None, :, :, None]).sum(axis=1)
        IDv = np.eye(dim)[None, :, :] + timeStep * Dv;
        SpS = (S[:, :, :, None] * (IDv[:, :, :, None] * pS[:, :, None, :]).sum(axis=1)[:, None, :, :]).sum(axis=2)
        xDv = -((fx[:, :, None] * betax)[:, :, None, :] * a[None, :, :, None]).sum(axis=1)
        xIDv = np.eye(dim)[None, :, :] + timeStep * xDv;
        # print xS.shape, xIDv.shape, pxS.shape
        xSpxS = (xS[:, :, :, None] * (xIDv[:, :, :, None] * pxS[:, :, None, :]).sum(axis=1)[:, None, :, :]).sum(axis=2)

        aa = a @ a.T
        pxa = px @ a.T
        pca = pc @ a.T
        # ba = (b.reshape([M,1,dim])*a.reshape([1,M,dim])).sum(axis=2)
        ba = b @ a.T
        pbbetax = (pb[:, None, :] * betax).sum(axis=2)
        pbb = (pb * b).sum(axis=1)
        betaxa = (betax * a[None, :, :]).sum(axis=2)
        # betaca = (betac*a.reshape([1, M, dim])).sum(axis=2)

        betaxSym = betax[:, :, :, None] * betax[:, :, None, :]
        betaSym = betac[:, :, :, None] * betac[:, :, None, :]
        betacc = (R2 * diffc[:, :, None, :]).sum(axis=3)
        betaSymcc = betacc[:, :, :, None] * betacc[:, :, None, :]
        dst = (betacc * diffc).sum(axis=2)
        gcc = np.sqrt((detR[:, None] * detR[None, :]) / ((sig2 ** dim) * detR2)) * np.exp(-dst / 2)
        spsa = (SpS[:, None, :, :] * a[None, :, None, :]).sum(axis=3)
        xspsa = (xSpxS[:, None, :, :] * a[None, :, None, :]).sum(axis=3)
        # print np.fabs(betacc + betacc.transpose([1,0,2])).sum()
        fpb = fx * pbbetax * ba
        Rpb = (R[None, :, :, :] * pb[:, None, None, :]).sum(axis=3)
        Ra = (R * a[:, None, :]).sum(axis=2)
        # print '?', (pbbetac**2).sum(), (Rpb**2).sum()

        zpb = - np.dot(fx * pbbetax, a) + (fx * betaxa).sum(axis=1)[:, None] * pb

        u = (pxa * fx)[:, :, None] * betax
        zpx = u.sum(axis=1)
        zpc = - u.sum(axis=0)

        u = fpb[:, :, None] * betax
        zpx += u.sum(axis=1)
        zpc -= u.sum(axis=0)
        u = (fx * ba)[:, :, None] * Rpb
        zpx -= u.sum(axis=1)
        zpc += u.sum(axis=0)
        u = (fx * pbb[:, None] * betaxa)[:, :, None] * betax
        zpx -= u.sum(axis=1)
        zpc += u.sum(axis=0)
        u = fx[:, :, None] * pbb[:, None, None] * Ra[None, :, :]
        zpx += u.sum(axis=1)
        zpc -= u.sum(axis=0)
        BmA = betaxSym - R[None, :, :, :]
        u = fx.reshape([N, M, 1]) * (BmA * xspsa[:, :, None, :]).sum(axis=3)
        zpx -= 2 * u.sum(axis=1)
        zpc += 2 * u.sum(axis=0)

        pSDv = (pxS[:, :, :, None] * xDv[:, None, :, :]).sum(axis=2)
        zpxS = -pSDv - np.transpose(pSDv, (0, 2, 1)) - timeStep * (xDv[:, :, :, None] * pSDv[:, :, None, :]).sum(axis=1)

        u = (pca * fc)[:, :, None] * betac
        zpc += u.sum(axis=1) - u.sum(axis=0)

        BmA = betaSym - R[None, :, :, :]
        u = fc[:, :, None] * (BmA * spsa[:, :, None, :]).sum(axis=3)
        zpc -= 2 * (u.sum(axis=1) - u.sum(axis=0))
        zpc -= 2 * ((gcc * aa)[:, :, None] * betacc).sum(axis=1)

        zpS = -0.5 * (fpb[:, :, None, None] * betaxSym).sum(axis=0)
        RpbbT = Rpb[:, :, :, None] * betax[:, :, None, :]
        bRpbT = Rpb[:, :, None, :] * betax[:, :, :, None]
        zpS += 0.5 * ((fx * ba)[:, :, None, None] * (RpbbT + bRpbT)).sum(axis=0)
        zpS += 0.5 * ((fx * betaxa * pbb[:, None])[:, :, None, None] * betaxSym).sum(axis=0)
        betaRaT = Ra[None, :, :, None] * betax[:, :, None, :]
        RabetaT = Ra[None, :, None, :] * betax[:, :, :, None]
        zpS -= 0.5 * ((fx * pbb[:, None])[:, :, None, None] * (betaRaT + RabetaT)).sum(axis=0)
        zpS -= 0.5 * ((fx * pxa)[:, :, None, None] * betaxSym).sum(axis=0)
        u = (fx * (xspsa * betax).sum(axis=2))
        zpS += (u[:, :, None, None] * betaxSym).sum(axis=0)
        u = (fx[:, :, None, None] * xspsa[:, :, :, None] * betax[:, :, None, :]).sum(axis=0)
        u = (R[:, :, :, None] * u[:, None, :, :]).sum(axis=2)
        zpS -= u + u.transpose((0, 2, 1))

        # zpS += 0.5*((fc*ba).reshape([M,M,1,1])*(RpbbT + np.transpose(RpbbT, (0,1,3,2)))).sum(axis=0)
        zpS -= 0.5 * ((fc * pca)[:, :, None, None] * betaSym).sum(axis=0)
        pSDv = (pS[:, :, :, None] * Dv[:, None, :, :]).sum(axis=2)
        zpS += -pSDv - np.transpose(pSDv, (0, 2, 1)) - timeStep * (Dv[:, :, :, None] * pSDv[:, :, None, :]).sum(axis=1)
        u = (fc * (spsa * betac).sum(axis=2))
        zpS += (u[:, :, None, None] * betaSym).sum(axis=0)
        u = (fc[:, :, None, None] * spsa[:, :, :, None] * betac[:, :, None, :]).sum(axis=0)
        u = (R[:, :, :, None] * u[:, None, :, :]).sum(axis=2)
        zpS -= u + u.transpose((0, 2, 1))
        zpS += ((gcc * aa)[:, :, None, None] * (betaSymcc - R2 + R[:, None, :, :])).sum(axis=1)

        pbt[T - t - 2, :, :] = pb - timeStep * zpb
        pxt[T - t - 2, :, :] = px - timeStep * zpx
        pxSt[T - t - 2, :, :] = pxS - timeStep * zpxS
        pct[T - t - 2, :, :] = pc - timeStep * zpc
        pSt[T - t - 2, :, :, :] = pS - timeStep * zpS

        if not (affine is None):
            pbt[T - t - 2, :, :] -= timeStep * pbt[T - t - 1, :, :] @ A[T - t - 1].T
            pct[T - t - 2, :, :] -= timeStep * pct[T - t - 1, :, :] @ A[T - t - 1]
            pSt[T - t - 2, :, :, :] -= \
                timeStep * ((A[T - t - 1, None, :, :, None] * pSt[T - t - 1, :, None, :, :]).sum(axis=2)
                            + (pSt[T - t - 1, :, :, None, :] * A[T - t - 1, None, None, :, :]).sum(axis=3))
            pxt[T - t - 2, :, :] -= timeStep * pxt[T - t - 1, :, :] @ A[T - t - 1]
            pxSt[T - t - 2, :, :, :] -= \
                timeStep * ((A[T - t - 1, None, :, :, None] * pxSt[T - t - 1, :, None, :, :]).sum(axis=2)
                            + (pxSt[T - t - 1, :, :, None :] * A[T - t - 1, None, None,  :, :]).sum(axis=3))

    return pct, pSt, pbt, pxt, pxSt, ct, St, bt, xt, xSt


# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def gaussianDiffeonsGradientPset(c0, S0, x0, at, pc1, pS1, px1, sigma, regweight, affine=None,
                                 withJacobian=None, euclidean=False):

    dA = None
    db = None
    pJt = None
    Jt = None
    if withJacobian is not None:
        # print withJacobian
        J0 = withJacobian[0]
        pJ1 = withJacobian[1]

    if withJacobian:
        (pct, pSt, pxt, pJt, ct, St, xt, Jt) = gaussianDiffeonsCovectorPset(c0, S0, x0, at, pc1, pS1, px1, sigma,
                                                                        affine=affine,
                                                                        withJacobian=withJacobian)
    else:
        (pct, pSt, pxt, ct, St, xt) = gaussianDiffeonsCovectorPset(c0, S0, x0, at, pc1, pS1, px1, sigma,
                                                                   affine=affine)
    # print (pct**2).sum()**0.5, (pSt**2).sum()**0.5, (pxt**2).sum()**0.5

    dat = np.zeros(at.shape)
    M = c0.shape[0]
    dim = c0.shape[1]
    if affine is not None:
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for t in range(at.shape[0]):
        a = at[t, :, :]
        x = xt[t, ...]
        c = ct[t, :, :]
        S = St[t, :, :, :]
        px = pxt[t, ...]
        pc = pct[t, :, :]
        pS = pSt[t, :, :, :]
        if withJacobian:
            J = Jt[t, ...]
            pJ = pJt[t, ...]
            [grc, grS, grx, grJ, gcc] = gd.gaussianDiffeonsGradientMatricesPset(c, S, x, a, pc, pS, px, sigma,
                                                                                1.0 / at.shape[0], withJacobian=(J, pJ))
            da = 2 * regweight * np.dot(gcc, a) - grx - grc - grS - grJ
        else:
            [grc, grS, grx, gcc] = gd.gaussianDiffeonsGradientMatricesPset(c, S, x, a, pc, pS, px, sigma,
                                                                           1.0 / at.shape[0])
            da = 2 *  regweight * np.dot(gcc, a) - grx - grc - grS

        if affine is not None:
            # print px.shape, x.shape, pc.shape, c.shape
            pSS = (pS @ S).sum(axis=0)
            dA[t] = (px[..., :, None] * x[..., None, :]).sum(axis=tuple(range(x.ndim - 1))) + pc.T @ c \
                    + 2*pSS + affine[0][t] @ (pSS + pSS.T) / at.shape[0]
                    #(pS[:, :, :, None] * S[:, :, None, :]).sum(axis=1).sum(axis=0)
            db[t] = px.sum(axis=tuple(range(x.ndim - 1))) + pc.sum(axis=0)

        if euclidean:
            dat[t, :, :] = da
        else:
            (L, W) = LA.eigh(gcc)
            dat[t, :, :] = LA.solve(gcc + (L.max() / 1000) * np.eye(M), da)
        # dat[t, :, :] = LA.solve(gcc, da)

    output = {
        'dat': dat,
        'dA': dA,
        'db': db,
        'ct': ct,
        'St': St,
        'xt': xt,
        'pct': pct,
        'pxt': pxt,
        'pSt': pSt,
        'Jt': Jt,
        'pJt': pJt
    }

    return output
    # if affine is None:
    #     if getCovector == False:
    #         if withJacobian:
    #             return dat, ct, St, xt, Jt
    #         else:
    #             return dat, ct, St, xt
    #     else:
    #         if withJacobian:
    #             return dat, ct, St, xt, Jt, pct, pSt, pxt, pJt
    #         else:
    #             return dat, ct, St, xt, pct, pSt, pxt
    # else:
    #     if getCovector == False:
    #         if withJacobian:
    #             return dat, dA, db, ct, St, xt, Jt
    #         else:
    #             return dat, dA, db, ct, St, xt
    #     else:
    #         if withJacobian:
    #             return dat, dA, db, ct, St, xt, Jt, pct, pSt, pxt, pJt
    #         else:
    #             return dat, dA, db, ct, St, xt, pct, pSt, pxt


def gaussianDiffeonsGradientNormals(c0, S0, b0, x0, xS0, at, pc1, pS1, pb1, px1, pxS1, sigma, regweight,
                                    getCovector=False, affine=None, euclidean=False):
    (pct, pSt, pbt, pxt, pxSt, ct, St, bt, xt, xSt) = gaussianDiffeonsCovectorNormals(c0, S0, b0, x0, xS0, at,
                                                                                      pc1, pS1, pb1, px1, pxS1, sigma,
                                                                                      regweight, affine=affine)

    # print (pct**2).sum()**0.5, (pSt**2).sum()**0.5, (pbt**2).sum()**0.5

    dat = np.zeros(at.shape)
    M = c0.shape[0]
    dim = c0.shape[1]
    if not (affine is None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for t in range(at.shape[0]):
        a = np.squeeze(at[t, :, :])
        b = np.squeeze(bt[t, :, :])
        x = np.squeeze(xt[t, :, :])
        xS = np.squeeze(xSt[t, :, :])
        c = np.squeeze(ct[t, :, :])
        S = np.squeeze(St[t, :, :])
        pb = np.squeeze(pbt[t, :, :])
        px = np.squeeze(pxt[t, :, :])
        pxS = np.squeeze(pxSt[t, :, :])
        pc = np.squeeze(pct[t, :, :])
        pS = np.squeeze(pSt[t, :, :])
        [grc, grS, grb, grx, grxS, gcc] = gd.gaussianDiffeonsGradientMatricesNormals(c, S, b, x, xS, a, pc, pS, pb, px,
                                                                                     pxS, sigma, 1.0 / at.shape[0])
        # print t, (grS**2).sum(), (grb**2).sum()

        da = 2 * np.dot(gcc, a) - grb - grc - grS - grx - grxS
        if not (affine is None):
            dA[t] = -np.dot(b.T, pb) + np.dot(pc.T, c) - 2 * np.multiply(pS.reshape([M, dim, dim, 1]),
                                                                         S.reshape([M, dim, 1, dim])).sum(axis=1).sum(
                axis=0)
            db[t] = pc.sum(axis=0)

        if euclidean:
            dat[t, :, :] = da
        else:
            (L, W) = LA.eigh(gcc)
            dat[t, :, :] = LA.solve(gcc + (L.max() / 1000) * np.eye(M), da)
        # dat[t, :, :] = LA.solve(gcc, da)

    if affine is None:
        if getCovector == False:
            return dat, ct, St, bt, xt, xSt
        else:
            return dat, ct, St, bt, xt, xSt, pct, pSt, pbt, pxt, pxSt
    else:
        if getCovector == False:
            return dat, dA, db, ct, St, bt, xt, xSt
        else:
            return dat, dA, db, ct, St, bt, xt, xSt, pct, pSt, pbt, pxt, pxSt
