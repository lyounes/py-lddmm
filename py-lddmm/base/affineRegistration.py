import numpy as np
import scipy.linalg as linalg

def rigidRegistrationLmk(x, y):
    N = x.shape[0]
    mx = x.sum(axis=0)/N
    my = y.sum(axis=0)/N
    rx = x-mx
    ry = y-my

    M = np.dot(ry.T, rx)
    sM = linalg.inv(np.real(linalg.sqrtm(np.dot(M.T, M))))
    R = M*sM
    T = my - np.dot(mx, R.T)
    return R,T


def rotpart(A, rotation=True):
    #d = linalg.det(A)
    U, S, Vh = linalg.svd(A)
    # UU = np.dot(U.T, U)
    # [D, Q] = linalg.eigh(UU)
    # dD = np.sqrt(D)
    if rotation and linalg.det(A)<0:
        #nr = True
        # print('Not rotation')
        j = np.argmin(S)
        Vh[j,:] *= -1
    #else:
    #    nr = False
    #     dD[0] = - dD[0]
    #R = np.dot(U, np.dot(Q, np.dot(np.diag(1./dD), Q.T)))
    R = np.dot(U, Vh)
    # if nr and linalg.det(R) > 0:
    #     print('(fixed)')
    return R

def saveRigid(filename, R, T):
    with open(filename,'w') as fn:
        for k,r in enumerate(R):
            str = '{0: f} {1: f} {2: f} {3: f} \n'.format(r[0], r[1], r[2], T[0,k])
            fn.write(str)

            
def _flipMidPoint(Y,X):
    M = Y.shape[0]
    dim = Y.shape[1]
    mx = X.sum(axis=0)/X.shape[0]
    my = Y.sum(axis=0)/M

    mid = ((mx+my)/2).reshape([1,dim])
    u = (mx - my).reshape([1, dim])
    nu = np.sqrt((u**2).sum())
    if (nu < 1e-10):
        u = np.ndarray([1,0,0])
    else:
        u = u/nu
    S = np.eye(3) - 2*np.dot(u.T, u)
    T = 2*np.multiply(mid, u).sum() * u
    Z = np.dot(Y, S.T) + T

    return Z, S, T



def rigidRegistration(surfaces = None, temperature = 1.0, rotWeight = 1.0, rotationOnly=False,
                      translationOnly=False, flipMidPoint=False,
                      annealing = True, verb=False, landmarks = None, normals = None):
#  [R, T] = rigidRegistrationSurface(X0, Y0, t)
# compute rigid registration using soft-assign maps
# computes R (orthogonal matrix) and T (translation so that Y0 ~ R*X0+T
# X0 and Y0 are point sets (size npoints x dimension)
# OPT.t: scale parameter for soft assign(default : 1)
# OPT.rotationOnly: 1 to estimate rotations only, 0 for rotations and
# symmetries (0: default)

#  Warning: won't work with large data sets (npoints should be less than a
#  few thousands).

    if (surfaces is None):
        surf = False
        norm = False
        Nsurf = 0
        Msurf = 0
        if landmarks is None:
            lmk = False
            print('Provide either surface points or landmarks or both')
            return
        else:
            lmk = True
            Nlmk = landmarks[1].shape[0]
            X0 = landmarks[1]
            Y0 = landmarks[0]
    else:
        surf = True
        X0 = surfaces[1]
        Y0 = surfaces[0]
        Nsurf = X0.shape[0]
        Msurf = Y0.shape[0]
        if landmarks is None:
            lmk = False
            N = Nsurf
            M = Msurf
            Nlmk = 0
        else:
            lmk = True
            Nlmk = landmarks[1].shape[0]
            N = Nsurf + Nlmk
            M = Msurf + Nlmk
            X0 = np.concatenate((X0, landmarks[1]))
            Y0 = np.concatenate((Y0, landmarks[0]))
        if normals is None:
            norm = False
        else:
            norm = True
            norm0 = normals[0]
            norm1 = normals[1]

    norm = False
    if lmk:
        if len(landmarks)==3:
            lmkWeight = landmarks[2]
        else:
            lmkWeight = 1.0

    if norm:
        if len(normals) == 3:
            normWeight = normals[2]
        else:
            normWeight = 1

    rotWeight *= X0.shape[0]
            
    t1 = temperature
    if flipMidPoint:
        [Y1, S1, T1] = _flipMidPoint(Y0, X0)
        if norm:
            norm0 *= -1
        #print S1, T1
    else:
        Y1 = Y0

    dimn = X0.shape[1]


    if surf:
        # Alternate minimization for surfaces with or without landmarks
        R = np.eye(dimn)
        T= np.zeros([1, dimn])
        RX = np.dot(X0,  R.T) + T
        d = (RX**2).sum(axis=1).reshape([N,1]) - 2 * np.dot(RX, Y1.T) + (Y1**2).sum(axis=1).reshape([1,M])
        if norm:
            Rn = np.dot(norm0, R.T)
            d[0:Nsurf, 0:Msurf] += normWeight*((Rn**2).sum(axis=1).reshape([Nsurf,1]) - 2 * np.dot(Rn, norm1.T) + (norm1**2).sum(axis=1).reshape([1,Msurf]))
        dSurf = d[0:Nsurf, 0:Msurf]
        Rold = np.copy(R)
        Told = np.copy(T)
        t0 = 10*t1
        t = t0
        c = .89
        w1 = np.zeros([N,M])
        w2 = np.zeros([N,M])
        if lmk:
            w1[Nsurf:N, Msurf:M] = lmkWeight*np.eye(Nlmk)
            w2[Nsurf:N, Msurf:M] = lmkWeight*np.eye(Nlmk)

        for k  in range(10000):
            # new weights
            if annealing and (k < 21):
                t = t*c 
            dmin = dSurf.min()
            wSurf = np.minimum((dSurf-dmin)/t, 500.)
            wSurf = np.exp(-wSurf)
            #wSurf = w[0:Nsurf, 0:Msurf] 
    
            #    w = sinkhorn(w, 100) ;
            Z  = wSurf.sum(axis=1)
            w1Surf = wSurf / Z[:, np.newaxis]
            Z  = wSurf.sum(axis = 0)
            w2Surf = wSurf / Z[np.newaxis, :]
            w1[0:Nsurf, 0:Msurf] = w1Surf
            w2[0:Nsurf, 0:Msurf] = w2Surf
            w = w1 + w2
            #ener = rotWeight*(3-np.trace(R)) + (np.multiply(w, d) + t*(np.multiply(w1Surf, np.log(w1Surf)) + np.multiply(w2Surf, np.log(w2Surf)))).sum()
            #if verb:
            #    print 'ener = ', ener


            # new transformation
            wX = np.dot(w.T, X0).sum(axis=0)
            wY = np.dot(w, Y1).sum(axis=0)

            Z = w.sum()
            #print Z, dSurf.min(), dSurf.max()
            mx = wX/Z
            my = wY/Z
            Y = Y1 - my
            X = X0 - mx
    
            if not translationOnly: 
                U = np.dot( np.dot(w, Y).T, X) + rotWeight * np.eye(dimn)
                if norm:
                    U += normWeight * np.dot(np.dot(w1Surf+w2Surf, norm0).T, norm1)
                R = rotpart(U, rotation=rotationOnly)
                # if rotationOnly:
                #     R = rotpart(U)
                # else:
                #     sU = linalg.inv(np.real(linalg.sqrtm(np.dot(U.T, U))))
                #     R = np.dot(U, sU)

            T = my - np.dot(mx, R.T)
            #print R, T
            RX = np.dot(X0, R.T) + T
        
            d = (RX**2).sum(axis=1).reshape([N,1]) - 2 * np.dot(RX, Y1.T) + (Y1**2).sum(axis=1).reshape([1,M])
            if norm:
                Rn = np.dot(norm0 ,R.T)
                d[0:Nsurf, 0:Msurf] += normWeight * ((Rn ** 2).sum(axis=1).reshape([Nsurf ,1]) - 2 * np.dot(Rn ,norm1.T) + (
                            norm1 ** 2).sum(axis=1).reshape([1 ,Msurf]))
            dSurf = d[0:Nsurf, 0:Msurf]
            ener = rotWeight*(dimn-np.trace(R)) + (w*d).sum() + t*((w1Surf*np.log(w1Surf)) + (w2Surf*np.log(w2Surf))).sum()
            #ener = rotWeight*(3-np.trace(R)) + (np.multiply(w, d) + t*(np.multiply(w1, np.log(w1)) + np.multiply(w2, np.log(w2)))).sum()

            if verb:
                print('ener = ', ener, 'var = ', np.fabs((R-Rold)).sum(), np.fabs((T-Told)).sum())

            if (k > 21) and (np.fabs((R-Rold)).sum() < 1e-3) and (np.fabs((T-Told)).sum() < 1e-2):
                break
            else:
                Told = np.copy(T)
                Rold = np.copy(R)
    else:
        # landmarks only
        R = np.eye(dimn)
        mx = X0.sum(axis=0)/Nlmk
        my = Y1.sum(axis=0)/Nlmk
        Y = Y1 - my
        X = X0 - mx
    
        if not translationOnly: 
            U = np.dot(Y.T, X) #+ rotWeight * np.eye(dimn)
            #print(U)
            if rotationOnly:
                R = rotpart(U)
            else:
                sU = linalg.inv(np.real(linalg.sqrtm(np.dot(U.T, U))))
                R = np.dot(U, sU)

        T = my - np.dot(mx, R.T)

    R = linalg.inv(R)
    T = - np.dot(T, R.T)
    T = T.reshape([1, dimn])

    #print R,T
    if flipMidPoint:
        T += np.dot(T1, R.T)
        R = np.dot(R, S1)

        #print R, T
    return R,T


def rigidRegistration_multi(surfaces, temperature=1.0, rotWeight=1.0, rotationOnly=False, translationOnly=False,
                            annealing=True, verb=False):
    #  [R, T] = rigidRegistrationSurface(X0, Y0, t)
    # compute rigid registration using soft-assign maps
    # computes R (orthogonal matrix) and T (translation so that Y0 ~ R*X0+T
    # X0 and Y0 are point sets (size npoints x dimension)
    # OPT.t: scale parameter for soft assign(default : 1)
    # OPT.rotationOnly: 1 to estimate rotations only, 0 for rotations and
    # symmetries (0: default)

    #  Warning: won't work with large data sets (npoints should be less than a
    #  few thousands).

    X0 = surfaces[1]
    Y1 = surfaces[0]
    Nsurf = np.zeros(len(X0), dtype=int)
    Msurf = np.zeros(len(Y1), dtype=int)
    for k,s in enumerate(X0):
        Nsurf[k] = s.shape[0]
    for k,s in enumerate(Y1):
        Msurf[k] = s.shape[0]


    rotWeight *= Nsurf.sum()

    t1 = temperature

    dimn = X0[0].shape[1]
    ns = len(X0)

    # Alternate minimization for surfaces with or without landmarks
    R = np.eye(dimn)
    T = np.zeros([1, dimn])
    RX = []
    dSurf = []
    for k in range(ns):
        RX.append(np.dot(X0[k], R.T) + T)
        dSurf.append((RX[k] ** 2).sum(axis=1).reshape([Nsurf[k], 1]) - 2 * np.dot(RX[k], Y1[k].T)
                 + (Y1[k] ** 2).sum(axis=1).reshape([1, Msurf[k]]))
    Rold = np.copy(R)
    Told = np.copy(T)
    t0 = 10 * t1
    t = t0
    c = .89
    w1 = []
    w2 = []
    w = []
    for k in range(ns):
        w1.append(np.zeros([Nsurf[k], Msurf[k]]))
        w2.append(np.zeros([Nsurf[k], Msurf[k]]))
        w.append(np.zeros([Nsurf[k], Msurf[k]]))

    for k0 in range(10000):
        # new weights
        if annealing and (k0 < 21):
            t = t * c
        for k in range(ns):
            dmin = dSurf[k].min()
            wSurf = np.minimum((dSurf[k] - dmin) / t, 500.)
            wSurf = np.exp(-wSurf)
            # wSurf = w[0:Nsurf, 0:Msurf]

            #    w = sinkhorn(w, 100) ;
            Z = wSurf.sum(axis=1)
            w1[k] = wSurf / Z[:, np.newaxis]
            Z = wSurf.sum(axis=0)
            w2[k] = wSurf / Z[np.newaxis, :]
            w[k] = w1[k] + w2[k]
            # ener = rotWeight*(3-np.trace(R)) + (np.multiply(w, d) + t*(np.multiply(w1Surf, np.log(w1Surf)) + np.multiply(w2Surf, np.log(w2Surf)))).sum()
            # if verb:
            #    print 'ener = ', ener

            # new transformation
        mx = np.zeros(dimn)
        my = np.zeros(dimn)
        Z = 0
        for k in range(ns):
            wX = np.dot(w[k].T, X0[k]).sum(axis=0)
            wY = np.dot(w[k], Y1[k]).sum(axis=0)

            Z += w[k].sum()
            # print Z, dSurf.min(), dSurf.max()
            mx += wX
            my += wY
        mx /= Z
        my /= Z

        Y = []
        X = []
        for k in range(ns):
            Y.append(Y1[k] - my)
            X.append(X0[k] - mx)

        if not translationOnly:
            U = np.zeros((dimn, dimn))
            for k in range(ns):
                U += np.dot(np.dot(w[k], Y[k]).T, X[k])
            U += rotWeight * np.eye(dimn)
            if rotationOnly:
                R = rotpart(U)
            else:
                sU = linalg.inv(np.real(linalg.sqrtm(np.dot(U.T, U))))
                R = np.dot(U, sU)

        T = my - np.dot(mx, R.T)
        # print R, T
        RX = []
        dSurf = []
        for k in range(ns):
            RX.append(np.dot(X0[k], R.T) + T)
            dSurf.append((RX[k] ** 2).sum(axis=1).reshape([Nsurf[k], 1]) - 2 * np.dot(RX[k], Y1[k].T)
                     + (Y1[k] ** 2).sum(axis=1).reshape([1, Msurf[k]]))

        ener = rotWeight * (dimn - np.trace(R))
        for k in range(ns):
            ener += (w[k] * dSurf[k]).sum() + t * (
                    (w1[k] * np.log(w1[k])) + (w2[k] * np.log(w2[k]))).sum()
        # ener = rotWeight*(3-np.trace(R)) + (np.multiply(w, d) + t*(np.multiply(w1, np.log(w1)) + np.multiply(w2, np.log(w2)))).sum()

        if verb:
            print('ener = ', ener, 'var = ', np.fabs((R - Rold)).sum(), np.fabs((T - Told)).sum())

        if (k0 > 21) and (np.fabs((R - Rold)).sum() < 1e-3) and (np.fabs((T - Told)).sum() < 1e-2):
            break
        else:
            Told = np.copy(T)
            Rold = np.copy(R)

    R = linalg.inv(R)
    T = - np.dot(T, R.T)
    T = T.reshape([1, dimn])

    # print R,T
    return R, T

