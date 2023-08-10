import logging

import numpy as np
from .kernelFunctions_util import applyK1K2, applyDiffK1K2T
from pointSets_util import det2D, det3D, rot90


def varifoldNorm0(fv1, KparDist, KparIm, imKernel = None):
    c2 = fv1.centers
    a2 = fv1.weights * fv1.volumes
    if imKernel is None:
        A1 = fv1.image
    else:
        A1 = np.dot(fv1.image, imKernel)

    return ((applyK1K2(c2, c2, KparDist.name, KparDist.sigma, KparDist.order,
                     A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a2[:, None], cpu=False))*a2[:, None]).sum()


# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot product
def varifoldNormDef(fvDef, fv1, KparDist, KparIm, imKernel = None):
    c1 = fvDef.centers
    c2 = fv1.centers
    a1 = fvDef.weights * fvDef.volumes
    a2 = fv1.weights * fv1.volumes
    if imKernel is None:
        A1 = fvDef.image
    else:
        A1 = np.dot(fvDef.image, imKernel)

    obj = ((applyK1K2(c1, c1, KparDist.name, KparDist.sigma, KparDist.order,
                     A1, fvDef.image, KparIm.name, KparIm.sigma, KparIm.order, a1[:, None], cpu=False) -
           2*applyK1K2(c1, c2, KparDist.name, KparDist.sigma, KparDist.order,
                     A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a2[:, None], cpu=False)) * a1[:, None]).sum()

    return obj

def varifoldNormDef_old(fvDef, fv1, KparDist, imKernel = None):
    c1 = fvDef.centers
    c2 = fv1.centers
    a1 = fvDef.weights * fvDef.volumes
    a2 = fv1.weights * fv1.volumes
    if imKernel is None:
        betax1 = a1[:, None] * fvDef.image
        betay1 = betax1
        betay2 = a2[:, None] * fv1.image
    else:
        A1 = np.dot(fvDef.image, imKernel)
        betax1 = a1[:, None] * A1
        betay1 = a1[:, None] * fvDef.image
        betay2 = a2[:, None] * fv1.image

    obj = (betax1*KparDist.applyK(c1, betay1)).sum() \
          - 2*(betax1*KparDist.applyK(c2,betay2, firstVar=c1)).sum()
    return obj

# def varifoldNormDef_old(fvDef, fv1, KparDist, imKernel = None):
#     c1 = fvDef.centers
#     c2 = fv1.centers
#     a1 = fvDef.weights * fvDef.volumes
#     a2 = fv1.weights * fv1.volumes
#     if imKernel is None:
#         cr1cr1 = (fvDef.image[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
#         cr1cr2 = (fvDef.image[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
#     else:
#         A1 = np.dot(fvDef.image, imKernel)
#         cr1cr1 = (A1[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
#         cr1cr2 = (A1[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
#
#     a1a1 = a1[:, np.newaxis] * a1[np.newaxis, :]
#     a1a2 = a1[:, np.newaxis] * a2[np.newaxis, :]
#
#     beta1 = cr1cr1 * a1a1
#     beta2 = cr1cr2 * a1a2
#
#     obj = (KparDist.applyK(c1, beta1[..., np.newaxis], matrixWeights=True).sum()
#            - 2 * KparDist.applyK(c2, beta2[..., np.newaxis], firstVar=c1, matrixWeights=True).sum())
#     return obj
#


# Returns |fvDef - fv1|^2 for current norm
def varifoldNorm(fvDef, fv1, KparDist, KparIm, imKernel = None):
    return varifoldNormDef(fvDef, fv1, KparDist, KparIm, imKernel=imKernel) \
           + varifoldNorm0(fv1, KparDist, KparIm, imKernel=imKernel)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def varifoldNormGradient(fvDef, fv1, KparDist, KparIm, with_weights=False, imKernel=None):
    c1 = fvDef.centers
    nf = c1.shape[0]
    c2 = fv1.centers
    a1 = fvDef.weights
    a2 = fv1.weights
    a1v = fvDef.weights * fvDef.volumes
    a2v = fv1.weights * fv1.volumes
    if imKernel is None:
        A1 = fvDef.image
    else:
        A1 = np.dot(fvDef.image, imKernel)

    crx1 = a1[:, None] * A1
    crx1v = a1v[:, None] * A1
    cry1v = a1v[:, None] * fvDef.image
    cry2v = a2v[:, None] * fv1.image

    dim = c1.shape[1]
    normals = np.zeros((dim+1, nf, dim))
    if dim == 2:
        k1 = 2
        k2 = 3
        #J = np.array([[0, -1], [1,0]])
        normals[0, :, :] = fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :]
        normals[1, :, :] = fvDef.vertices[fvDef.faces[:, 0], :] - fvDef.vertices[fvDef.faces[:, 2], :]
        normals[2, :, :] = fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :]
        normals = np.flip(normals, axis=2)
        normals[:,:,0] = - normals[:,:,0]
        #normals  = normals @ J
    else:
        k1 = 6
        k2 = 4
        normals[0,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 1], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :])
        normals[1,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[2,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[3,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :])

    # u1 = a1a1[:,:] * cr1cr1[:,:] * fvDef.volumes[None, :]
    # u2 = a1a2[:,:] * cr1cr2[:,:] * fv1.volumes[None, :]
    #z1 = (crx1 * KparDist.applyK(c1, cry1v) - crx1 * KparDist.applyK(c2, cry2v, firstVar=c1)).sum(axis=1)
    z1 = (a1[:, None]*applyK1K2(c1, c1, KparDist.name, KparDist.sigma, KparDist.order,
                                A1, fvDef.image, KparIm.name, KparIm.sigma, KparIm.order, a1v[:, None])
          - a1[:, None]*applyK1K2(c1, c2, KparDist.name, KparDist.sigma, KparDist.order,
                                  A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a2v[:, None])).sum(axis=1)
    # z1_ = (KparDist.applyK(c1, u1[:,:, None], matrixWeights=True) -
    #      KparDist.applyK(c2, u2[:,:,None], firstVar=c1, matrixWeights=True))
    z1 = z1[None, :, None] * normals/k1

    # beta1 = a1a1v * cr1cr1
    # beta2 = a1a2v * cr1cr2
    dz1 = (applyDiffK1K2T(c1, c1, KparDist.name, KparDist.sigma, KparDist.order,
                         A1, fvDef.image, KparIm.name, KparIm.sigma, KparIm.order, a1v[:, None], a1v[:, None])
           - applyDiffK1K2T(c1, c2, KparDist.name, KparDist.sigma, KparDist.order,
                         A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a1v[:, None], a2v[:, None]))/k2

    # dz1 = (KparDist.applyDiffKT(c1, crx1v, cry1v) - KparDist.applyDiffKT(c2, crx1v, cry2v, firstVar=c1))/k2
    # dz1 = (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))/k2


    px = np.zeros(fvDef.vertices.shape)
    for i in range(dim+1):
        I = fvDef.faces[:, i]
        for k in range(I.size):
            px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[i, k, :]
    return 2*px

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def varifoldNormGradient_old(fvDef, fv1, KparDist, with_weights=False, imKernel=None):
    c1 = fvDef.centers
    nf = c1.shape[0]
    c2 = fv1.centers
    a1 = fvDef.weights
    a2 = fv1.weights
    a1v = fvDef.weights * fvDef.volumes
    a2v = fv1.weights * fv1.volumes
    if imKernel is None:
        cr1cr1 = (fvDef.image[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
        cr1cr2 = (fvDef.image[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
    else:
        A1 = np.dot(fvDef.image, imKernel)
        cr1cr1 = (A1[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
        cr1cr2 = (A1[:, None, :] * fv1.image[None, :, :]).sum(axis=2)

    a1a1 = a1[:, np.newaxis] * a1[np.newaxis, :]
    a1a2 = a1[:, np.newaxis] * a2[np.newaxis, :]
    a1a1v = a1v[:, np.newaxis] * a1v[np.newaxis, :]
    a1a2v = a1v[:, np.newaxis] * a2v[np.newaxis, :]
    dim = c1.shape[1]
    normals = np.zeros((dim+1, nf, dim))
    if dim == 2:
        k1 = 2
        k2 = 3
        #J = np.array([[0, -1], [1,0]])
        normals[0, :, :] = fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :]
        normals[1, :, :] = fvDef.vertices[fvDef.faces[:, 0], :] - fvDef.vertices[fvDef.faces[:, 2], :]
        normals[2, :, :] = fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :]
        normals = np.flip(normals, axis=2)
        normals[:,:,0] = - normals[:,:,0]
        #normals  = normals @ J
    else:
        k1 = 6
        k2 = 4
        normals[0,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 1], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :])
        normals[1,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[2,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[3,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :])

    u1 = a1a1[:,:] * cr1cr1[:,:] * fvDef.volumes[None, :]
    u2 = a1a2[:,:] * cr1cr2[:,:] * fv1.volumes[None, :]
    z1 = (KparDist.applyK(c1, u1[:,:, None], matrixWeights=True) -
         KparDist.applyK(c2, u2[:,:,None], firstVar=c1, matrixWeights=True))
    z1 = z1[None, :, :] * normals/k1

    beta1 = a1a1v * cr1cr1
    beta2 = a1a2v * cr1cr2
    dz1 = (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))/k2


    px = np.zeros(fvDef.vertices.shape)
    for i in range(dim+1):
        I = fvDef.faces[:, i]
        for k in range(I.size):
            px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[i, k, :]
    return 2*px

########
##Internal costs
########

def square_divergence(x,v,faces):
    return square_divergence_(x,v,faces)

def normalized_square_divergence(x,v,faces):
    return square_divergence_(x,v,faces, normalize=True)

def square_divergence_(x, v, faces, normalize = False):
    dim = x.shape[1]
    nf = faces.shape[0]
    vol = np.zeros(nf)
    div = np.zeros(nf)
    if dim==2:
        x0 = x[faces[:, 0], :]
        x1 = x[faces[:, 1], :]
        x2 = x[faces[:, 2], :]
        v0 = v[faces[:, 0], :]
        v1 = v[faces[:, 1], :]
        v2 = v[faces[:, 2], :]
        vol = np.fabs(det2D(x1-x0, x2-x0))
        div = det2D(v2, x0-x1) + det2D(v0, x1-x2) + det2D(v1, x2-x0)
    elif dim == 3:
        x0 = x[faces[:, 0], :]
        x1 = x[faces[:, 1], :]
        x2 = x[faces[:, 2], :]
        x3 = x[faces[:, 3], :]
        v0 = v[faces[:, 0], :]
        v1 = v[faces[:, 1], :]
        v2 = v[faces[:, 2], :]
        v3 = v[faces[:, 3], :]
        vol = np.fabs(det3D(x1-x0, x2-x0, x3-x0))
        div = det3D(v3, x0-x1, x0-x2) + det3D(v0, x1-x2, x1-x3) + det3D(v1, x2-x3, x2-x0) + det3D(v2, x3-x0, x3-x1)
    else:
        logging.warning('square divergence: unrecognized dimension')

    if normalize:
        res = ((div ** 2)/np.maximum(vol, 1e-10)).sum() / vol.sum()
    else:
        res = ((div ** 2)/np.maximum(vol, 1e-10)).sum()
    return res

def square_divergence_grad(x,v,faces, variables='both'):
    return square_divergence_grad_(x,v,faces, variables=variables)

def normalized_square_divergence_grad(x,v,faces, variables='both'):
    return square_divergence_grad_(x,v,faces, normalize=True, variables=variables)


def square_divergence_grad_(x, v, faces, variables = 'both', normalize=False):
    dim = x.shape[1]
    nf = faces.shape[0]
    gradx = np.zeros(x.shape)
    gradphi = np.zeros(v.shape)
    test = True
    grad = dict()
    #logging.info(f"dim = {dim}, variables = {variables}")
    if dim==2:
        x0 = x[faces[:, 0], :]
        x1 = x[faces[:, 1], :]
        x2 = x[faces[:, 2], :]
        v0 = v[faces[:, 0], :]
        v1 = v[faces[:, 1], :]
        v2 = v[faces[:, 2], :]
        vol = np.fabs(det2D(x1-x0, x2-x0))
        div = det2D(v2, x0-x1) + det2D(v0, x1-x2) + det2D(v1, x2-x0)
        c1 = 2 * (div / vol)[:, None]
        if normalize:
            totalVol = vol.sum()
            sqdiv = (div ** 2 / vol).sum()
        else:
            totalVol = 1
            sqdiv = 1
        if variables == 'phi' or variables == 'both':
            dphi2 = -rot90(x0-x1) * c1
            dphi0 = -rot90(x1-x2) * c1
            dphi1 = -rot90(x2-x0) * c1
            for k, f in enumerate(faces):
                gradphi[f[0], :] += dphi0[k, :]
                gradphi[f[1], :] += dphi1[k, :]
                gradphi[f[2], :] += dphi2[k, :]
            grad['phi'] = gradphi / totalVol
            if test == True:
                eps = 1e-10
                h = np.random.normal(0,1,v.shape)
                fp = square_divergence_(x, v+eps*h, faces, normalize=normalize)
                fm = square_divergence_(x, v-eps*h, faces, normalize=normalize)
                logging.info(f"test sqdiv v: {(grad['phi']*h).sum():.4f} {(fp-fm)/(2*eps):.4f}")
            #gradphi = -gradphi
        if variables == 'x' or variables == 'both':
            c2 = ((div/vol)**2)[:, None]
            dx0 = -rot90(v1 - v2) * c1 + rot90(x1-x2)*c2
            dx1 = -rot90(v2 - v0) * c1 + rot90(x2-x0)*c2
            dx2 = -rot90(v0 - v1) * c1 + rot90(x0-x1)*c2
            if normalize:
                dx0 -= rot90(x1-x2) * sqdiv / totalVol
                dx1 -= rot90(x2-x0) * sqdiv /totalVol
                dx2 -= rot90(x0-x1) * sqdiv / totalVol

            for k, f in enumerate(faces):
                gradx[f[0], :] += dx0[k, :]
                gradx[f[1], :] += dx1[k, :]
                gradx[f[2], :] += dx2[k, :]
            grad['x'] = gradx/totalVol
            #gradx = -gradx
            if test == True:
                eps = 1e-10
                h = np.random.normal(0, 1, x.shape)
                fp = square_divergence_(x + eps * h, v, faces, normalize=normalize)
                fm = square_divergence_(x - eps * h, v, faces, normalize=normalize)
                logging.info(f"test sqdiv x: {(grad['x']*h).sum():.4f} {(fp - fm) / (2 * eps):.4f}")
    elif dim == 3:
        x0 = x[faces[:, 0], :]
        x1 = x[faces[:, 1], :]
        x2 = x[faces[:, 2], :]
        x3 = x[faces[:, 3], :]
        v0 = v[faces[:, 0], :]
        v1 = v[faces[:, 1], :]
        v2 = v[faces[:, 2], :]
        v3 = v[faces[:, 3], :]
        vol = np.fabs(det3D(x1-x0, x2-x0, x3-x0))
        div = det3D(v3, x0-x1, x0-x2) + det3D(v0, x1-x2, x1-x3) + det3D(v1, x2-x3, x2-x0) + det3D(v2, x3-x0, x3-x1)
        c1 = 2 * (div / vol)[:, None]
        if normalize:
            totalVol = vol.sum()
            sqdiv = (div ** 2 / vol).sum()
        else:
            totalVol = 1
            sqdiv = 1
        if variables == 'phi' or variables == 'both':
            dphi0 = np.cross(x1-x2, x1-x3) * c1
            dphi1 = np.cross(x2-x3, x2-x0) * c1
            dphi2 = np.cross(x3-x0, x3-x1) * c1
            dphi3 = np.cross(x0-x1, x0-x2) * c1
            for k, f in enumerate(faces):
                gradphi[f[0], :] += dphi0[k, :]
                gradphi[f[1], :] += dphi1[k, :]
                gradphi[f[2], :] += dphi2[k, :]
                gradphi[f[3], :] += dphi3[k, :]
            grad['phi'] = gradphi/totalVol
            if test == True:
                eps = 1e-10
                h = np.random.normal(0,1,v.shape)
                fp = square_divergence_(x, v+eps*h, faces, normalize=normalize)
                fm = square_divergence_(x, v-eps*h, faces, normalize=normalize)
                logging.info(f"test sqdiv v: {(grad['phi']*h).sum():.4f} {(fp-fm)/(2*eps):.4f}")

        if variables == 'x' or variables == 'both':
            c2 = ((div/vol)**2)[:, None]
            dx0 = (np.cross(v1, x3-x2) + np.cross(v2, x3-x1) + np.cross(v3, x2-x1)) * c1 - c2 * np.cross(x1-x3, x2-x3)
            dx1 = (np.cross(v2, x0-x3) + np.cross(v3, x0-x2) + np.cross(v0, x3-x2)) * c1 - c2 * np.cross(x2-x0, x3-x0)
            dx2 = (np.cross(v3, x1-x0) + np.cross(v0, x1-x3) + np.cross(v1, x0-x3)) * c1 - c2 * np.cross(x3-x1, x0-x1)
            dx3 = (np.cross(v0, x2-x1) + np.cross(v1, x2-x0) + np.cross(v2, x1-x0)) * c1 - c2 * np.cross(x0-x2, x1-x2)
            if normalize:
                dx0 -= np.cross(x1 - x3, x2 - x3) * sqdiv / totalVol
                dx1 -= np.cross(x2 - x0, x3 - x0) * sqdiv / totalVol
                dx2 -= np.cross(x3 - x1, x0 - x1) * sqdiv / totalVol
                dx3 -= np.cross(x0 - x2, x1 - x2) * sqdiv / totalVol

            for k, f in enumerate(faces):
                gradx[f[0], :] += dx0[k, :]
                gradx[f[1], :] += dx1[k, :]
                gradx[f[2], :] += dx2[k, :]
                gradx[f[3], :] += dx3[k, :]
            grad['x'] = gradx/totalVol
            if test == True:
                eps = 1e-10
                h = np.random.normal(0, 1, x.shape)
                fp = square_divergence_(x + eps * h, v, faces, normalize=normalize)
                fm = square_divergence_(x - eps * h, v, faces, normalize=normalize)
                logging.info(f"test sqdiv x: {(grad['x']*h).sum():.4f} {(fp - fm) / (2 * eps):.4f}")

    else:
        logging.warning('square divergence grad: unrecognized dimension')

    return grad
    # if variables == 'both':
    #     return (gradphi, gradx)
    # elif variables == 'phi':
    #     return gradphi
    # elif variables == 'x':
    #     return gradx
    # else:
    #     logging.info('Incorrect option in square_divergence_grad')
