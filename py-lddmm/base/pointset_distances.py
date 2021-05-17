import numpy as np

def L2Norm0(x1):
    return (x1**2).sum()

def L2NormDef(xDef, x1):
    return -2*(xDef*x1).sum() + (xDef**2).sum()

def L2NormGradient(xDef,x1):
    return 2*(xDef-x1)


# Measure norm of fv1
def measureNorm0(fv1, KparDist):
    cr2 = np.ones((fv1.shape[0],1))/fv1.shape[0]
    return KparDist.applyK(fv1, cr2).sum()


# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct
def measureNormDef(fvDef, fv1, KparDist):
    cr1 = np.ones((fvDef.shape[0],1))/fvDef.shape[0]
    cr2 = np.ones((fv1.shape[0],1))/fv1.shape[0]
    obj = (np.multiply(cr1, KparDist.applyK(fvDef, cr1)).sum()
           - 2 * np.multiply(cr1, KparDist.applyK(fv1, cr2, firstVar=fvDef)).sum())
    return obj


# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist):
    dim = fvDef.shape[1]
    cr1 = np.ones(fvDef.shape[0])/fvDef.shape[0]
    cr2 = np.ones(fv1.shape[0])/fv1.shape[0]

    dz1 = (KparDist.applyDiffKT(fvDef, cr1[np.newaxis, :, np.newaxis], cr1[np.newaxis, :, np.newaxis]) -
                       KparDist.applyDiffKT(fv1, cr1[np.newaxis, :, np.newaxis], cr2[np.newaxis, :, np.newaxis],
                                            firstVar=fvDef))

    return 2 * dz1


