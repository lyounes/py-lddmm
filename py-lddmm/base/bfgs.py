import numpy as np
import logging
from copy import deepcopy
from .linesearch import line_search_wolfe, line_search_weak_wolfe, line_search_goldstein_price

# added comment to test git, 4-26-19

# Class running BFGS
# opt is an optimizable class that must provide the following functions:
#   getVariable(): current value of the optimzed variable
#   objectiveFun(): value of the objective function
#   updateTry(direction, step, [acceptThreshold]) computes a temporary variable by moving the current one in the direction 'direction' with step 'step'
#                                                 the temporary variable is not stored if the objective function is larger than acceptThreshold (when specified)
#                                                 This function should not update the current variable
#   acceptVarTry() replace the current variable by the temporary one
#   getGradient(coeff) returns coeff * gradient; the result can be used as 'direction' in updateTry
#
# optional functions:
#   startOptim(): called before starting the optimization
#   startOfIteration(): called before each iteration
#   endOfIteration() called after each iteration
#   endOptim(): called once optimization is completed
#   dotProduct(g1, g2): returns a list of dot products between g1 and g2, where g1 is a direction and g2 a list of directions
#                       default: use standard dot product assuming that directions are arrays
#   addProd(g0, step, g1): returns g0 + step * g1 for directions g0, g1
#   copyDir(g0): returns a copy of g0
#   randomDir(): Returns a random direction
# optional attributes:
#   gradEps: stopping theshold for small gradient
#   gradCoeff: normalizaing coefficient for gradient.
#
# verb: for verbose printing
# TestGradient evaluate accracy of first order approximation (debugging)
# epsInit: initial gradient step

def __dotProduct(x,y):
    res = []
    for yy in y:
        res.append((x*yy).sum())
    return res

def __addProd(x,y,a):
    return x + a*y

def __prod(x, a):
    return a*x

def __copyDir(x):
    return np.copy(x)

def __stopCondition():
    return True

def bfgs(opt, verb = True, maxIter=1000, TestGradient = False, epsInit=0.01, memory=25, Wolfe = True,
         lineSearch = 'Weak_Wolfe'):
    if (hasattr(opt, 'getVariable')==False or hasattr(opt, 'objectiveFun')==False or hasattr(opt, 'updateTry')==False
            or hasattr(opt, 'acceptVarTry')==False or hasattr(opt, 'getGradient')==False):
        logging.error('Error: required functions are not provided')
        return

    if lineSearch == "Wolfe":
        line_search = line_search_wolfe
    elif lineSearch == "Goldstein_Price":
        line_search = line_search_goldstein_price
    elif lineSearch == "Weak_Wolfe":
        line_search = line_search_weak_wolfe
    else:
        logging.warning('Unrecognized line search: using weak wolfe condition')
        line_search = line_search_weak_wolfe


    if hasattr(opt, 'dotProduct_euclidean'):
        dotProduct = opt.dotProduct_euclidean
    elif hasattr(opt, 'dotProduct'):
        dotProduct = opt.dotProduct
    else:
        dotProduct = __dotProduct

    if not(hasattr(opt, 'addProd')):
        addProd = __addProd
    else:
        addProd = opt.addProd

    if not(hasattr(opt, 'stopCondition')):
        stopCondition = __stopCondition
    else:
        stopCondition = opt.stopCondition

    if not(hasattr(opt, 'prod')):
        prod = __prod
    else:
        prod = opt.prod

    if not(hasattr(opt, 'copyDir')):
        copyDir = __copyDir
    else:
        copyDir = opt.copyDir

    if hasattr(opt, 'startOptim'):
        opt.startOptim()

    if hasattr(opt, 'gradEps'):
        gradEps = opt.gradEps
    elif hasattr(opt, 'options') and 'gradEps' in opt.options.keys():
        gradEps = opt.options['gradEps']
    else:
        gradEps = None

    if hasattr(opt, 'gradCoeff'):
        gradCoeff = opt.gradCoeff
    else:
        gradCoeff = 1.0

    if hasattr(opt, 'epsMax'):
        epsMax = opt.epsMax
    else:
        epsMax = 1.

    if hasattr(opt, 'burnIn'):
        burnIn = opt.burnIn
    else:
        burnIn = 0
    eps = epsInit
    epsMin = 1e-10
    opt.converged = False

    if hasattr(opt, 'reset') and opt.reset:
        opt.obj = None

    obj = opt.objectiveFun()
    opt.reset = False
    #obj = opt.objectiveFun()
    logging.info('iteration 0: obj = {0: .5f}'.format(obj))
    # if (obj < 1e-10):
    #     return opt.getVariable()


    storedGrad = []
    noUpdate = 0
    it = 0
    diffVar = None
    grdOld = None
    obj_old = None
    gval = None
    while it < maxIter:
        #gval = None
        if hasattr(opt, 'startOfIteration'):
            opt.startOfIteration()
        if opt.reset:
            opt.obj = None
            obj = opt.objectiveFun()
            obj_old = None
            gval = None

        #gval = None
        if gval is None:
            grd = opt.getGradient(gradCoeff)
        else:
            grd = deepcopy(gval)

        if TestGradient:
            if hasattr(opt, 'randomDir'):
                dirfoo = opt.randomDir()
            else:
                dirfoo = np.random.normal(size=grd.shape)
            epsfoo = 1e-8
            objfoo1 = opt.updateTry(dirfoo, epsfoo, obj-1e8)
            objfoo2 = opt.updateTry(dirfoo, -epsfoo, obj-1e8)
            [grdfoo] = dotProduct(grd, [dirfoo])
            logging.info('Test Gradient: %.4f %.4f' %((objfoo1 - objfoo2)/(2*epsfoo), -grdfoo * gradCoeff ))

        if (not opt.reset)  and it > 0:
            storedGrad.append([diffVar, addProd(grd, grdOld, -1)])
            if len(storedGrad) > memory:
                storedGrad.pop(0)
            q = copyDir(grd)
            rho = []
            alpha = []
            for m in reversed(storedGrad):
                rho.append(1/dotProduct(m[1], [m[0]])[0])
                alpha.append(rho[-1]*dotProduct(m[0],[q])[0])
                q = addProd(q, m[1], -alpha[-1])
            rho.reverse()
            alpha.reverse()
            m = storedGrad[-1]
            c = dotProduct(m[0],[m[1]])[0]/dotProduct(m[1],[m[1]])[0]
            if c < 1e-10:
                c = 1
            q = prod(q,c)
            for k,m in enumerate(storedGrad):
                beta = rho[k] * dotProduct(m[1],[q])[0]
                q = addProd(q, m[0], alpha[k]-beta)
            #q = opt.prod(q,-1)
        else:
            storedGrad = []
            q = copyDir(grd)
            #opt.reset = False



        grd2 = dotProduct(grd, [grd])[0]
        grdTry = np.sqrt(np.maximum(1e-20,dotProduct(q,[q])[0]))
        dir0 = deepcopy(q)
        if it == 0 or opt.reset:
            t_init = 1. / grdTry
        else:
            t_init = 1.

        grdOld = deepcopy(grd)

        if it == 0 or it == burnIn:
            if gradEps is None:
                gradEps = max(0.001 * np.sqrt(grd2), 0.0001)
            else:
                gradEps = min(gradEps, 0.0001 * np.sqrt(grd2))
            logging.info(f'Gradient threshold: {gradEps:.6f}')

        if it < burnIn:
            Wolfe = False
            eps = epsInit
        else:
            if Wolfe:
                eps = 1.
            else:
                epsBig = epsMax / (grdTry)
                if eps > epsBig:
                    eps = epsBig

        opt.reset = False
        _eps = eps

        ### Starting Line Search
        eps, fc, gc, phi_star, old_fval, gval = line_search(opt, dir0, gfk=grd, old_fval=obj,
                           old_old_fval=obj_old, c1=1e-4, c2=0.9, amax=None, t_init=t_init,
                           maxiter=10)
        if eps is not None:
            diffVar = prod(dir0, -eps)
            obj_old = obj
            opt.acceptVarTry()  #
            obj = phi_star
        else:
            logging.info('Wolfe search unsuccessful')
            if opt.reset:
                logging.info('Cannot go any further')
                opt.converged = False
                if hasattr(opt, 'endOfProcedure'):
                    opt.endOfProcedure()
                elif hasattr(opt, 'endOfIteration'):
                    opt.endOfIteration()
                break
            opt.reset = True


        if (np.fabs(obj-obj_old) < 1e-7) and stopCondition():
            logging.info(f'iteration {it + 1:d}: obj = {obj:.5f}, eps = {eps:.5f}, gradient: {np.sqrt(grd2):.5f}')
            if it > burnIn and opt.reset:
                logging.info('Stopping Gradient Descent: small variation')
                opt.converged = True
                if hasattr(opt, 'endOfProcedure'):
                    opt.endOfProcedure()
                elif hasattr(opt, 'endOfIteration'):
                    opt.endOfIteration()
                break
            else:
                opt.reset = True

        if verb | (it == maxIter):
            if eps is None:
                logging.info(f'iteration {it+1:d}: obj = {obj:.5f}, eps = None, gradient: {np.sqrt(grd2):.5f}')
            else:
                logging.info(f'iteration {it+1:d}: obj = {obj:.5f}, eps = {eps:.5f}, gradient: {np.sqrt(grd2):.5f}')

        if np.sqrt(grd2) <gradEps and stopCondition():
            logging.info('Stopping Gradient Descent: small gradient')
            opt.converged = True
            if hasattr(opt, 'endOfProcedure'):
                opt.endOfProcedure()
            elif hasattr(opt, 'endOfIteration'):
                opt.endOfIteration()
            break
        if eps is not None:
            eps = np.minimum(100*eps, epsMax)
        else:
            eps = epsMax

        if hasattr(opt, 'endOfIteration'):
            opt.endOfIteration()
        it += 1

    if it == maxIter and hasattr(opt, 'endOfProcedure'):
        opt.endOfProcedure()

    if hasattr(opt, 'endOptim'):
        opt.endOptim()

    return opt.getVariable()

