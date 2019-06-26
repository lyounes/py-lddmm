import numpy as np
import logging

# added comment to test git, 4-26-19

# Class running BFGS
# opt is an optimizable class that must provide the following functions:
#   getVariable(): current value of the optimzed variable
#   objectiveFun(): value of the objective function
#   updateTry(direction, step, [acceptThreshold]) computes a temporary variable by moving the current one in the direction 'dircetion' with step 'step'
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

def bfgs(opt, verb = True, maxIter=1000, TestGradient = False, epsInit=0.01, memory=25, Wolfe = True):

    if (hasattr(opt, 'getVariable')==False | hasattr(opt, 'objectiveFun')==False | hasattr(opt, 'updateTry')==False | hasattr(opt, 'acceptVarTry')==False | hasattr(opt, 'getGradient')==False):
        logging.error('Error: required functions are not provided')
        return

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

    burnIn = 20
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
    while it < maxIter:
        if hasattr(opt, 'startOfIteration'):
            opt.startOfIteration()

        try_BFGS = True
        while(try_BFGS):
            if opt.reset:
                opt.obj = None
                obj = opt.objectiveFun()

            grd = opt.getGradient(gradCoeff)

            if TestGradient:
                if hasattr(opt, 'randomDir'):
                    dirfoo = opt.randomDir()
                else:
                    dirfoo = np.random.normal(size=grd.shape)
                epsfoo = 1e-8
                objfoo = opt.updateTry(dirfoo, epsfoo, obj-1e10)
                [grdfoo] = dotProduct(grd, [dirfoo])
                logging.info('Test Gradient: %.4f %.4f' %((objfoo - obj)/epsfoo, -grdfoo * gradCoeff ))

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
            dir0 = copyDir(q)

            grdOld = copyDir(grd)

            if Wolfe:
                eps *= 2.
                if eps > 1.:
                    eps = 1.
            else:
                epsBig = epsMax / (grdTry)
                if eps > epsBig:
                    eps = epsBig

            objTry = opt.updateTry(dir0, eps, obj)

            if objTry > obj:
                #fprintf(1, 'iteration %d: obj = %.5f, eps = %.5f\n', it, objTry, eps) ;
                epsSmall = np.maximum(1e-6/(grdTry), epsMin)
                #print 'Testing small variation, eps = {0: .10f}'.format(epsSmall)
                objTry0 = opt.updateTry(dir0, epsSmall, obj)
                if objTry0 > obj:
                    if opt.reset:
                        logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, gradient: {3:.5f}'.format(it+1, obj, eps, np.sqrt(grd2)))
                        logging.info('Stopping Gradient Descent: bad direction')
                        break
                    else:
                        logging.info('Resetting BFGS')
                        opt.reset = True
                else:
                    try_BFGS = False
            else:
                try_BFGS = False


        opt.reset = False



        ### Starting Line Search

        if Wolfe:
            #var = opt.getVariable()
            d1 = dotProduct(dir0, [grd])[0]
            c1 = 0.1
            c2 = 0.5
            lb1 = obj - c1*eps*d1
            while (objTry > lb1):
                eps /= 1.5
                lb1 = obj - c1 * eps * d1
                objTry = opt.updateTry(dir0, eps, obj)
            lb2 = c2*d1
            #varTry = opt.update(dir0, -eps)
            grdTry = opt.getGradient(coeff=gradCoeff, update=[dir0, eps])
            d2 = dotProduct(dir0, [grdTry])[0]
            while (objTry > lb1) or (abs(d2) < abs(lb2)):
                eps /= 1.5
                lb1 = obj - c1 * eps * d1
                objTry = opt.updateTry(dir0, eps, obj)
                #varTry = addProd(var, dir0, -eps)
                grdTry = opt.getGradient(gradCoeff, update=[dir0,eps])
                d2 = dotProduct(dir0, [grdTry])[0]
        else:
            while (objTry > obj) and (eps > epsMin):
                eps = eps / 2
                objTry = opt.updateTry(dir0, eps, obj)
                        #opt.acceptVarTry()

                        #print 'improve'
            ## reducing step if improves
            contt = 1
            while contt==1:
                objTry2 = opt.updateTry(dir0, .5*eps, objTry)
                if objTry > objTry2:
                    eps = eps / 2
                    objTry=objTry2
                else:
                    contt=0


        # increasing step if improves
            contt = 5
            #eps0 = eps / 4
            while contt>=1 and eps<epsBig:
                objTry2 = opt.updateTry(dir0, 1.25*eps, objTry)
                if objTry > objTry2:
                    eps *= 1.25
                    objTry=objTry2
                    #contt -= 1
                else:
                    contt=0
        ### end of line search


        #print obj+obj0, objTry+obj0
        if (np.fabs(obj-objTry) < 1e-6) and stopCondition():
            logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, gradient: {3:.5f}'.format(it+1, obj, eps, np.sqrt(grd2)))
            if it > burnIn:
                logging.info('Stopping Gradient Descent: small variation')
                opt.converged = True
                break
            if hasattr(opt, 'endOfIteration'):
                opt.endOfIteration()

        diffVar = prod(dir0, -eps)
        opt.acceptVarTry()
        obj = objTry
        #logging.info('Obj Fun CG: ' + str(opt.objectiveFun(force=True)))
        if verb | (it == maxIter):
            logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, gradient: {3:.5f}'.format(it+1, obj, eps, np.sqrt(grd2)))

        if np.sqrt(grd2) <gradEps and stopCondition():
            logging.info('Stopping Gradient Descent: small gradient')
            opt.converged = True
            if hasattr(opt, 'endOfProcedure'):
                opt.endOfProcedure()
            break
        eps = np.minimum(100*eps, epsMax)

        if hasattr(opt, 'endOfIteration'):
            opt.endOfIteration()
        it += 1

    if it == maxIter and hasattr(opt, 'endOfProcedure'):
        opt.endOfProcedure()

    if hasattr(opt, 'endOptim'):
        opt.endOptim()

    return opt.getVariable()

