import logging
from scipy.optimize.linesearch import  scalar_search_wolfe2, scalar_search_wolfe1

def line_search_goldstein_price(opt, pk, gfk=None, old_fval=None,
                       old_old_fval=None, c1=0.001, c2=0.9, amax=None,
                       maxiter=100, euclidean = True):
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]


    def phi(alpha):
        fc[0] += 1
        return opt.updateTry(pk, alpha)


    if gfk is None:
        gfk = opt.getGradient(opt.gradCoeff)
    if euclidean:
        derphi0 = -opt.dotProduct_euclidean(gfk, [pk])[0]
    else:
        derphi0 = -opt.dotProduct(gfk, [pk])[0]

    it = 0
    # if amax is None:
    amax = 0
    amin = 0
    if old_fval is None:
        old_fval = phi(0)
    # if old_old_fval is not None:
    #     t = 2*(old_fval - old_old_fval)/derphi0
    # else:
    #     t = 0.01

    t = 1.
    fval = phi(t)
    r = (fval - old_fval) / t
    # logging.info(f'derphi0={derphi0}')
    while it < maxiter:
        if r <  c1*derphi0:
            if r > c2*derphi0:
                break
            else:
                amin = t
        else:
            amax = t
        if amax < 1e-10:
            t *= 2
        else:
            t = (amax+amin)/2
        fval = phi(t)
        r = (fval - old_fval)/t
        it += 1

    # logging.info(f'{t}, {fval}, {old_fval + 2*t * c2 * derphi0}, {old_fval + t * c1 * derphi0}')
    # logging.info(f'Goldstein Price: {it} iterations')

    if it == maxiter:
        logging.warning('Goldstein Price condition: maximum number of iterations attained')
        t = None

    return t, fc[0], gc[0], fval, old_fval, gval[0]


def line_search_weak_wolfe(opt, pk, gfk=None, old_fval=None,
                           old_old_fval=None, c1=0.001, c2=0.9, amax=None,
                           maxiter=100, euclidean=True, t_init = 1.):
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha):
        fc[0] += 1
        return opt.updateTry(pk, alpha)

    def derphi(alpha):
        gc[0] += 1
        gval[0] = opt.getGradient(opt.gradCoeff, update=[pk, alpha])  # store for later use
        gval_alpha[0] = alpha
        if euclidean:
            return -opt.dotProduct_euclidean(gval[0], [pk])[0]
        else:
            return -opt.dotProduct(gval[0], [pk])[0]

    if gfk is None:
        gfk = opt.getGradient(opt.gradCoeff)
    if euclidean:
        derphi0 = -opt.dotProduct_euclidean(gfk, [pk])[0]
    else:
        derphi0 = -opt.dotProduct(gfk, [pk])[0]

    it = 0
    itGrad = 0
    amax = 0
    amin = 0
    if old_fval is None:
        old_fval = phi(0)
    t = t_init
    maxiter2 = 100
    # if old_old_fval is not None:
    #     t = 2*(old_fval - old_old_fval)/derphi0
    # else:
    #     t = 0.01
    # if amax is None:
    #     amax = 1
    # amin = 0
    # t = (amax + amin) / 2
    # if old_fval is None:
    #     old_fval = phi(0)

    # df = derphi(t)
    fval = phi(t)
    while it < maxiter2 and itGrad < maxiter:
        armijo = False
        if fval < old_fval + c1 * t * derphi0:
            df = derphi(t)
            armijo = True
            itGrad += 1
            if df > c2 * derphi0:
                break
            else:
                amin = t
        else:
            amax = t
        if amax < 1e-16:
            t *= 2
        else:
            if armijo:
                t = 0.5*amax+0.5*amin
            else:
                tt = -(t**2 * derphi0) / (2*(fval - old_fval - t * derphi0))
                t = max(0.9*amin + 0.1*amax, min(0.5*amax+0.5*amin, tt))
                #logging.info(f'{t} {tt}')
        fval = phi(t)
        it += 1

    if itGrad == maxiter:
        logging.warning('Weak Wolfe condition: maximum gradient computations attained: switching to backtracking')
        while it < maxiter2:
            armijo = False
            if fval < old_fval + c1 * t * derphi0:
                df = derphi(t)
                break
            else:
                t /= 2
            fval = phi(t)
            it += 1

    if it == maxiter2:
        logging.warning('Weak Wolfe condition: maximum number of iterations attained')
        t = None

    #logging.info(f'Objective after line search {fval:.4f}, {opt.obj:.4f}, {opt.objTry:.4f}')
    return t, fc[0], gc[0], fval, old_fval, gval[0]


def line_search_wolfe(opt, pk, gfk=None, old_fval=None,
                       old_old_fval=None, c1=0.001, c2=0.9, amax=None,
                       maxiter=100, euclidean = True):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.
    """

    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]


    def phi(alpha):
        fc[0] += 1
        return opt.updateTry(pk, alpha)


    def derphi(alpha):
        gc[0] += 1
        gval[0] = opt.getGradient(opt.gradCoeff, update=[pk, alpha])  # store for later use
        gval_alpha[0] = alpha
        if euclidean:
            return -opt.dotProduct_euclidean(gval[0], [pk])[0]
        else:
            return -opt.dotProduct(gval[0], [pk])[0]


    if gfk is None:
        gfk = opt.getGradient(opt.gradCoeff)
    if euclidean:
        derphi0 = -opt.dotProduct_euclidean(gfk, [pk])[0]
    else:
        derphi0 = -opt.dotProduct(gfk, [pk])[0]



    # alpha_star, phi_star, old_fval = scalar_search_wolfe1(
    #     phi, derphi, old_fval, old_old_fval, derphi0, c1, c2)
    # # if alpha_star is None:
    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
        phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
        None, maxiter=maxiter)

    # if derphi_star is None:
    #     warn('The line search algorithm did not converge', LineSearchWarning)
    # else:
    #     # derphi_star is a number (derphi) -- so use the most recently
    #     # calculated gradient used in computing it derphi = gfk*pk
    #     # this is the gradient at the next step no need to compute it
    #     # again in the outer loop.
    #     derphi_star = gval[0]


    return alpha_star, fc[0], gc[0], phi_star, old_fval, gval[0]
