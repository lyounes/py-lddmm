import numpy as np
import scipy as sp
import os

def read3DVector(filename):
    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline()
            N = int(ln0[0])
            #print 'reading ', filename, ':', N, ' landmarks'
            v = np.zeros([N, 3])

            for i in range(N):
                ln = fn.readline()
                ln0 = fn.readline().split()
                #print ln0
                for k in range(3):
                    v[i,k] = float(ln0[k])
                
    except IOError:
        print 'cannot open ', filename
        raise
    return v




def loadlmk(filename, dim=3):
# [x, label] = loadlmk(filename, dim)
# Loads 3D landmarks from filename in .lmk format.
# Determines format version from first line in file
#   if version number indicates scaling and centering, transform coordinates...
# the optional parameter s in a 3D scaling factor

    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline()
            versionNum = 1
            versionStrs = ln0.split("-")
            if len(versionStrs) == 2:
                try:
                    versionNum = int(float(versionStrs[1]))
                except:
                    pass

            #print fn
            ln = fn.readline().split()
            #print ln0, ln
            N = int(ln[0])
            #print 'reading ', filename, ':', N, ' landmarks'
            x = np.zeros([N, dim])
            label = []

            for i in range(N):
                ln = fn.readline()
                label.append(ln) 
                ln0 = fn.readline().split()
                #print ln0
                for k in range(dim):
                    x[i,k] = float(ln0[k])
            if versionNum >= 6:
                lastLine = ''
                nextToLastLine = ''
                # read the rest of the file
                # the last two lines contain the center and the scale variables
                while 1:
                    thisLine = fn.readline()
                    if not thisLine:
                        break
                    nextToLastLine = lastLine
                    lastLine = thisLine
                    
                centers = nextToLastLine.rstrip('\r\n').split(',')
                scales = lastLine.rstrip('\r\n').split(',')
                if len(scales) == dim and len(centers) == dim:
                    if scales[0].isdigit and scales[1].isdigit and scales[2].isdigit and centers[0].isdigit and centers[1].isdigit and centers[2].isdigit:
                        x[:, 0] = x[:, 0] * float(scales[0]) + float(centers[0])
                        x[:, 1] = x[:, 1] * float(scales[1]) + float(centers[1])
                        x[:, 2] = x[:, 2] * float(scales[2]) + float(centers[2])
                
    except IOError:
        print 'cannot open ', filename
        raise
    return x, label




def  savelmk(x, filename):
# savelmk(x, filename)
# save landmarks in .lmk format.

    with open(filename, 'w') as fn:
        str = 'Landmarks-1.0\n {0: d}\n'.format(x.shape[0])
        fn.write(str)
        for i in range(x.shape[0]):
            str = '"L-{0:d}"\n'.format(i)
            fn.write(str)
            str = ''
            for k in range(x.shape[1]):
                str = str + '{0: f} '.format(x[i,k])
            str = str + '\n'
            fn.write(str)
        fn.write('1 1 \n')

        
# Saves in .vtk format
def savePoints(fileName, x, vector=None, scalars=None):
    if x.shape[1] <3:
        x = np.concatenate((x, np.zeros((x.shape[0],3-x.shape[1]))), axis=1)
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n') 
        fvtkout.write('\nPOINTS {0: d} float'.format(x.shape[0]))
        for ll in range(x.shape[0]):
            fvtkout.write('\n{0: f} {1: f} {2: f}'.format(x[ll,0], x[ll,1], x[ll,2]))
        if vector is None and scalars is None:
            return
        fvtkout.write(('\nPOINT_DATA {0: d}').format(x.shape[0]))
        if scalars is not None:
            fvtkout.write('\nSCALARS scalars float 1\nLOOKUP_TABLE default')
            for ll in range(x.shape[0]):
                fvtkout.write('\n {0: .5f} '.format(scalars[ll]))

        if vector is not None:
            fvtkout.write('\nVECTORS vector float')
            for ll in range(x.shape[0]):
                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vector[ll, 0], vector[ll, 1], vector[ll, 2]))

        fvtkout.write('\n')

# Saves in .vtk format
def saveTrajectories(fileName, xt):
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nCurves \nASCII\nDATASET POLYDATA\n') 
        fvtkout.write('\nPOINTS {0: d} float'.format(xt.shape[0]*xt.shape[1]))
        if xt.shape[2] == 2:
            xt = np.concatenate(xt, np.zeros([xt.shape[0],xt.shape[1], 1]))
        for t in range(xt.shape[0]):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(xt[t,ll,0], xt[t,ll,1], xt[t,ll,2]))
        nlines = (xt.shape[0]-1)*xt.shape[1]
        fvtkout.write('\nLINES {0:d} {1:d}'.format(nlines, 3*nlines))
        for t in range(xt.shape[0]-1):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(t*xt.shape[1]+ll, (t+1)*xt.shape[1]+ll))

        fvtkout.write('\n')



def epsilonNet(x, rate):
    #print 'in epsilon net'
    n = x.shape[0]
    dim = x.shape[1]
    inNet = np.zeros(n, dtype=int)
    inNet[0]=1
    net = np.nonzero(inNet)[0]
    survivors = np.ones(n, dtype=np.int)
    survivors[0] = 0 ;
    dist2 = ((x.reshape([n, 1, dim]) -
              x.reshape([1,n,dim]))**2).sum(axis=2)
    d2 = np.sort(dist2, axis=0)
    i = np.int_(1.0/rate)
    eps2 = (np.sqrt(d2[i,:]).sum()/n)**2
    #print n, d2.shape, i, np.sqrt(eps2)
    

    i1 = np.nonzero(dist2[net, :] < eps2)
    survivors[i1[1]] = 0
    i2 = np.nonzero(survivors)[0]
    while len(i2) > 0:
        closest = np.unravel_index(np.argmin(dist2[net.reshape([len(net),1]), i2.reshape([1, len(i2)])].ravel()), [len(net), len(i2)])
        inNet[i2[closest[1]]] = 1 
        net = np.nonzero(inNet)[0]
        i1 = np.nonzero(dist2[net, :] < eps2)
        survivors[i1[1]] = 0
        i2 = np.nonzero(survivors)[0]
        #print len(net), len(i2)
    idx = - np.ones(n, dtype=np.int)
    for p in range(n):
        closest = np.unravel_index(np.argmin(dist2[net, p].ravel()), [len(net), 1])
        #print 'p=', p, closest, len(net)
        idx[p] = closest[0]
        
        #print idx
    return net, idx


def L2Norm0(x1):
    return (x1**2).sum()

def L2NormDef(xDef, x1):
    return -2*(xDef*x1).sum() + (xDef**2).sum()

def L2NormGradient(xDef,x1):
    return 2*(xDef-x1)

def classScore(xDef, c1, u=None):
    if u is None:
        u = np.ones((xDef.shape[1],1))
    return np.exp(-(np.dot(xDef,u)*c1)).sum()

def classScoreGradient(xDef, c1, u = None):
    if u is None:
        u = np.ones((xDef.shape[1],1))
    return -c1*np.exp(-np.dot(xDef,u)*c1) * u.T

def LogisticScore(xDef, c1, u, w = None, intercept=True):
    if w is None:
        w = np.ones((xDef.shape[0],1))
    if intercept:
        xDef1 = np.concatenate((np.ones((xDef.shape[0], 1)), xDef), axis=1)
        gu = np.dot(xDef1,u)
    else:
        gu = np.dot(xDef, u)
    res = (np.ravel(w) * (- gu[np.arange(gu.shape[0])[:, np.newaxis], c1].sum(axis=1) + np.log(np.exp(gu).sum(axis=1)))).sum()
    return res

def LogisticScoreGradient(xDef, c1, u, w = None, intercept=True):
    if w is None:
        w = np.ones((xDef.shape[0],1))
    if intercept:
        xDef1 = np.concatenate((np.ones((xDef.shape[0], 1)), xDef), axis=1)
        gu = np.exp(np.dot(xDef1, u))
        pu = gu/gu.sum(axis=1)[:,np.newaxis]
        m = np.dot(pu, u[np.arange(1,u.shape[0]),:].T)
        return (-u[np.arange(1,u.shape[0]), c1] +m)*w
    else:
        gu = np.exp(np.dot(xDef, u))
        pu = gu/gu.sum(axis=1)[:,np.newaxis]
        m = np.dot(pu, u.T)
        return (-u[np.arange(u.shape[0]), c1] +m)*w

def LogisticScoreGradientInU(xDef, c1, u, w=None, intercept=True):
    if w is None:
        w = np.ones((xDef.shape[0],1))/xDef.shape[0]
    if intercept:
        xDef1 = np.concatenate((np.ones((xDef.shape[0], 1)), xDef), axis=1)
        gu = np.exp(np.dot(xDef1,u))
        pu = gu/gu.sum(axis=1)[:, np.newaxis]

        wxDef = w*xDef1
        r = np.dot((wxDef).T, pu)
        grad = np.zeros(u.shape)
        for k in range(1, u.shape[1]):
            grad[:,k] = - (wxDef)[np.ravel(c1)==k,:].sum(axis=0) + r[:,k]
    else:
        gu = np.exp(np.dot(xDef, u))
        pu = gu / gu.sum(axis=1)[:, np.newaxis]

        wxDef = w * xDef
        r = np.dot((wxDef).T, pu)
        grad = np.zeros(u.shape)
        for k in range(1, u.shape[1]):
            grad[:, k] = - (wxDef)[np.ravel(c1) == k, :].sum(axis=0) + r[:, k]
    return grad

def learnLogistic(x, y, w=None, u0=None, l1Cost = 0, intercept=True, random = 1.):
    J1 = []
    dim = x.shape[1]
    nclasses = y.max()+1
    for k in range(nclasses):
        J1.append(y == k)
    if u0 is None:
        if intercept:
            fu = np.zeros((dim+1, nclasses))
        else:
            fu = np.zeros((dim, nclasses))
    else:
        fu = u0
    J = np.random.rand(x.shape[0]) < random
    x0 = x[J,:]
    y0 = y[J]
    w0 = w[J]
    for k in range(1000):
        fuOld = fu
        obj0 = LogisticScore(x0, y0, fu, w=w0, intercept=intercept)
        g = LogisticScoreGradientInU(x0, y0, fu, w=w0, intercept=intercept)
        # ep = 1e-8
        # fu1 = fu + ep * g
        # obj1 = LogisticScore(x, y, fu1, w=w)
        # print (obj1-obj0)/ep, (g**2).sum()

        ep = .01
        fu1 = fu - ep * g
        obj1 = LogisticScore(x0, y0, fu1, w=w0, intercept=intercept)
        while obj1 > obj0:
            ep /= 2
            fu1 = fu - ep * g
            obj1 = LogisticScore(x0, y0, fu1, w=w0, intercept=intercept)
        fu = fu1
        ll = ep * l1Cost
        fu = np.sign(fu) * np.maximum(np.fabs(fu) - ll, 0)
        if np.fabs(fu-fuOld).max() < 1e-8:
            break
        #print 'Iteration ', k, ': ',  pointSets.LogisticScore(self.fvDef, self.fv1, fu), ' ep: ', ep
    return fu
