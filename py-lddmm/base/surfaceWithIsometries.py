from .conjugateGradient import cg
from .bfgs import bfgs
from . import pointEvolution as evol
from .surfaceMatching import SurfaceMatching, Control, State
from .surfaces import Surface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .affineBasis import *


## Main class for surface matching
#        Template: sequence of surface classes (from surface.py); if not specified, opens files in fileTemp
#        Target: sequence of surface classes (from surface.py); if not specified, opens files in fileTarg
#        par: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        regWeightOut: multiplicative constant on background regularization
#        affineWeight: multiplicative constant on affine regularization
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        mu: initial value for quadratic penalty normalization
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        typeConstraint: 'stiched', 'sliding', 'slidingV2'
#        affine: 'affine', 'euclidean' or 'none'
#        maxIter_cg: max iterations in conjugate gradient
#        maxIter_al: max interation for augmented lagrangian

class SurfaceWithIsometries(SurfaceMatching):
    def __init__(self, Template=None, Target=None, options = None):
        super().__init__(Template=Template, Target=Target, options=options)

        if hasattr(self.fv0, 'edges') == False:
            self.fv0.getEdges()

                


            #self.I0 = np.mat(self.I0)

        #self.x0 = self.fv0.vertices


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['Isometries'] = None
        options['centerRadius'] = None
        options['maxIter_grad'] = 1000
        options['maxIter_al'] = 100
        options['mu'] = 1.0
        self.converged = False
        return options

    def initialize_variables(self):
        super().initialize_variables()
        if self.options['Isometries'] == None:
            if self.options['centerRadius'] == None:
                print('Isometries must be specified')
                return
            else:
                self.options['Isometries'] = []
                x0 = self.options['centerRadius'][0:3]
                r = self.options['centerRadius'][3]
                for edg in self.fv0.edges:
                    #print max( ((self.fv0.vertices[edg[0], :] - x0)**2).sum(), ((self.fv0.vertices[edg[1], :] - x0)**2).sum()) 
                    if (((self.fv0.vertices[edg[0], :] - x0)**2).sum() > r*r) | (((self.fv0.vertices[edg[1], :] - x0)**2).sum() > r*r):
                        self.options['Isometries'].append([edg[0], edg[1]])

        Isometries = self.options['Isometries']
    
        self.c = []
        for u in Isometries:
            if (u in self.fv0.edges):
                self.c.append(u)
            else:
                if (u.reverse() in self.fv0.edges):
                    self.c.append(u.reverse())
    
        self.c = np.array(self.c)
        self.nconstr = self.c.shape[0]
        print('Number of constrained edges: ', self.c.shape[0], 'out of', len(self.fv0.edges))
        self.I0 = np.zeros([self.fv0.vertices.shape[0], self.c.shape[0]])
        # self.I1 = np.zeros(self.fv0.vertices.shape[0], self.c.shape[0])
    
        for k in range(self.c.shape[0]):
            self.I0[self.c[k, 0]][k] = 1
            self.I0[self.c[k, 1]][k] = -1
        self.cval = np.zeros([self.Tsize+1, self.c.shape[0]])
        if self.nconstr > 0:
            self.ntau0 = np.sqrt(np.power(self.control['x0'][self.c[:,0], :] - self.control['x0'][self.c[:,1], :],2).sum(axis=1))
        #self.lmb = 10*np.multiply(np.random.randn(self.Tsize, 1), np.ones([self.Tsize, self.c.shape[0]]))
        self.lmb = np.ones([self.Tsize+1, self.c.shape[0]])
        self.mu = self.options['mu']

        self.color=np.ones(self.fv0.vertices.shape[0])
        u = np.fabs(self.I0).sum(axis=1)
        for kk in range(self.fv0.vertices.shape[0]):
            if u[kk] > 0:
                self.color[kk] = 2

    def initial_plot(self):
        self.fv0.saveVTK(self.outputDir+'/Template.vtk',
        scalars=self.color, scal_name='constraints')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')

    def constraintTerm(self, xt):
        obj = 0
        timeStep = 1.0/self.Tsize
        c = self.c
        cval = np.zeros(self.cval.shape)
        for t in range(self.Tsize+1):
            z = np.squeeze(xt[t, :, :]) 
            ntau = np.sqrt(np.power(z[c[:,0], :] - z[c[:,1], :],2).sum(axis=1))
            cval[t, :] = np.divide(ntau, self.ntau0) - 1
            obj += timeStep * (- np.multiply(self.lmb[t, :], cval[t,:]).sum() 
                               + np.multiply(cval[t, :], cval[t, :]).sum()/(2*self.mu))
        return obj, cval

    def constraintTermGrad(self, xt):
        c = self.c
        I0 = self.I0
        lmb = np.zeros(self.cval.shape)
        dxcval = np.zeros(self.xt.shape)

        for t in range(self.Tsize+1):
            z = np.squeeze(xt[t, :, :])
            tau = z[c[:,0], :] - z[c[:,1], :]
            ntau = np.sqrt(np.power(tau,2).sum(axis=1))
            lmb[t, :] = self.lmb[t, :] - (np.divide(ntau , self.ntau0)-1)/self.mu
            tau = np.divide(tau, (ntau*self.ntau0).reshape([self.nconstr, 1]))
            ltau = np.multiply(lmb[t, :].reshape([tau.shape[0], 1]), tau)
            dxcval[t, :, :] = np.dot(self.I0, ltau)
        return lmb, dxcval

    def testConstraintTerm(self, xt):
        xtTry = []
        eps = 0.00000001
        xtTry = xt + eps*np.random.randn(self.Tsize+1, self.npt, self.dim)

        u0 = self.constraintTerm(xt)
        ux = self.constraintTerm(xtTry)
        [l, dx] = self.constraintTermGrad(xt)
        vx = np.multiply(dx, xtTry-xt).sum()/eps
        print('Testing constraints:')
        print('var x:', self.Tsize*(ux[0]-u0[0])/(eps), -vx)

    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian = False):
        f = super().objectiveFunDef(control, var = var, withTrajectory=True, withJacobian=withJacobian)
        # cstr = self.constraintTerm(f[1])
        # obj = f[0]+cstr[0]
        # timeStep = 1.0/control['at'].shape[0]
        # Jt = None
        # if self.affineDim > 0:
        #     dim2 = self.dim ** 2
        #     A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, control['Afft'][t]) 
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
        # else:
        #     A = None
        #     
        # st = State()
        # if withJacobian:
        #     xJ  = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, control['at'], self.options['KparDiff'], 
        #                                             withJacobian = True, affine=A)
        #     st['xt'] = xJ[0]
        #     st['Jt'] = xJ[1]
        # else:
        #     st['xt']  = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, control['at'], self.options['KparDiff'], 
        #                                                   affine=A)
        # 
        # foo = Surface(surf=self.fv0)
        # obj=0
        # for t in range(control['at'].shape[0]):
        #     z = st['xt'][t, :, :] 
        #     a = st['at'][t, :, :] 
        #     ra = self.options['KparDiff'].applyK(z, a)
        #     obj += self.options['regWeight']*timeStep*np.multiply(a, ra).sum()
        #     if self.options['internalCost']:
        #         foo.updateVertices(z)
        #         obj += self.options['internalWeight']*self.options['regWeight']*self.options['internalCost'](foo, ra)*timeStep
        #     if self.affineDim > 0:
        #         obj +=  timeStep * (self.affineWeight.reshape(control['Afft'][t].shape) * control['Afft'][t]**2).sum()
        # 
        cstr = [[], []]
        obj = f[0]
        if self.nconstr > 0:
            cstr = self.constraintTerm(f[1])
            obj += cstr[0]
        
            #print xt.sum(), at.sum(), obj
        if withJacobian or withTrajectory:
            return obj, f[1], cstr[1]
        else:
            return obj

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError']**2)
            (self.obj, self.state, self.cval) = self.objectiveFunDef(self.control, withTrajectory=True)
            self.fvDef.updateVertices(self.state['xt'][self.Tsize, :, :])
            self.obj += self.obj0 + self.fun_obj(self.fvDef, self.fv1) / (self.options['sigmaError']**2)

        return self.obj

    def getVariable(self):
        return self.control

    def updateTry(self, dr, eps, objRef=None):
        objTry = self.obj0
        controlTry = Control()
        for k in dr.keys():
            if dr[k] is not None:
                controlTry[k] = self.control[k] - eps * dr[k]
        # atTry = self.at - eps * dir.diff
        # AfftTry = self.Afft - eps * dir.aff
        obj_, st, cval = self.objectiveFunDef(controlTry, withTrajectory=True)
        objTry += obj_
        
        ff = Surface(surf=self.fvDef)
        ff.updateVertices(st['xt'][-1, :, :])
        objTry +=  self.fun_obj(ff, self.fv1) / (self.options['sigmaError']**2)
        #self.fvDef.vertices = ff

        if (objRef is None) | (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry
            self.cval = cval

        return objTry

    # def acceptVarTry(self):
    #     self.obj = self.objTry
    #     self.at = np.copy(self.atTry)
    #     self.Afft = np.copy(self.AfftTry)
    #     #print self.at

    def covectorEvolution(self, control, px1):
        N = self.npt
        dim = self.dim
        M = self.Tsize
        timeStep = 1.0/M
        dim2 = self.dim**2

        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
        affine = self.affB.getTransforms(control['Afft'])
        xt = evol.landmarkDirectEvolutionEuler(control['x0'], control['at'], 
                                               self.options['KparDiff'], affine=affine)

        #### Restart here
        pxt = np.zeros([M, N, dim])
        if self.nconstr > 0:
            (lmb, dxcval) = self.constraintTermGrad(xt)
            pxt[M-1, :, :] = px1 + dxcval[M] *timeStep
        else:
            pxt[M-1, :, :] = px1
        # print c
        foo = Surface(surf=self.fv0)
        for t in range(M-1):
            px = pxt[M-t-1, :, :]
            z = xt[M-t-1, :, :]
            a = control['at'][M-t-1, :, :]
            if self.nconstr > 0:
                zpx = np.copy(dxcval[M-t-1])
            else:
                zpx = np.zeros(px1.shape)

            #a1 = np.concatenate((px[np.newaxis, ...], a[np.newaxis, ...], -2 * self.options['regWeight'] * a[np.newaxis, ...]))
            #a2 = np.concatenate((a[np.newaxis, ...], px[np.newaxis, ...], a[np.newaxis, ...]))
            #a1 = [px, a, -2*self.options['regWeight']*a]
            #a2 = [a, px, a]
            foo.updateVertices(z)
            v = self.options['KparDiff'].applyK(z, a)
            if self.options['internalCost']:
                grd = self.options['internalCostGrad'](foo, v)
                Lv = grd[0]
                DLv = self.options['internalWeight'] * self.options['regWeight'] * grd[1]
                #                Lv = -2*foo.laplacian(v)
                #                DLv = self.options['internalWeight']*foo.diffNormGrad(v)
                # a1 = np.concatenate((px[np.newaxis, ...], a[np.newaxis, ...], -2 * self.options['regWeight'] * a[np.newaxis, ...],
                #                      -self.options['internalWeight'] * self.options['regWeight'] * a[np.newaxis, ...], Lv[np.newaxis, ...]))
                # a2 = np.concatenate((a[np.newaxis, ...], px[np.newaxis, ...], a[np.newaxis, ...], Lv[np.newaxis, ...],
                #                      -self.options['internalWeight'] * self.options['regWeight'] * a[np.newaxis, ...]))
                zpx += self.options['KparDiff'].applyDiffKT(z, px, a, regweight=self.options['regWeight'],lddmm=True,
                                                            extra_term=self.options['internalWeight']*self.options['regWeight']*Lv) - DLv
            else:
                # a1 = np.concatenate((px[np.newaxis, ...], a[np.newaxis, ...], -2 * self.options['regWeight'] * a[np.newaxis, ...]))
                # a2 = np.concatenate((a[np.newaxis, ...], px[np.newaxis, ...], a[np.newaxis, ...]))
                zpx += self.options['KparDiff'].applyDiffKT(z, px, a, regweight=self.options['regWeight'], lddmm=True)
            #zpx += self.options['KparDiff'].applyDiffKT(z, a1, a2)
            if affine is not None:
                zpx += np.dot(px, affine[0][M-t-1])
            pxt[M-t-2, :, :] = np.squeeze(pxt[M-t-1, :, :]) + timeStep * zpx

        return pxt, xt


    def HamiltonianGradient(self, px1, control=None, getCovector = False):
        if control is None:
            control = self.control
        (pxt, xt) = self.covectorEvolution(control, px1)
        dat = np.zeros(control['at'].shape)
        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        # dim2 = self.dim**2
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
        affine = self.affB.getTransforms(control['Afft'])
        if affine is not None:
            dA = np.zeros(affine[0].shape)
            db = np.zeros(affine[1].shape)
        foo = Surface(surf=self.fv0)
        for t in range(self.Tsize):
            z = xt[t,...]
            a = control['at'][t, :, :]
            px = pxt[t, :, :]
            foo.updateVertices(z)
            v = self.options['KparDiff'].applyK(z,a)
            if self.options['internalCost']:
                Lv = self.options['internalCostGrad'](foo, v, variables='phi')
                #Lv = -foo.laplacian(v)
                dat[t, :, :] = 2*self.options['regWeight']*a-px + self.options['internalWeight'] * self.options['regWeight'] * Lv
            else:
                dat[t, :, :] = 2*self.options['regWeight']*a-px
            #dat[t, :, :] = (2*self.options['regWeight']*a-px)
            if affine is not None:
                dA[t] = np.dot(pxt[t].T, xt[t])
                db[t] = pxt[t].sum(axis=0)

        if affine is not None:
            if getCovector == False:
                return dat, dA, db, xt
            else:
                return dat, dA, db, xt, pxt
        else:
            if getCovector == False:
                return dat, xt
            else:
                return dat, xt, pxt


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            endPoint = self.fvDef
            A = self.affB.getTransforms(self.control['Afft'])
        else:
            control = Control()
            for k in update[0].keys():
                if update[0][k] is not None:
                    control[k] = self.control[k] - update[1]*update[0][k]
            A = self.affB.getTransforms(control['Afft'])
            xt = evol.landmarkDirectEvolutionEuler(control['x0'], control['at'], self.options['KparDiff'], affine=A)
            endPoint = Surface(surf=self.fv0)
            endPoint.updateVertices(xt[-1, :, :])
        


        px1 = -self.endPointGradient()
        foo = self.HamiltonianGradient(px1, control=control)
        grd = Control()
        grd['at'] = foo[0]/(coeff*self.Tsize)
        if self.affineDim > 0:
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            dim2 = self.dim ** 2
            dA = foo[1]
            db = foo[2]
            #dAfft = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft)
            grd['Afft'] = 2 * self.control['Afft']
            for t in range(self.Tsize): 
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd['Afft'][t] -=  np.divide(dAff.reshape(grd['Afft'][t].shape), self.affineWeight.reshape(grd['Afft'][t].shape))
            grd['Afft'] /= (coeff*self.Tsize)
            #            dAfft[:,0:self.dim**2]/=100
        return grd
        

    def endOfIteration(self, forceSave=False):
        self.iter += 1
        (obj1, self.xt, Jt, self.cval) = self.objectiveFunDef(self.control, withTrajectory=True,
                                                              withJacobian=True)
        #self.testConstraintTerm(self.xt)
        if self.nconstr > 0:
            print('mean constraint', np.sqrt((self.cval**2).sum()/self.cval.size),
                  np.fabs(self.cval).sum() / self.cval.size)
        a0, foo = self.fv0.computeVertexArea()
        for kk in range(self.Tsize+1):
            #print self.xt[kk, :, :]
            self.fvDef.updateVertices(self.state['xt'][kk, :, :])
            ak, foo = self.fvDef.computeVertexArea()
            JJ = np.log(np.maximum(1e-10, np.divide(ak,a0+1e-10)))
            #print ak.shape, a0.shape, JJ.shape
            self.fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk', scalars = JJ.flatten(),
                               scal_name='Jacobian')
        if self.iter%10 == 0:
            if self.pplot:
                fig=plt.figure(4)
                #fig.clf()
                ax = Axes3D(fig)
                lim0 = self.addSurfaceToPlot(self.fv1, ax, ec = 'k', fc = 'b')
                lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
                ax.set_xlim(min(lim0[0][0],lim1[0][0]), max(lim0[0][1],lim1[0][1]))
                ax.set_ylim(min(lim0[1][0],lim1[1][0]), max(lim0[1][1],lim1[1][1]))
                ax.set_zlim(min(lim0[2][0],lim1[2][0]), max(lim0[2][1],lim1[2][1]))
                plt.pause(0.1)


    def optimizeMatching(self):
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = np.sqrt(grd2) / 1000
        self.muEps = 1.0
        it = 0

        while (self.muEps > 0.005) & (it<self.options['maxIter_al'])  :
            print('Starting Minimization: gradEps = ', self.gradEps, ' muEps = ', self.muEps, ' mu = ', self.mu)
            if self.options['algorithm'] == 'cg':
                cg(self, verb = self.options['verb'], maxIter = self.options['maxIter_grad'],
                      TestGradient=self.options['testGradient'], epsInit=.01,
                      Wolfe=self.options['Wolfe'])
            elif self.options['algorithm'] == 'bfgs':
                bfgs(self, verb = self.options['verb'], maxIter = self.options['maxIter_grad'],
                      TestGradient=self.options['testGradient'], epsInit=1.,
                      Wolfe=self.options['Wolfe'], lineSearch=self.options['lineSearch'], memory=50)
            if self.nconstr == 0:
                break
            for t in range(self.lmb.shape[0]):
                self.lmb[t, :] -= self.cval[t, :]/self.mu
            print('mean lambdas', np.fabs(self.lmb).sum() / self.lmb.size)
            if self.converged:
                self.gradEps *= .75
                if (((self.cval**2).sum()/self.cval.size) > self.muEps**2):
                    self.mu *= 0.5
                else:
                    self.muEps = self.muEps /2
            else:
                self.mu *= 0.9
            self.obj = None
            it = it+1

            
        return self.fvDef

