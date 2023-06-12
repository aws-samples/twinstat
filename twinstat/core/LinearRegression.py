# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


import torch
import numpy as np
from scipy.optimize import minimize
from itertools import product
from scipy.stats import t as st

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearModel(object):
    '''
    Base linear regression object.


    '''
    def __init__(self, backend:str='torch'):
        '''

        Parameters
        ----------
        backend : str, optional
            The backend solver can be "numpy" or "torch". Pytorch provides
            GPU functionality if needed for large problems.
            The default is 'torch'.

        Returns
        -------
        None.

        '''
        self.backend = backend

    def _expand_matrix(self):
        '''function to be overridden by
        specific model type'''

    def fit(self,X:np.array,y:np.array, beta_relevance:list[float] = None):
        '''

        Fit a linear regression to the data.

        Attributes
        ----------

        mse: mean squared error
        betas: regression coefficients
        betas_se: standard error of the betas
        beta_pvalues: pvalue for significance of betas


        Parameters
        ----------
        X : np.array
            Input data.
        y : np.array
            Output data
        beta_relevance : list[float], optional
            If set, the coefficient pvalues will be
            relative to the beta_relevance value.

            E.g. if t_crit = (beta[0] - beta_relevance[0])/se

            The default is None.

        Returns
        -------
        None.

        '''

        X = np.array(X)
        y = np.array(y)

        self.mean = np.mean(X)
        self.scale = np.std(X)
        X = X - self.mean
        X = X/self.scale

        Xd = self._expand_matrix(X)
        self.dof = Xd.shape[0] - Xd.shape[1]

        #check if there is a zero column
        col_check = np.sum(np.abs(Xd), axis=0)
        if any([x==0. for x in col_check]):
            self.betas = np.zeros((col_check.shape[0],1))
            if self.backend == 'torch':
                self.betas = torch.Tensor(self.betas).double()
            return None

        if self.backend == 'torch':
            #use pytorch to complete the linear algebra
            Xd = torch.Tensor(Xd).double()
            if not device.type =='cpu':
                Xd = Xd.cuda()

            XpX = Xd.T.mm(Xd).double()
            b = torch.eye(XpX.shape[0]).double()
            XpXinv = torch.linalg.solve(XpX, b )
            XpXinvX = XpXinv.mm(Xd.T)

            yp = torch.Tensor(np.expand_dims(y, -1)).double()
            self.betas = XpXinvX.mm(yp)

            self.XpXinv = XpXinv
            self.sse = yp.T.mm(yp) - self.betas.T.mm(Xd.T.mm(yp))
            self.mse = self.sse / self.dof

        else:
            XpX = np.matmul(Xd.T, Xd)
            b = np.eye(XpX.shape[0])
            XpXinv = np.linalg.solve(XpX, b )
            XpXinvX = np.matmul(XpXinv, Xd.T)

            yp = np.expand_dims(y, -1)
            self.betas = np.matmul(XpXinvX, yp)

            self.XpXinv = XpXinv
            self.sse = yp.T @ yp - self.betas.T @ Xd.T @ yp
            self.mse = self.sse / self.dof
        self.mse = np.squeeze(self.mse)
        self.betas_se = np.sqrt(np.diag(self.mse * XpXinv+1e-12))

        # calculate the pvalues of the beta coefficients, but
        # allow the user to provide a number of significance
        if beta_relevance is None:
            rel = np.zeros((Xd.shape[1],))
        else:
            rel = beta_relevance
        self.beta_pvalues = []
        for d in range(Xd.shape[1]):
            tcrit = (self.betas[[d]] - rel[d])/ (self.betas_se[d] + 1e-6)
            pvalue = 1 - st.cdf( tcrit , self.dof)
            self.beta_pvalues.append(np.squeeze(pvalue))


    def predict(self, Xnew:np.array, return_uncertainty:bool=False, uncertainty:str='confidence') -> np.array:
        '''

        Parameters
        ----------
        Xnew : np.array

        return_uncertainty : bool, optional
            If true, return both the prediction and the uncertainty of the prediction.
            The default is False.
        uncertainty : str, optional
            Can be "confidence" or "prediction" intervals.
            The default is 'confidence'.

        Returns
        -------
        prediction: np.array

        If return_uncertainty is True also return the SE of the prediction.

        sigma: np.array


        '''


        Xnew = np.array(Xnew)

        Xnew = Xnew - self.mean
        Xnew = Xnew/self.scale

        Xexp = self._expand_matrix(Xnew)
        if self.backend == 'torch':
            Xexp = torch.Tensor(Xexp).double()
            yhat = Xexp.mm( self.betas )
            yhat = np.ravel(yhat.tolist())

            if uncertainty == 'confidence':
                sigma = np.diag(self.mse * Xexp.mm(self.XpXinv).mm( Xexp.T) )
            else:
                sigma = np.diag(self.mse *( 1 + Xexp.mm(self.XpXinv).mm( Xexp.T) ))

        else:
            yhat = np.matmul( Xexp, self.betas )

            if uncertainty == 'confidence':
                sigma = np.diag(self.mse * Xexp @ self.XpXinv @ Xexp.T)
            else:
                sigma = np.diag(self.mse *( 1 + Xexp @ self.XpXinv @ Xexp.T))

        #convert from variance to std
        sigma = np.sqrt(sigma)

        if return_uncertainty:
            return yhat, sigma
        else:
            return yhat




#--------------------------------------------------------------

class Polynomial(LinearModel):
    def __init__(self, poly_order:int, **kwargs):
        '''
        Sets the parametric function form of the linear regression
        to be a polynomial.

        Parameters
        ----------
        poly_order : int
            Polynomial order.

        Returns
        -------
        None.

        '''

        self.poly_order = poly_order
        LinearModel.__init__(self, **kwargs)

    def _expand_matrix(self, X):
        #X is the design matrix
        Xd = np.ones(X.shape[0], dtype=np.float64)
        Xd = np.expand_dims(Xd, -1)
        for order in range(1,self.poly_order+1):
            new_dim  = np.power(X,order)
            new_dim = np.expand_dims(new_dim, -1)
            Xd = np.hstack([Xd, new_dim])
        return Xd

#--------------------------------------------------------------

class PiecewisePolynomial(LinearModel):
    def __init__(self, poly_order:int, n_knots:int = 1, **kwargs):
        '''
        Sets the parametric function form of the linear regression
        to be a polynomial.  However, add the ability to create
        piecewise knot points that discontinours change slope.
        L0 continuity and not L1 continuity.

        Parameters
        ----------
        poly_order : int
            Polynomial order.
        n_knots : int, optional
             The default is 1.

        Returns
        -------
        None.

        '''
        self.poly_order = poly_order
        self.n_knots = n_knots
        self.knotlst = [0]*n_knots
        LinearModel.__init__(self, **kwargs)

    def _expand_matrix(self, X):
        #X is the design matrix
        Xd = np.ones(X.shape[0], dtype=np.float64)
        Xd = np.expand_dims(Xd, -1)
        for order in range(1,self.poly_order+1):
            new_dim  = np.power(X,order)
            new_dim = np.expand_dims(new_dim, -1)
            Xd = np.hstack([Xd, new_dim])

        for i in range(self.n_knots):
            if isinstance(self.knotlst, np.float64):
                knot = self.knotlst
            else:
                knot = self.knotlst[i]
            knot = (knot - self.mean)/self.scale
            Xidx = np.zeros(X.shape[0])
            idx = np.where(X>knot)
            Xidx[idx] = np.ones(idx[0].shape[0])
            xstar = (X - knot)*Xidx
            new_dim = np.expand_dims(xstar, -1)
            Xd = np.hstack([Xd, new_dim])

        return Xd

    def fit_knots(self,X:np.array, y:np.array ) -> None:
        '''
        Find the optimal location for the knots.

        Parameters
        ----------
        X : np.array
        y : np.array

        Returns
        -------
        None

        '''

        smallest_knot = np.min(X, axis=-1)
        largest_knot = np.max(X, axis=-1)

        def obj_fun(k):
            k = np.clip(k,smallest_knot,largest_knot)
            self.knotlst = k
            self.fit(X,y)
            yhat = self.predict(X)
            mse = np.var(y - yhat)
            return mse

        #first run through a grid of possible knots to serve as initial optimization
        #points that are given to the line search algorithm
        grid = np.linspace(np.quantile(X, 0.05), np.quantile(X, 0.95), num=10)
        if self.n_knots > 1:
            grid = list(product( *[grid]*self.n_knots  ) )
            #all knots must be unique
            grid = [x for x in grid if not self.check_for_duplicates(x)]

        #TODO: need parallel? yes if user wants 6+ knots
        grid_hunt=[]
        for k in grid:
            #k = grid[0]
            grid_hunt.append(obj_fun(k))

        self.grid_hunt=grid_hunt
        knoti = grid[np.argmin(grid_hunt)]
        #knoti = sorted(knoti)
        minimize(obj_fun, knoti, method='bfgs')


    def check_for_duplicates(self, thelist):
        #https://stackoverflow.com/questions/1541797/how-do-i-check-if-there-are-duplicates-in-a-flat-list
        seen = set()
        for x in thelist:
          if x in seen: return True
          seen.add(x)
        return False


#--------------------------------------------------------------

class Exponential(LinearModel):
    def __init__(self, **kwargs):
        '''
        Sets the parametric function form of the linear regression
        to be exponential.

        Parameters
        ----------

        Returns
        -------
        None.

        '''
        LinearModel.__init__(self, **kwargs)

    def _expand_matrix(self, X):
        #X is the design matrix
        Xd = np.ones(X.shape[0], dtype=np.float64)
        Xd = np.expand_dims(Xd, -1)

        new_dim  = np.exp(X)
        new_dim = np.expand_dims(new_dim, -1)
        Xd = np.hstack([Xd, new_dim])

        return Xd




#%% main
if __name__ == '__main__':

    ''' Example.'''


    import matplotlib.pyplot as plot

    plot.close('all')

    n_data = 100
    x = np.linspace(0,50,num=n_data)
    y = 0.5*x[:50] + np.random.normal(0, 5.5, size=int(0.5*n_data))
    y = np.append(y, 0.7*(x[50:]-20)**2  + np.random.normal(0, 5.5, size=int(0.5*n_data)) )

    lm = PiecewisePolynomial(3, n_knots=0, backend='numpy')
    yhat = lm.fit_knots(x,y)
    #lm.knotlst = [25.0]
    #yhat = lm.fit(x,y)

    yhat = lm.predict(x)

    plot.figure(figsize=(14,5))
    plot.scatter(x,y, label='Measurment')
    plot.plot(x,yhat, label='PiecewisePoly')
    plot.legend()
    plot.tight_layout()