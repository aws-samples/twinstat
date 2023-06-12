# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################



import numpy as np
from sklearn.neighbors._regression import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights
from sklearn.neighbors import NearestNeighbors
import pandas
import matplotlib.pyplot as plot
from scipy.signal import find_peaks

class QuantileKNNRegressor(KNeighborsRegressor):
    '''
    Extends the Scikit KNN Regression to enable quantile regressions.

    All of the existing functionality of scikit is supported.
    '''

    def __init__(self,
                 tau: float=0.5,
                 **kwargs):
        '''
        Parameters
        ----------
        tau : float, optional
            The percentile that is used when fitting the data. The default is 0.5.
        **kwargs :
            Send any scikit specific options.

        Returns
        -------
        None.

        '''
        self.tau=tau
        KNeighborsRegressor.__init__(self,**kwargs)


    def predict(self, X: np.array) -> np.array:
        '''

        Parameters
        ----------
        X : np.array
            (n data x n features)

        Returns
        -------
        y_pred : np.array
            (n data x 1)
        '''
        if self.weights == "uniform":
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            #y_pred = np.mean(_y[neigh_ind], axis=1)
            y_pred = np.quantile(_y[neigh_ind], self.tau, axis=1)
        else:
            y_pred = np.empty((neigh_dist.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

#---------------------------------------------------------------------------------------------


class OutlierKNNDetector(NearestNeighbors):
    '''

    Automatically find outliers and remove from the data set.

    '''
    def __init__(self,
                 outlier_distance_threshold:float=None,
                 outlier_percent_threshold:float=None,
                 endog_idx:int=None,
                 removal_iterations:int=10,
                 **kwargs):
        '''

        Method #1 ----
        If outlier_percent_threshold is set to a value between 0 and 1, the
        endog_idx is checked. If None, a KDTree is generated to determine the
        distance between all points.  The outliers are flagged as the points
        above and below the outlier_percent_threshold threshold.

        If endog_idx is not None, then a QuantileKNNRegressor is used to relate
        all of the data to the endog_idx variable.  All data points above and below
        the regressed outlier_percent_threshold of QuantileKNNRegressor will be
        removed from the data set.

        Method #2 ----
        If outlier_distance_threshold is provided a QuantileKNNRegressor will
        be used to regress the endog_idx on to the remaining data.  The residuals
        of prediction are calculated, i.e. r = y - yhat.

        All data points exhibiting residuals larger than outlier_distance_threshold
        will be flagged as outliers (i.e. they are not following the behavior of
        surrounding data points.) This process is carried out removal_iterations
        number of times.

        Parameters
        ----------
        outlier_distance_threshold : float, optional
            The default is None.
        outlier_percent_threshold : float, optional
            The default is None.
        endog_idx : int, optional
            Index of the response variable.
            The default is None.
        removal_iterations : int, optional
            Small clusters of outliers with varying magnitudes may
            only flag the most extreme outlier.  Repeating the method
            ensures surrounding outliers are also captured.
            The default is 10.
        **kwargs : scikit arguments


        Returns
        -------
        None.

        '''
        NearestNeighbors.__init__(self,**kwargs)

        if outlier_percent_threshold is not None:
            if outlier_percent_threshold < 0.0 or outlier_percent_threshold > 1.0:
                raise ValueError("ERROR: outlier_percent_threshold must be between 0 and 1")
            #ensure it is always upper since we are going to bracket it in the outlier check
            if outlier_percent_threshold < 0.5:
                outlier_percent_threshold = 1 - outlier_percent_threshold

        if outlier_distance_threshold is not None:
            if outlier_distance_threshold < 0.0 :
                raise ValueError("ERROR: outlier_distance_threshold must be greater than 0.0")

        if outlier_distance_threshold is not None and outlier_percent_threshold is not None:
            raise ValueError("ERROR: Cannot provide both outlier_distance_threshold "
                             +"and outlier_percent_threshold.")

        self.outlier_distance_threshold = outlier_distance_threshold
        self.outlier_percent_threshold = outlier_percent_threshold
        self.endog_idx = endog_idx
        self.removal_iterations = removal_iterations

    def remove_outliers(self, X: np.array, make_plot:bool=True) -> np.array:
        '''

        Parameters
        ----------
        X : np.array
        make_plot : bool, optional
            The default is True.

        Returns
        -------
        newX : np.array

        '''
        X = np.array(X)

        pass_back_endog = False
        if self.endog_idx is not None and len(X.shape) == 1:
            self.endog_idx=1
            t = np.linspace(0,X.shape[0],num=X.shape[0])
            X = np.vstack((t,X)).T
            pass_back_endog=True

        if self.outlier_percent_threshold is not None:
            self.fit(X)
            distances, indexes = self.kneighbors(X)
            distances= np.mean(distances,axis = 1)

            if self.endog_idx is None:
                qlb = np.quantile(distances, 1-self.outlier_percent_threshold)
                qub = np.quantile(distances, self.outlier_percent_threshold)
                idx = np.where((distances > qub) | (distances < qlb))
            else:
                qnn_ub = QuantileKNNRegressor(tau=self.outlier_percent_threshold,
                                              n_neighbors=self.n_neighbors)
                qnn_lb = QuantileKNNRegressor(tau=1-self.outlier_percent_threshold,
                                              n_neighbors=self.n_neighbors)

                y = X[:,self.endog_idx]
                if X.shape[1] > 2 :
                    Xsub = np.delete(X, self.endog_idx, axis=1)
                    qnn_ub.fit(Xsub, y)
                    qnn_lb.fit(Xsub, y)
                else:
                    idx = 1 if self.endog_idx == 0 else 0
                    Xsub = X[:,idx].reshape(-1,1)
                    qnn_ub.fit(Xsub, y)
                    qnn_lb.fit(Xsub, y)
                qub = qnn_ub.predict(Xsub)
                qlb = qnn_lb.predict(Xsub)
                idx = np.where((y > qub) | (y < qlb))

            newX = np.delete(X, idx, axis=0)
        else:
            if self.outlier_distance_threshold is None:
                self.outlier_distance_threshold=0
            qnn = QuantileKNNRegressor(tau=0.5,
                                       n_neighbors=self.n_neighbors)
            y = X[:,self.endog_idx]
            if X.shape[1] > 2 :
                Xsub = np.delete(X, self.endog_idx, axis=1)
                qnn.fit(Xsub, y)
            else:
                idx = 1 if self.endog_idx == 0 else 0
                Xsub = X[:,idx].reshape(-1,1)
                qnn.fit(Xsub, y)
            qub = qnn.predict(Xsub)
            residual = np.abs(y - qub)

            newX = self._clean_array(X, residual, loop=self.removal_iterations)

            if make_plot:
                plot.figure()
                plot.scatter(qub, residual, label='Residuals')
                #plot.scatter(qub[outliers], residual[outliers], color='orange', label='Outliers')
                plot.ylabel('Residual')
                plot.xlabel('Fitted Values')
                plot.legend()
                plot.tight_layout()

        if pass_back_endog:
            newX = newX[:,1]
        return newX

    def _clean_array(self,X, residual, loop=10):
        newX = np.copy(X)
        residuals = np.copy(residual)
        #outliers = np.array([])
        for i in range(loop):
            idx, _ = find_peaks(residuals, height=self.outlier_distance_threshold)
            # if i ==0:
            #     outliers = np.append(outliers, idx)
            residuals = np.delete(residuals, idx, axis=0)
            newX = np.delete(newX, idx, axis=0)
       # outliers = outliers.astype(np.int32)
        return newX


#%% main looked at quantile
if __name__ == '__main__':

    y = 0.5*np.linspace(0,100, num=100)
    t = np.array(list(range(100)))
    y += np.random.normal(0,t,size=100)


    qknn09 = QuantileKNNRegressor(tau=0.9, n_neighbors=20)
    qknn09.fit(t.reshape(-1,1), y)
    yhat09 = qknn09.predict(t.reshape(-1,1))

    qknn05 = QuantileKNNRegressor(tau=0.5, n_neighbors=20)
    qknn05.fit(t.reshape(-1,1), y)
    yhat05 = qknn05.predict(t.reshape(-1,1))

    qknn01 = QuantileKNNRegressor(tau=0.1, n_neighbors=20)
    qknn01.fit(t.reshape(-1,1), y)
    yhat01 = qknn01.predict(t.reshape(-1,1))


    plot.figure()
    plot.plot(t, y)
    plot.plot(t, yhat09, color='orange')
    plot.plot(t, yhat05, color='orange')
    plot.plot(t, yhat01, color='orange')
    plot.ylabel('y')
    plot.xlabel('x')
    plot.tight_layout()

#%% look at knn outlier

    plot.close('all')

    n_data = 1000

    y = []
    for t in range(n_data):
        if np.random.uniform() > 0.95 and t>0:
            y.append(y[t-1]*2)
        else:
            y.append(t*0.1 + np.random.normal() + 1e-3*t**2 - 1e-6 * t**3 )

    t = list(range(n_data))

    plot.figure()
    plot.title("Original")
    plot.plot(t, y)

    data = pandas.DataFrame([t,y])
    data = data.T

    od = OutlierKNNDetector(outlier_distance_threshold=3.0, n_neighbors=8, endog_idx=1)
    newX= od.remove_outliers(data)

    plot.figure()
    plot.title("Outliers Removed")
    plot.plot(newX[:,0], newX[:,1])








