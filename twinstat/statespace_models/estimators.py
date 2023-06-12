#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################



import numpy as np
import pandas
import copy
from joblib import Parallel, delayed

import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments


#--------------------------------------------------------------------------------------
class kalman(object):
    '''
     Provides an interface for various types of
     kalman filters.

    Parameters
    ----------


    kf_type: str
        - 'kf' : standard kalman filter
        - 'skf' : smoothed standard kalman filter
        - 'ukf' : unscented kalman filter
        - 'akf' : adaptive kalman filter


    n_em_iterations: int, optional
        'kf' type uses the EM algorithm to optimize the
        transition and observation matrices.  This option sets the
        number of allowable iterations.

    Methods
    -------

    get_estimate

    '''

    def __init__(self, kf_type: str,
                       data:np.array,
                       transition_matrix:np.array=None,
                       observation_matrix:np.array=None,
                       measurement_noise:np.array=None,
                       initial_state:np.array=None,
                       initial_state_covariance:np.array=None,
                       process_covariance:np.array=None,
                       n_em_iterations: int=10,
                       window_std_estimate:int = 5,
                       ncpu:int=1,
                       use_threads:bool = False):
        '''

        See https://en.wikipedia.org/wiki/Kalman_filter for more information on
        how Kalman Filters work.

        Parameters
        ----------
        kf_type : str
            kf, skf, akf, ukf
        data : np.array
        transition_matrix : np.array, optional
            If not present, identity matrix is used. The default is None.
        observation_matrix : np.array, optional
            If not present, identity matrix is used. The default is None.
        measurement_noise : np.array, optional
            If not present, TwinStat will utilize a rolling variance calculation
            to estimate each individual variable setting the diagnols of the matrix.
            The default is None.
        initial_state : np.array, optional
            If not present TwinStat will use the first data points in the provided data. The default is None.
        initial_state_covariance : np.array, optional
            If not present, identity matrix is used. The default is None.
        process_covariance : np.array, optional
            If not present, the measurement covariance matrix is used. The default is None.
        n_em_iterations : int, optional
            When using akf, the EM algorithm will use this many iterations. The default is 10.
        window_std_estimate : int, optional
            When estimating the measurment noise, TwinStat will using this many lags in
            the rolling variance estimation. The default is 5.
        ncpu : int, optional
            UKF will use ncpu for sigma point evaluation. The default is 1.


        Returns
        -------
        None.

        '''

        self.kf_type = kf_type
        self.ncpu = ncpu
        if use_threads:
            self.thread_prefer = 'threads'
        else:
            self.thread_prefer = None

        self.n_em_iterations = n_em_iterations
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix

        # Set up the initial state and covariance matrix
        if transition_matrix is not None:
            ndim = transition_matrix.shape[1]
        else:
            ndim = 1 if len(data.shape) == 1 else data.shape[1]

        self.ndim = ndim
        self.ndata = data.shape[0]
        self.nmeasured = 1 if len(data.shape) == 1 else data.shape[1]
        ninfer = ndim - self.nmeasured

        # Define initial state and covariance
        if initial_state is None:
            if ndim == 1:
                self.x0 = data[0]
            else:
                self.x0 = data[0,:]
            if ndim != self.nmeasured:
                e = np.zeros((ninfer,))
                self.x0 = np.concatenate((self.x0,e))
        else:
            self.x0 = initial_state

        # initial covariance matrix
        if initial_state_covariance is None:
            self.P0 = np.eye(ndim)
        else:
            self.P0 = initial_state_covariance

        if self.transition_matrix is None:
            self.transition_matrix = np.eye(ndim)

        if self.observation_matrix is None:
            self.observation_matrix = np.eye(ndim)
            if ninfer > 0:
                for i in range(self.nmeasured, self.nmeasured + ninfer):
                    self.observation_matrix[i,i] = 0

        # Define process noise and measurement noise covariances
        # initial covariance matrix
        if measurement_noise is None:
            df = pandas.DataFrame(data)
            roll_var = df.rolling(window=window_std_estimate).var().mean()
            estimated_var = roll_var.values
            #self.R = np.eye(ndim) * estimated_std

            if ninfer > 0:
                self.R = np.eye(ndim)
                for i in range(self.nmeasured):
                    self.R[i,i] = estimated_var[i]
                for i in range(self.nmeasured, self.nmeasured + ninfer):
                    self.R[i,i] = 1e-5
            else:
                self.R = np.eye(ndim) * estimated_var
        else:
            self.R = measurement_noise

        # initial covariance matrix
        if process_covariance is None:
            self.Q = copy.deepcopy(self.R)
            if ninfer > 0:
                print("WARNING: process covariance not set, auto setting inferred variable"
                      +" variance equal the measurment noise variable 1 variance.")
                for i in range(self.nmeasured, self.nmeasured + ninfer):
                    self.Q[i,i] = self.Q[0,0]
        else:
            self.Q = process_covariance

        # update with initialized
        self.process_covariance = self.Q
        self.measurement_noise = self.R

        # check that dimensions make sense
        if self.x0.shape[0] != ndim:
            raise ValueError("ERROR: initial hidden state must be same number of dimensions"
                             +" as the transition matrix. ")

        #---------

        if kf_type not in ['kf', 'skf', 'ukf', 'akf']:
            raise ValueError("ERROR: unknown kalman filter type ({})".format(
                            kf_type ))


    #follow standard KF for prediction step https://en.wikipedia.org/wiki/Kalman_filter#Predict
    def _linear_kf(self, xt_1_t_1, Pt_1_t_1, Y):
        #run normal linear KF

        A = self.transition_matrix
        H = self.observation_matrix
        Q = self.Q
        R = self.R

        xt_t_1 = A @ xt_1_t_1
        Pt_t_1 = A @ Pt_1_t_1 @ A.T + Q
        S_pred = H @ Pt_t_1 @ H.T + R

        #the adaptive KF can sometimes result in singularity
        #matrix as it is resolving the initial iterations of the
        #transition matrix.  Adding a jitter to enable refinement.
        #this only happens when there are many inferred variables.
        try:
            K = Pt_t_1 @ H.T @ np.linalg.inv(S_pred)
        except:
            S_pred += np.random.normal(0,1e-12,size=S_pred.shape)
            K = Pt_t_1 @ H.T @ np.linalg.inv(S_pred)

        xt_t = xt_t_1 + K @ ( Y - H @ xt_t_1  )
        Pt_t = Pt_t_1 - K @ H @ Pt_t_1
        return xt_t, Pt_t, K


    def _smoothed_linear_kf(self, xt_1_t_1, Pt_1_t_1, x_n_t, P_n_t,  Y):
        #Rauch–Tung–Striebel smoothing

        #run the backward smoothing algorithm for the KF inputs

        A = self.transition_matrix
        Q = self.Q

        #foreward kf
        Pt_t_1 = A @ Pt_1_t_1 @ A.T + Q

        #backward kf
        Jt_1 = Pt_1_t_1 @ A.T @ np.linalg.inv(Pt_t_1)
        x_n_t_1 = xt_1_t_1 + Jt_1 @ (x_n_t - A @ xt_1_t_1)
        P_n_t_1 = Pt_1_t_1 + Jt_1 @ (P_n_t - Pt_t_1) @ Jt_1.T

        return x_n_t_1, P_n_t_1, Jt_1


    # Define the state function
    def state_func(self, x:np.array ) -> np.array:
        '''
        A user is expected to provide their own state
        transition function for UKF, otherwise the default
        is the identity matrix.

        Parameters
        ----------
        x : np.array
            Input vector

        Returns
        -------
        Predicted next hidden variable value

        '''
        return self.transition_matrix @ x

    # Define the observation function
    def obs_func(self, x:np.array ) -> np.array:
        '''
        A user is expected to provide their own
        observation function for UKF, otherwise the default
        is the identity matrix.

        Parameters
        ----------
        x : np.array
            Input vector

        Returns
        -------
        Predicted next hidden variable value

        '''
        return self.observation_matrix @ x


    # Define the unscented transform function
    def _unscented_kalman_filter(self, x0, P0, Q, R, Y, alpha=0.001, beta=2.0, kappa=0.0):
        '''
        Coefficients and algorithm using standard UKF

        See https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf

        '''

        # Initialize state and covariance
        x_pred = x0
        P_pred = P0

        # Calculate the sigma points
        n = len(x0)
        ny =  len(R)
        lambda_ = alpha**2 * (n + kappa) - n
        c = n + lambda_
        W_m = np.concatenate(([lambda_ / c], 0.5 / c * np.ones(2 * n)), axis=None)
        W_c = np.concatenate(([lambda_ / c + (1 - alpha**2 + beta)], 0.5 / c * np.ones(2 * n)), axis=None)

        sigma_points = np.zeros((n,2 * n + 1))

        #check eigenvalues
        # print(np.linalg.eigvalsh(np.abs(P_pred)))
        #TODO: can be made more efficient, by checking eig
        #before loop, then determine if use choleskly or sqrt, before loop
        # then simply call the column in the loop

        sigma_points[:,0] = x_pred
        for j in range(1, n+1):
            # cholesky better matches inferred results with standard KF
            # however sometimes the cov is not quite positive definite
            # during temporal convolution, so switch to sqrt
            try:
                sigma_points[:,j] = x_pred + np.linalg.cholesky(c * np.abs(P_pred))[:,j-1] * np.sign(P_pred[:,j-1])
            except:
                sigma_points[:,j] = x_pred + np.sqrt(c * np.abs(P_pred))[:,j-1] * np.sign(P_pred[:,j-1])
        for j in range(n+1, 2*n+1):
            try:
                sigma_points[:,j] = x_pred - np.linalg.cholesky(c * np.abs(P_pred))[:,j-n-1]* np.sign(P_pred[:,j-n-1])
            except:
                sigma_points[:,j] = x_pred - np.sqrt(c * np.abs(P_pred))[:,j-n-1] * np.sign(P_pred[:,j-n-1])

        # Propagate the sigma points through the state function
        # x_sigma_pred = np.zeros((n, 2 * n + 1))
        # for i in range(2 * n + 1):
        #     x_sigma_pred[:, i] = ukf.state_func(sigma_points[:, i])

        x_sigma_pred = Parallel(n_jobs=self.ncpu, prefer=self.thread_prefer)(
                delayed(self.state_func)(sigma_points[:, i]) for i in range(2 * n + 1) )
        x_sigma_pred = np.array(x_sigma_pred)
        x_sigma_pred = x_sigma_pred.T

        # Calculate the predicted state and covariance
        x_pred = np.sum(W_m * x_sigma_pred, axis=1)
        P_pred = np.zeros((n, n))
        for i in range(2 * n + 1):
            P_pred += W_c[i] * np.outer(x_sigma_pred[:, i] - x_pred, x_sigma_pred[:, i] - x_pred)
        P_pred += Q

        # Propagate the sigma points through the observation function
        y_sigma_pred = np.zeros((ny, 2 * n + 1))
        for i in range(2 * n + 1):
            y_sigma_pred[:, i] = self.obs_func(x_sigma_pred[:, i])

        # Calculate the predicted measurement and covariance
        y_pred = np.sum(W_m * y_sigma_pred, axis=1)
        Pyy_pred = np.zeros((ny, ny))
        for i in range(2 * n + 1):
            Pyy_pred += W_c[i] * np.outer(y_sigma_pred[:, i] - y_pred, y_sigma_pred[:, i] - y_pred)
        Pyy_pred += R

        # Calculate the cross-covariance
        Pxy_pred = np.zeros((n, n))
        for i in range(2 * n + 1):
            Pxy_pred += W_c[i] * np.outer(x_sigma_pred[:, i] - x_pred, y_sigma_pred[:, i] - y_pred)

        # Calculate the Kalman gain
        K = Pxy_pred @ np.linalg.inv(Pyy_pred)

        # Update the state and covariance
        x_pred += K @ (Y - y_pred)
        P_pred -= K @ Pyy_pred @ K.T

        #ensure variance never goes non-negative due to round off
        np.fill_diagonal(P_pred, np.maximum(np.diag(P_pred),0)+1e-10)

        return x_pred, P_pred, K



    # Define the EM algorithm function
    def _em_kalman_filter(self,
                         x0, P0, Q, R, Y, A, H,
                         max_iters=10):

        ndata = Y.shape[0]
        ndim = self.ndim

        def _recursive_covariance(Jt_1, Jt_2, Pt_t_1_n, Pt_1_t_1):
            P_t_1t_2_n = Pt_1_t_1 @ Jt_2.T + Jt_1 @ (Pt_t_1_n - A @ Pt_1_t_1 ) @ Jt_2.T
            return P_t_1t_2_n

        # Run the EM algorithm
        for i in range(max_iters):

            # E-step: run the Kalman filter to estimate the hidden state
            x_filt = np.zeros((ndata, ndim))
            P_filt = np.zeros((ndata, ndim,ndim))
            K = np.zeros((ndata, ndim,ndim))
            J = np.zeros((ndata, ndim,ndim))

            x_smooth = np.zeros((ndata, ndim))
            P_smooth = np.zeros((ndata, ndim,ndim))

            P_lag1 = np.zeros((ndata, ndim,ndim))

            x_filt[0,:] = x0
            P_filt[0,:] = P0

            # First run normal kf
            for j in range(1,len(Y)):
                out = self._linear_kf(x_filt[j-1,:], P_filt[j-1,:], Y[j])
                x_filt[j,:] = out[0]
                P_filt[j,:] = out[1]
                K[j,:] = out[2]

            x_smooth[-1,:] = x_filt[-1,:]
            P_smooth[-1,:] = P_filt[-1,:]
            # Then run kf smoothing
            for j in reversed(range(1,len(Y))):
                #print(j)
                out = self._smoothed_linear_kf(x_filt[j-1,:], P_filt[j-1,:], x_filt[j,:], P_filt[j,:],  Y[j])
                x_smooth[j-1,:] = out[0]
                P_smooth[j-1,:] = out[1]
                J[j-1,:] = out[2]


            # MLE also depends on recursive lag covariance
            I = np.eye(ndim)
            P_nn_1_n = (I - K[-1] @ H) @ A @ P_smooth[-2,:]
            P_lag1[-1,:] = P_nn_1_n
            for j in reversed(range(1,len(Y))):
                P_lag1[j-1,:] = _recursive_covariance(J[j-1,:], J[j-2,:], P_lag1[j,:], P_filt[j-1,:])


            # M-step: update the model parameters using the estimated state
            E = np.zeros((ndim, ndim))
            D = np.zeros((ndim, ndim))
            F = np.zeros((ndim, ndim))
            for j in range(len(Y)-1):
                E += np.outer(x_smooth[j+1, :], x_smooth[j, :]) + P_lag1[j, :, :]
                D += np.outer(x_smooth[j, :], x_smooth[j, :]) + P_smooth[j, :, :]
                F += np.outer(x_smooth[j+1, :], x_smooth[j+1, :]) + P_smooth[j+1, :, :]

            A = E @ np.linalg.inv(D)

            Q = 1/len(Y) * (F - A @ E.T )

            R = np.zeros((ndim, ndim))
            for j in range(len(Y)):
                R +=  np.outer( Y[j] - H @ x_smooth[j, :] , Y[j] - H @ x_smooth[j, :] ) + H @ P_smooth[j, :, :] @ H.T
            R *= 1/len(Y)

            #Q = np.maximum(Q, 1e-10)
            #R = np.maximum(R, 1e-10)

            #ensure variance never goes non-negative due to round off
            np.fill_diagonal(Q, np.maximum(np.diag(Q),0)+1e-10)
            np.fill_diagonal(R, np.maximum(np.diag(R),0)+1e-10)

            # Update object attributes
            self.Q = Q
            self.R = R
            self.A = A
            self.transition_matrix = A
            self.observation_matrix = H

        return x_smooth, P_smooth, A, H, Q, R, K


    def get_estimate(self, Y: np.array) -> np.array and np.array :
        '''
        Use the kalman filter type to estimate the 0:n
        estimations of the hidden statespace variable for the
        provided data.

        Parameters
        ----------
        data : np.array

        Returns
        -------
        filtered_state_means : np.array
        filtered_state_covariances : np.array

        '''

        ndata = Y.shape[0]
        ndim = self.ndim

        #mod Y with blank columns to ensure linear algebra works out
        inferred = ndim - Y.shape[1]
        blank = np.zeros((ndata, inferred))
        Y = np.hstack((Y, blank))

        # initialize predicted state matrix
        x_filt = np.zeros((ndata, ndim))
        # initialize predicted state covariance matrix
        P_filt = np.zeros((ndata, ndim,ndim))
        # initialize Kalman Gain
        #K = np.zeros((ndata, ndim,1))
        K = np.zeros((ndata, ndim, ndim))
        # initialize Smoothing Gain
        J = np.zeros((ndata, ndim,ndim))

        # initialize predicted state matrix with smoothing
        x_smooth = np.zeros((ndata, ndim))
        # initialize predicted state covariance matrix with smoothing
        P_smooth = np.zeros((ndata, ndim,ndim))

        x0 = self.x0
        P0 = self.P0
        Q = self.Q
        R = self.R

        if self.kf_type == 'kf':
            x_filt[0,:] = x0
            P_filt[0,:] = P0
            for j in range(1, self.ndata):
                out = self._linear_kf( x_filt[j-1,:], P_filt[j-1,:], Y[j])
                x_filt[j,:] = out[0]
                P_filt[j,:] = out[1]
                K[j,:] = out[2]

            filtered_state_means = x_filt
            filtered_state_covariances = P_filt
            self.K = K

        elif self.kf_type == 'skf':
            x_filt[0,:] = x0
            P_filt[0,:] = P0
            for j in range(1, self.ndata):
                out = self._linear_kf( x_filt[j-1,:], P_filt[j-1,:], Y[j])
                x_filt[j,:] = out[0]
                P_filt[j,:] = out[1]
                K[j,:] = out[2]

            x_smooth[-1,:] = x_filt[-1,:]
            P_smooth[-1,:] = P_filt[-1,:]
            self.K = K

            #then run kf smoothing
            for j in reversed(range(1,len(Y))):
                #print(j)
                out = self._smoothed_linear_kf(x_filt[j-1,:], P_filt[j-1,:], x_filt[j,:], P_filt[j,:],  Y[j])
                x_smooth[j-1,:] = out[0]
                P_smooth[j-1,:] = out[1]
                J[j-1,:] = out[2]

            filtered_state_means = x_smooth
            filtered_state_covariances = P_smooth
            self.J = J

        elif self.kf_type == 'akf':

            H = self.observation_matrix
            A = self.transition_matrix

            filtered_state_means, filtered_state_covariances, A, H, Q_hat, R_hat, K = \
                                        self._em_kalman_filter(x0, P0, Q, R, Y,
                                                              A = A,
                                                              H = H,
                                                              max_iters=self.n_em_iterations
                                                              )

            # updating so a user can review the EM results if desired
            self.Q = Q_hat
            self.R = R_hat
            self.transition_matrix = A
            self.process_covariance = self.Q
            self.measurement_noise = self.R
            self.K = K

        elif self.kf_type == 'ukf':
            x_filt = [x0]
            P_filt = [P0]
            K = []
            for j in range(1,self.ndata):
                out = self._unscented_kalman_filter(x_filt[j-1], P_filt[j-1], Q, R, Y[j])
                x_filt.append(out[0])
                P_filt.append(out[1])
                K.append(out[2])

            filtered_state_means = np.array(x_filt)
            filtered_state_covariances = np.array(P_filt)
            self.K = K
        else:
            raise ValueError(f"ERROR: unknown KF type provided: {self.kf_type}.")


        return filtered_state_means, filtered_state_covariances




#--------------------------------------------------------------------------------------
#TODO: add template for multi-output
class HiddenVarModel(ssm.StateSpaceModel):
    """
    Definition of the hidden variable model to be used in the
    particle filtering.

    This object currently assumes both a gaussian hidden markov
    variables and a guassian measurment error.

    Also assumes a unity observation/transition function.

    Users can instantiate this object and alter various parts based on
    their application.  Use the customized object with the 'statespace_model'
    model input.

    Parameters
    ----------

    sigmaX: float, optional
    sigmaY: float, optional


    """
    default_params = { 'sigmaX': 0.5, 'sigmaY': 0.5}


    def PX0(self):  # Distribution of X_0
        '''
        Define the function that initializes the particle filter
        '''
        return dists.Normal()

    def f(self, x):
        '''
        Define the transition function
        '''
        return x

    def PX(self, t, xp):  #  Distribution of X_t given X_{t-1} = xp (p=past)
        '''
        Define the sampling distribution of the transition function
        '''
        return dists.Normal(loc=self.f(xp), scale=self.sigmaX)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x, and X_{t-1}=xp
        '''
        Define the observation sampling distribution
        '''
        return dists.Normal(loc=x, scale=self.sigmaY)


class particle_filter(object):
    '''
     Provides an interface for sequential monte carlo
     particle filtering of time series signals. Relying on
     the "particles" package.

    Default model:

        .. code-block:: python

            X = X(n-1) + N(0,sigmaX)
            Y = X + N(0,sigmaY)


    While the assumed model is gaussian variability, user can provide
    their own statespace model by extending the particles
    StateSpaceModel object.  See the default _HiddenVarModel object
    as an example for setup.

    Full package options can be found here:

        https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/

    Parameters
    ----------

    y:np.array, optional
        required if no values are provided for sigmaX and sigmaY
    sigmaX: float, optional,
        Assumed gaussian noise of the hidden variable.
        If user provides no value, a moving average with window of 'window_std_estimate'
        of the standard deviation will be used. Note that this variable will control
        the level of smoothing relative to the sigmaY value. The smaller sigmaX
        is relative to sigmaY, the greater the smoothing.
    sigmaY: float, optional,
        Assumed gaussian noise of the measurement variable.
        If user provides no value, a moving average with window of 'window_std_estimate'
        of the standard deviation will be used.
    n_particles: float, optional
        Number of particles to use in monte carlo sampling.
    statespace_model: object, optional
    window_std_estimate: int, optional, default = 5
        Window size used when estimating the mesurement variability.


    '''

    def __init__(self, y:np.array=None,
                       sigmaX: float=None,
                       sigmaY: float=None,
                       n_particles: float=500,
                       statespace_model:ssm.StateSpaceModel=None,
                       window_std_estimate:int=5):


        if (sigmaX is None or sigmaY is None) and statespace_model is None:
            if y is None:
                raise ValueError("ERROR: if either sigmaX or sigmaY is not provided"
                                 +" the 'y' input must be provided to estimate the"
                                 +" the variability of the data.")

            #use a rolling moving average of the standard deviation with a window
            #of 5 to obtain an estimate of the variability
            df = pandas.DataFrame(y)
            roll_std = df.rolling(window=window_std_estimate).std().mean()
            estimated_std = float(roll_std)
            print(f"Estimated std {estimated_std:.2f}")
            if sigmaX is None:
                sigmaX = estimated_std
            if sigmaY is None:
                sigmaY = estimated_std

        if statespace_model is None:
            self.the_ssm = HiddenVarModel(sigmaX=sigmaX, sigmaY=sigmaY)
        else:
            self.the_ssm = statespace_model

        self.n_particles = n_particles

        self.sigmaY = sigmaY
        self.sigmaX = sigmaX


    def get_estimate(self, data: np.array) -> np.array and np.array :
        '''
        Use the SIR particle filter to estimate the 0:n
        predictions of the provided data.

        Parameters
        ----------
        data : np.array


        Returns
        -------
        means : np.array
        variances : np.array

        '''

        fk_bootstrap = ssm.Bootstrap(ssm=self.the_ssm, data=data)

        #TODO: add user hooks for the various options, use kwargs if needed
        alg = particles.SMC(fk=fk_bootstrap, N=self.n_particles,
                            qmc=False,
                            resampling='systematic',
                            ESSrmin=0.5,
                            store_history=True,
                            verbose=False,
                            collect=[Moments()])
        alg.run()

        # the package creates a list of dictionaries,
        # so lets make this more usable
        tmp = [[m['mean'], m['var']] for m in alg.summaries.moments]
        means, variances = list(zip(*tmp))

        return means, variances


#----------------------------------------------------------
#%% main, KF examples
if __name__ == '__main__':


    import matplotlib.pyplot as plot

    n_data = 100
    x = np.linspace(0,100,num=n_data)
    y_org = np.sin(50*x)
    #gaussian noise
    np.random.seed(0)
    y1 = y_org + np.random.normal(0, 0.5, size=n_data)
    y2 = y_org + np.random.normal(0, 0.5, size=n_data)
    y1 = np.expand_dims(y1,-1)
    y2 = np.expand_dims(y2,-1)
    y = np.concatenate((y1,y2, y2),axis=-1)
    #y = np.concatenate((y1,y2),axis=-1)
    #y = y2


    kf = kalman('kf', y)
    skf = kalman('skf', y)
    akf = kalman('akf', y, n_em_iterations=20)
    ukf = kalman('ukf', y, ncpu=1)

    kf_hat,  kfP_hat = kf.get_estimate(y)
    skf_hat, skfP_hat = skf.get_estimate(y)
    ukf_hat, ukfP_hat = ukf.get_estimate(y)
    akf_hat, akfP_hat = akf.get_estimate(y)

    print(akf.transition_matrix, '\n\n', akf.process_covariance, '\n\n', akf.measurement_noise)

    print( np.mean(kfP_hat,axis=0), '\n\n', np.mean(akfP_hat,axis=0))

    dim = 1
    plot.figure(figsize=(14,5))
    plot.plot(x,y[:,dim], label='Noisy Measurment', color='lightgrey')
    plot.plot(x,kf_hat[:,dim] , label='Kalman Filter', linestyle='solid')
    #plot.plot(x,skf_hat[:,dim] , label='Smoothing Kalman Filter', linestyle='dashed')
    plot.plot(x,ukf_hat[:,dim] , label='Unscented Kalman Filter', linestyle='-.')
    #plot.plot(x,akf_hat[:,dim] , label='Adaptive Kalman Filter', linestyle='dashed')
    plot.legend()


    # plot.figure(figsize=(14,5))
    # plot.plot(x,y, label='Noisy Measurment', color='lightgrey')
    # plot.plot(x,kf_hat , label='Kalman Filter', linestyle='solid')
    # plot.plot(x,skf_hat , label='Smoothing Kalman Filter', linestyle='dashed')
    # plot.plot(x,ukf_hat , label='Unscented Kalman Filter', linestyle='-.')
    # plot.plot(x,akf_hat , label='Adaptive Kalman Filter', linestyle='dashed')
    # plot.legend()



#%% particle filter example

    #create a random walk with discontinuous jump
    y = [0]
    for i in range(n_data):
        if i == 50:
            bias = 0.5
        else:
            bias = 0
        y.append( y[i] + np.random.normal(0,0.1) + bias)

    x = np.arange(len(y))
    y = np.array(y)

    pobj = particle_filter(y=y, n_particles=1000)
    state_means, state_var = pobj.get_estimate(y)

    kf = kalman('kf', y)
    kf_hat,  kfP_hat = kf.get_estimate(y)

    print( np.mean(kfP_hat,axis=0), '\n\n', np.mean(state_var,axis=0))


    plot.figure(figsize=(14,5))
    plot.plot(y, label='Noisy Measurment', color='lightgrey')
    plot.plot(x,kf_hat , label='Kalman Filter', linestyle='solid')
    plot.plot(x,state_means , label='Particle Filter', linestyle='dashed')
    plot.legend()

#%% test kalman with inferred variables

    #2 measured, 3 hidden
    y = np.concatenate((y1,y2),axis=-1)

    #create some dummy transition matrix so that the
    #inferred variable is related to the measured
    f = np.eye(3)
    f[2,0] = 1/3
    f[2,1] = 1/3
    f[2,2] = 1/3

    kf = kalman('kf', y, transition_matrix=f )
    skf = kalman('skf', y, transition_matrix=f )
    ukf = kalman('ukf', y, transition_matrix=f , ncpu=5)
    akf = kalman('akf', y, n_em_iterations=20, transition_matrix=f )


    kf_hat,  kfP_hat = kf.get_estimate(y)
    skf_hat, skfP_hat = skf.get_estimate(y)
    ukf_hat, ukfP_hat = ukf.get_estimate(y)
    akf_hat, akfP_hat = akf.get_estimate(y)

    #review measured
    dim = 0
    plot.figure(figsize=(14,5))
    plot.title('measure')
    plot.plot(x,y[:,dim], label='Noisy Measurment', color='lightgrey')
    plot.plot(x,kf_hat[:,dim] , label='Kalman Filter', linestyle='solid')
    plot.plot(x,skf_hat[:,dim] , label='Smoothing Kalman Filter', linestyle='dashed')
    plot.plot(x,ukf_hat[:,dim] , label='Unscented Kalman Filter', linestyle='-.')
    plot.plot(x,akf_hat[:,dim] , label='Adaptive Kalman Filter', linestyle='dashed')
    plot.legend()

    #review inferred
    dim = 2
    plot.figure(figsize=(14,5))
    plot.title('infer')
    plot.plot(x,kf_hat[:,dim] , label='Kalman Filter', linestyle='solid')
    plot.plot(x,skf_hat[:,dim] , label='Smoothing Kalman Filter', linestyle='dashed')
    plot.plot(x,ukf_hat[:,dim] , label='Unscented Kalman Filter', linestyle='-.')
    plot.plot(x,akf_hat[:,dim] , label='Adaptive Kalman Filter', linestyle='dashed')
    plot.legend()