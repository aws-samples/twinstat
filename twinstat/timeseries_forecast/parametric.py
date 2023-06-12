#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


import pmdarima
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plot

class ts_model(object):
    '''
    Provides a pass through interface for
    AutoARIMA or ARCH models.

    Users should see the following links for full API inputs to
    these functions. This object currently provides no additional
    value for ARIMA.

    Main functionality is the additional of AutoARCH grid searching.

    AutoARIMA documentation:
        http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html#pmdarima.arima.AutoARIMA


    Arch documentation:
        https://arch.readthedocs.io/en/latest/

    Parameters
    ----------

    volatility_model: bool, optional
        Determines if an ARIMA or ARCH model should be used.
        Default is False, which uses an ARIMA model.


    Examples
    -------

    In this example, an ARIMA model is automatically selected that
    best fits the provided 'y' data. The automation performs a
    parallized grid search with the max AR lags of 8, max MA lags
    of 3, and 1 level of signal differencing is used.

    The predict function will start from the end of the training
    data and forecast n_periods into the future.

    .. code-block:: python

        from twinstat.timeseries_forecast.parametric import ts_model
        TS = ts_model(max_p =8,
                      max_q=3,
                      D=1,
                      n_jobs=-1,
                      stepwise=False)
        TS.model.fit(y)
        TS.model.summary()
        TS.model.predict(n_periods=5)

    '''

    def __init__(self,
                 volatility_model: bool=False,
                 **kwargs):

        self.volatility_model = volatility_model

        if not volatility_model:
            self.model = pmdarima.arima.AutoARIMA(**kwargs)


    def plot(self, series:np.array, lags:int = 40) -> None:
        '''
           Some useful diagnostic plots to determine
           what the proper AR and MA terms would be
           for the provided data.

           Note that the a decaying ACF will determine MA
           and a decarying PACF will determine AR.  Users may
           want to increase the number of lags shown, to
           assess if a season lag needs to be included
           in the models.

           Users are recommended to see a time series text
           to fully understand how to use these plots.

        Parameters
        ----------
        series : np.array
            Data to plot.
        lags : int, optional
            Number of lags to include in the plot. The default is 40.

        Returns
        -------
        None.

        '''

        fig, ax = plot.subplots(2, 1, figsize=(12,5))
        pmdarima.utils.plot_acf(series, lags=lags, ax=ax[0])
        pmdarima.utils.plot_pacf(series, lags=lags, ax=ax[1])
        plot.tight_layout()

        plot.figure(figsize=(12,5))
        plot.plot(series)
        plot.xlabel('Time')
        plot.ylabel('y')
        plot.tight_layout()


    def auto_arch(self, series:np.array,
                        max_p:int=3,
                        max_o:int=0,
                        max_q:int=3,
                        n_jobs:int=1,
                        **kwargs) -> None:
        '''
        Perform a grid search to find best ARCH model.
        Performance criteria based on the Bayesian Information Criterion (BIC)


        Parameters
        ----------
        series : np.array
            Data to fit.
        max_p : int, optional
            max AR terms to use in the fitting. The default is 3.
        max_o : int, optional
            max asymmetric innovation terms to use in the fitting. The default is 0.
        max_q : int, optional
            max MA terms to use in the fitting. The default is 3.
        n_jobs : int, optional
            Number of parallel processes to use.  -1 will
            attempt to use all available CPUs. The default is 1.
        rescale: bool, optional
            Internally rescale data to improve convergence.

        Returns
        -------
        None

        '''


        from itertools import product
        from joblib import Parallel, delayed


        arlst = range(1,max_p+1)
        malst = range(max_q+1)
        olst = range(max_o+1)
        grid = list(product(arlst,malst, olst))

        def get_arch_result(inputs):
            ar,ma, o = inputs[0], inputs[1], inputs[2]
            result = arch_model(series,p=ar, o=o, q=ma,
                                **kwargs).fit(disp="off")
            return [result.bic, result]

        results = Parallel(n_jobs=n_jobs)(delayed(get_arch_result)(x)
                                                      for x in grid)

        bic, models = list(zip(*results))
        best = np.argmin(bic)
        best_model = models[best]

        print(best_model.summary())

        self.model = best_model


#%% main
if __name__ == '__main__':

    '''ARIMA Example.'''



    n_data = 100
    x = np.linspace(0,100,num=n_data)
    y_org = np.sin(50*x)
    #gaussian noise
    np.random.seed(0)
    y = y_org + np.random.normal(0, 0.5, size=n_data)

    n_test = int(n_data*0.1)
    train = y[:-n_test]
    test = y[-n_test:]


    TS = ts_model(max_p =8,
                  max_q=3,
                  D=1,
                  n_jobs=-1,
                  stepwise=False)
    TS.model.fit(train)

    #multiple ways to generate a model for this oscillating data
    #clearly a seasonal model with a lag of ~37 would work, but
    #a differenced model would also nicely fit this data. Users
    #would want to review the AIC to determine which is
    # better for their data.
    TS.plot(train)
    TS.plot(np.diff(train))


    print(TS.model.summary())
    nperiod = 20
    y_hat, conf_int = TS.model.predict(n_periods=nperiod, return_conf_int=True)

    t = range(y.shape[0])
    t_train = range(train.shape[0])
    that = range(train.shape[0],train.shape[0]+nperiod)

    plot.figure(figsize=(12,5))
    plot.plot(t,y, label='All Data')
    plot.scatter(t_train,train, label='Training Data', color='black')
    plot.plot(that, y_hat,  color='orange', label='ARIMA')

    plot.fill_between(that,
                      conf_int[:, 0], conf_int[:, 1],
                      alpha=0.1, color='b')

    plot.ylabel('y')
    plot.xlabel('Time')
    plot.legend()
    plot.tight_layout()


#%% ARCH model example


    n_data = 300
    mu = 15
    y = []
    for i in range(n_data):
        if i % 20 ==0:
            np.random.seed(i+50)
            y.append( mu + np.random.normal(0,20+i/30)     )
        else:
            np.random.seed(i+50)
            y.append( mu + np.random.normal(0,1+i/30)     )


    #because this data is volatile and non-constant variance, ARIMA is invalid
    #use an ARCH model instead
    plot.figure()
    plot.plot(y)

    train = y

    TS = ts_model(volatility_model=True)

    TS.auto_arch(train,
                 max_p = 10,
                 max_q = 10,
                 n_jobs=-1)


    TS.model.plot()

    nperiod = 20
    samples = 10
    possible_results = []
    for _ in range(samples):
        scenario_forecasts = TS.model.forecast(horizon=nperiod,method="simulation", reindex = True)
        variance = np.ravel(scenario_forecasts._variance.tail(1))
        possible_results.append(variance)


    plot.figure()
    for i in range(samples):
        plot.plot(possible_results[i])

    plot.xlabel("Time")
    plot.ylabel("Simulated Variability")
    plot.tight_layout()























