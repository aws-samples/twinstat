# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:55:47 2022

@author: rpivovar
"""

import numpy as np
from twinstat.timeseries_forecast.parametric import ts_model


def get_random_data():
    n_data = 100
    x = np.linspace(0,100,num=n_data)
    y_org = np.sin(50*x)
    #gaussian noise
    np.random.seed(0)
    y = y_org + np.random.normal(0, 0.5, size=n_data)

    return y



def get_random_volatile_data():
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

    return y


def test_ARIMA():

    y = get_random_data()

    n_test = int(y.shape[0]*0.1)
    train = y[:-n_test]

    TS = ts_model(max_p = 2,
                  max_q = 2,
                  D=1,
                  n_jobs=1,
                  information_criterion='bic',
                  stepwise=False,
                  maxiter=150)
    TS.model.fit(train)

    print(TS.model.summary())

    nperiod = 5
    y_hat = TS.model.predict(n_periods=nperiod)

    good_arr = np.array([ 0.66747439, 0.71789502, 0.59072216, 0.54532741, 0.48335197])

    compare = np.allclose(y_hat,good_arr, rtol=1e-4,atol=1e-4)
    assert compare == True


def test_ARCH():

    train = get_random_volatile_data()

    TS = ts_model(volatility_model=True)

    TS.auto_arch(train,
                 max_p = 2,
                 max_q = 2,
                 n_jobs=-1)

    params = np.ravel(TS.model.params)

    good_arr = np.array([15.04326643,  0.35972621,  0.08430764,  0.91569236])

    compare = np.allclose(params,good_arr, rtol=1e-2,atol=1e-2)
    assert compare == True

