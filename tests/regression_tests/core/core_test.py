# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:55:47 2022

@author: rpivovar
"""


import numpy as np
from twinstat.core.statistical_tests import distribution_difference_MC_test
from twinstat.core.knn_models import OutlierKNNDetector
import pandas


def test_MC_integrate():

    n_var = 5
    n_data = 200
    X= None
    for i in range(8):
        np.random.seed(i)
        means1 = np.random.uniform(size=n_var)
        np.random.seed(i)
        means1 *= np.random.randint(1,15)
        np.random.seed(i)
        x = np.random.multivariate_normal(means1, np.eye(n_var)*1.5, size=n_data)
        if i >0:
            X = np.append(X,x, axis=0)
        else:
            X = x

    Y = None
    for i in range(3):
        np.random.seed(i+100)
        means2 = np.random.uniform(size=n_var)
        np.random.seed(i+100)
        means2 *= np.random.randint(1,50)
        np.random.seed(i+100)
        y = np.random.multivariate_normal(means2, np.eye(n_var)*1.5, size=n_data)
        if i >0:
            Y = np.append(Y,y, axis=0)
        else:
            Y = y


    P_given_f, bgm1, bgm2 = distribution_difference_MC_test(X, Y,
                                                            n_mixtures_X=10,
                                                            n_mixtures_Y=10,
                                                            )

    good_arr = [1, 0, 1, 1, 1, 1, 0, 1, 0, 1]

    new_arr = list(P_given_f.values())
    new_arr = [1 if x[0] > 1e-3 else 0 for x in new_arr ]

    compare = np.allclose(new_arr,good_arr, rtol=1e-3,atol=1e-3)
    assert compare == True



def test_knn_outlier():
    n_data = 10
    y = []
    for t in range(n_data):
        np.random.seed(t)
        if np.random.uniform() > 0.95 and t>0:
            y.append(y[t-1]*2)
        else:
            np.random.seed(t+100)
            y.append(t*0.1 + np.random.normal())

    t = list(range(n_data))

    data = pandas.DataFrame([t,y])
    data = data.T

    od = OutlierKNNDetector(outlier_distance_threshold=3.0, n_neighbors=8, endog_idx=1)
    newX= od.remove_outliers(data)



    good_arr = np.array([[ 0.        , -1.74976547],
           [ 1.        ,  2.80684984],
           [ 2.        ,  1.8680683 ],
           [ 3.        , -0.94927835],
           [ 4.        , -1.8985567 ],
           [ 5.        ,  0.25467433],
           [ 6.        ,  2.76493494],
           [ 7.        ,  1.19399194],
           [ 8.        , -0.22690451],
           [ 9.        ,  0.71226506]])


    compare = np.allclose(newX,good_arr, rtol=1e-3,atol=1e-3)
    assert compare == True
