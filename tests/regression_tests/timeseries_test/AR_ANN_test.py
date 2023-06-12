# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:55:47 2022

@author: rpivovar
"""

import numpy as np
from twinstat.timeseries_forecast.AR_NN_models import AR_quantile_neural_network
import random, os
import tensorflow as tf

def get_random_data():
    n_data = 50
    y = [0]
    for i in range(n_data):
        if i == 50:
            bias = 0.5
        else:
            bias = 0
        np.random.seed(i)
        y.append( y[i] + np.random.normal(0,0.1) + bias)

    return y


def find_weights_location():
    path ="../"
    folder_loc = ""
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npz"):
                folder_loc = root
    return folder_loc


def test_ANN():

    y = get_random_data()

    seed=0
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


    nobj = AR_quantile_neural_network(AR=1, tau=0.5,
                                      hidden_units= 128,
                                      isfilter=True,
                                      batch_size=50)

    # nobj.train(y,patience=200)
    # nobj.save_model('test_weights')

    fileloc = find_weights_location()

    nobj.load_model(fileloc+'/test_weights')
    quantile = nobj.get_estimate(y)

    good_arr = np.array([[0.21408284],
           [0.36039674],
           [0.31837547],
           [0.49566397],
           [0.4998356 ],
           [0.54349864],
           [0.5124224 ],
           [0.6799486 ],
           [0.6881898 ],
           [0.6882574 ],
           [0.8200984 ],
           [0.9926924 ],
           [1.0387071 ],
           [0.9679521 ],
           [1.1218837 ],
           [1.0902361 ],
           [1.1030505 ],
           [1.1303436 ],
           [1.1380793 ],
           [1.159924  ],
           [1.247336  ],
           [1.2417789 ],
           [1.2326989 ],
           [1.2987808 ],
           [1.4300762 ],
           [1.4520575 ],
           [1.4713565 ],
           [1.5985537 ],
           [1.642146  ],
           [1.6006024 ],
           [1.4756426 ],
           [1.4351665 ],
           [1.4008157 ],
           [1.3694082 ],
           [1.3936995 ],
           [1.2064811 ],
           [1.274335  ],
           [1.2686266 ],
           [1.368525  ],
           [1.5071491 ],
           [1.44634   ],
           [1.4198197 ],
           [1.4691263 ],
           [1.4943795 ],
           [1.4199408 ],
           [1.4229022 ],
           [1.4807996 ],
           [1.3965646 ],
           [1.2985796 ],
           [1.1957589 ]])


    compare = np.allclose(quantile,good_arr, rtol=1e-5,atol=1e-5)
    assert compare == True


