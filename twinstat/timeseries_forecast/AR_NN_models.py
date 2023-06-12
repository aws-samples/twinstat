#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################



import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
tfd = tfp.distributions


from twinstat.core.neural_network_base import base_neural_network



#--------------------------------------------------------------------------
class AR_quantile_neural_network(base_neural_network):
    '''

    Use a tensorflow to train an autoregressive
    quantile neural network.  The loss seeks to minimize
    the weighted residual based on the quantile which causes
    predicitions to estimate the expectation of the quantile.

    Parameters
    ----------

    tau: float, optional
        Determines the quantile [0.0-1.0] that the network
        will seek to follow.  The default is 0.5.

        Example: 0.5 will result in following the median of the data
        where as 0.95 will attempt to cover 95% of the data.

    loss_type: str, optional
        Determines the loss function to use while training the
        neural network.  The default is 'quantile'.  Accepts
        strings for any of the standard tensorflow loss functions.

    Methods
    -------

    '''

    def __init__(self,
                 tau: float=0.5,
                 loss_type:str='quantile',
                 **kwargs):
        '''


        Parameters
        ----------
        tau : float, optional
            The percentile to use in a quantile regression. The default is 0.5.
        loss_type : str, optional
            Any tensorflow loss or 'quantile'. The default is 'quantile'.

        Returns
        -------
        None.

        '''

        self.tau=tau
        self.loss_type = loss_type
        base_neural_network.__init__(self,**kwargs)

        self.model = self._create_model()
        if self.loss_type =='quantile':
            loss_type = tfa.losses.PinballLoss(tau)

        #example would be using 'mse' loss type instead of quantile losses
        #tenorflow accepts strings for standard losses
        self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss=loss_type)

    def _create_model(self):

        if self.isfilter:
            oneMore = 1
        else:
            oneMore = 0

        if self.include_endog:
            inputs = keras.Input(shape=(self.AR + oneMore) + (self.AR + 1) * self.n_exog_variables)
        else:
            inputs = keras.Input(shape=(self.AR + 1) * (self.n_exog_variables))


        xi = layers.Dense(
            units=self.hidden_units,
            activation=self.activation,
        )(inputs)
        xi = layers.BatchNormalization()(xi)
        xi = layers.Dropout(self.dropout_frac)(xi)

        for _ in range(self.n_layers-1):
            xi = layers.Dense(
                units=self.hidden_units,
                activation=self.activation,
            )(xi)
            xi = layers.BatchNormalization()(xi)
            xi = layers.Dropout(self.dropout_frac)(xi)

        quantile = layers.Dense(units=1)(xi)

        model_train = keras.Model(inputs=inputs, outputs=quantile)

        return model_train

    def load_model(self, filename):
         super(AR_quantile_neural_network,self).load_model(filename)
         self.model = self._create_model()
         self.model.load_weights(filename+'.h5')





#%% main
if __name__ == '__main__':

    #TODO: change these into test cases
    #test on some dummy data-------
    #-----------
    #-----------

     n_data = 1000
     n_exo = 0
     AR_signal = 1
     #create a random walk with discontinuous jump
     y = [0]
     for i in range(n_data):
         if i == 50:
             bias = 0.5
         else:
             bias = 0
         y.append( y[i]* AR_signal + np.random.normal(0,5.1) + bias)

     y = np.array(y)
     #dummy exo variables
     X = np.random.normal(0,0.1, size=(y.shape[0], n_exo))
     y = np.reshape(y,(-1,1))


     AR=5
     nobj = AR_quantile_neural_network(AR=AR, n_exog_variables= n_exo , tau=0.95)

     shifted = nobj._create_shifted_dataset(y)


     nobj.train(y)
     nobj.plot()

     result = nobj.get_estimate(y)


     t = list(range(len(y)))

     plot.figure(figsize=(14,5))
     plot.title(AR)
     plot.plot(t[AR:],y[AR:,0], label='Noisy Measurment', color='lightgrey')
     plot.plot(t[AR:], result[:,0] , label='ANN', linestyle='dotted')
     plot.legend()

