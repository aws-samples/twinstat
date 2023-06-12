# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################



import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



class base_neural_network(object):
    '''
    Base object used to define tensorflow objects.  Other
    downstream TwinStat objects are built off of this object
    or users can create custom neural networks off of this
    object.

    '''

    def __init__(self,
                 AR: int=5,
                 n_exog_variables: int = 0,
                 n_layers: int = 1,
                 hidden_units: int = 64,
                 activation: str="relu",
                 lr: float= 5e-3,
                 batch_size:int=32,
                 validation_frac:float=0.1,
                 epochs:int=10000,
                 dropout_frac: float=0.0,
                 optimize: bool=False,
                 include_endog:bool=True,
                 isfilter:bool=False,
                 scale_y:bool=False,
                 test_train_split_method:str='timeseries'):
        '''

        Parameters
        ----------

         AR : int, optional
            Number of autoregressive lags to include in the
            BNN. The default is 5.
         n_exog_variables : int, optional
             Number of exogenous predictors to generate a ANN model size.
             The default is 0.
         n_layers : int, optional
             Depth of the ANN. The default is 1.
         hidden_units : int, optional
             Width of the ANN. The default is 64.
         activation : str, optional
             Activation function used after each dense layer.
             Accepts all tensorflow input.
             The default is "relu".
         lr : float, optional
             Optimizer learning rate. The default is 5e-3.
         batch_size : int, optional
             SGD batch size. The default is 32.
         validation_frac : float, optional
             Fraction of data to be used in the validation set.
             The default is 0.1.
         epochs : int, optional
             SGD epochs. The default is 10000.
         dropout_frac : float, optional
             Fraction of dropout ANN nodes. Only used in training.
             The default is 0.0.
         optimize: bool, optional (not setup yet)
             If true, the NN will be optimized with a bayesian optimizer
             to find the best depth, width, learning rate, and activation function
             for the provided data. The default is False.
         include_endog : bool, optional
             If true, the lagged endogenous variable will be included
             as a predictor. The default is True.
         isfilter : bool, optional
             If true, the current time endogenous variable will be
             included as a predictor.  Warning, this is useful when
             creating a filter, but otherwise the neural network will
             overfit by learning to simply pass the current time through
             the graph and to the output to achieve perfect accuracy.
             The default is False.
          scale_y : bool, optional
              Scale the response variable prior to training and prediction.
              The default is False.
          test_train_split_method : str, optional
              If 'timeseries'  the split is a cut in the timeseries in which
              the first (1-validation_frac) is used for training and the
              last validation_frac is used for validation.

              If 'random' the split is a randomly selected (1 - validation_frac)
              for training and the remaining validation_frac used for
              validation.

              The default is 'timeseries'.

        Returns
        -------
        None.

        '''

        self.AR = AR
        self.n_exog_variables = n_exog_variables
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.lr=lr
        self.batch_size=batch_size
        self.validation_frac=validation_frac
        self.epochs=epochs
        self.include_endog = include_endog
        self.dropout_frac = dropout_frac
        self.isfilter = isfilter
        self.test_train_split_method = test_train_split_method

        self.scale_y = scale_y
        self.scale_mu_y = None
        self.scale_sigma_y = None

    def _create_shifted_dataset(self, data, include_current_time = False):

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if include_current_time:
            oneMore = 1
        else:
            oneMore = 0

        lst = []
        #TODO: for a large data set, this for loop could be quite slow
        # should be able to at least use list comprehension
        for row in range(data.shape[0]):
            if row >= self.AR:
                if len(data.shape) > 1:
                    tmp = data[row-self.AR:row+oneMore,:]
                    tmp = list(tmp.flatten())
                    tmp = np.flip(tmp)
                    lst.append( tmp )
                else:
                    tmp = list(data[row-self.AR:row+oneMore])
                    tmp = np.flip(tmp)
                    #tmp.append(data[row])
                    lst.append( tmp )
        shifted = np.array(lst)
        return shifted



    def _generate_endog_exog_data(self,y,X):
        shifted_y = self._create_shifted_dataset(y, include_current_time=self.isfilter)
        if X is not None:
            shifted_X = self._create_shifted_dataset(X, include_current_time=True)
            if self.include_endog:
                shifted = np.concatenate((shifted_y, shifted_X), axis=-1)
            else:
                shifted = shifted_X
        else:
            shifted = shifted_y
        return shifted



    def train_test_split(self,
                         y: np.array,
                         X: np.array) -> np.array and np.array    \
                                         and np.array and np.array:
        '''
        Timeseries train/test splitting uses
        the first (1-validation_frac) for the training
        set and the last validation_frac for the test set.

        Unless the 'test_train_split_method' is set to random in
        which the (1-validation_frac) is randomly selected.

        Parameters
        ----------
        data : np.array

        Returns
        -------
        train_y : np.array
        train_X : np.array
        test_y : np.array
        test_X : np.array

        '''

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(X, np.ndarray):
            X = np.array(X)


        data_length = y.shape[0] - self.AR
        n_train = int( (1 - self.validation_frac)*data_length)
        n_test = int( data_length - n_train)
        self.n_train = n_train

        #y is shifted to account for AR lags
        y_lag = y[self.AR:]

        if self.test_train_split_method == 'timeseries':
            train_y = y_lag[:n_train]
            test_y = y_lag[-n_test:]

            if len(X.shape) > 1:
                train_X = X[:n_train,:]
                test_X = X[-n_test:,:]
            else:
                train_X = X[:n_train]
                test_X = X[-n_test:]
        else:
            idx = np.random.choice(range(data_length),replace=False, size = n_train)
            self.idx_rnd_sample = idx

            train_y = y_lag[idx]
            test_y = np.delete(y_lag,idx, axis=0)

            train_X = X[idx]
            test_X = np.delete(X,idx, axis=0)


        return train_y, train_X, test_y, test_X


    def train(self,y: np.array,
                   X: np.array=None,
                   patience:int=500,
                   weights:np.array=None) -> None:
         '''

         Train the neural network.

         Parameters
         ----------
         y : np.array
         X : np.array, optional
         patience, int, optional
             Early stoppage criteria. Number of iterations
             with no improvement in the validation set.  The
             model with best validation score is kept.

         '''


         if X is not None and self.n_exog_variables ==0:
             raise ValueError("ERROR: input columns must equal 1 + n_exog_variables")


         shifted = self._generate_endog_exog_data(y,X)
         train_y, train_X, test_y, test_X = self.train_test_split(y, shifted)
         if weights is not None:
             if self.test_train_split_method == 'timeseries':
                 weights = weights[:self.n_train]
             else:
                 #idx_rnd_sample set during the train_test_split
                 weights = weights[self.idx_rnd_sample]


         if self.scale_y:
            self.scale_mu_y = np.mean(train_y)
            self.scale_sigma_y = np.std(train_y)
            train_y = (train_y - self.scale_mu_y) / self.scale_sigma_y
            test_y = (test_y - self.scale_mu_y) / self.scale_sigma_y

         self.y = y
         self.X = X
         #self.train_y, self.train_X, self.test_y, self.test_X = train_y, train_X, test_y, test_X

         callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=patience,
                                                     restore_best_weights=True)
         self.history = self.model.fit(
                           train_X,train_y,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_data = (test_X, test_y),
                           sample_weight=weights,
                           callbacks=[callback]
                           )


    def get_estimate(self, y: np.array=None,
                           X: np.array=None) -> np.array:
        '''

        Make a prediction with ANN.

        Parameters
        ----------
        y : np.array
            Endogenous Variable
        X : np.array
            Exogenous Variables

        Returns
        -------
        predictions : np.array
        '''


        shifted = self._generate_endog_exog_data(y,X)
        if len(shifted) == 0:
            raise ValueError("ERROR: Data length must be greater than AR terms.")

        if self.scale_y:
            return self.model.predict(shifted)*self.scale_sigma_y + self.scale_mu_y
        else:
            return self.model.predict(shifted)



    def plot(self, save_fig = False) -> None:
        '''
        Generates the following plots:

            *a plot of the training and validation
             loss functions over the number of epochs.
            *Residual vs yhat and yhat vs y
            *Timeseries of y and the fitted line yhat

        Parameters
        ----------
        save_fig : bool, optional
            Save a jpg of the figure. The default is False.

        Returns
        -------
        None.

        '''

        plot.figure(figsize=(14,5))
        plot.plot(self.history.history['loss'],label='loss')
        plot.plot(self.history.history['val_loss'],label='val_loss')
        plot.ylabel("Loss")
        plot.xlabel("Epoch")
        plot.legend()
        plot.tight_layout()
        if save_fig:
            plot.savefig('twinstat_AR-ANN_losses.jpg',dpi=600)


        # train_y, train_X, test_y, test_X = ANN.train_test_split(ANN.y, ANN.X[ANN.AR:])
        # y_hat = ANN.get_estimate(y=test_y, X=test_X)
        # y = test_y[ANN.AR:]
        # residuals = y - y_hat.ravel()

        if self.X is not None:
            train_y, train_X, test_y, test_X = self.train_test_split(self.y, self.X[self.AR:])
            y_hat_train = self.get_estimate(y=train_y, X=train_X)
            y_hat = self.get_estimate(y=test_y, X=test_X)
        else:
            train_y, train_X, test_y, test_X = self.train_test_split(self.y, self.y)
            y_hat_train = self.get_estimate(y=train_y)
            y_hat = self.get_estimate(y=test_y)

        y_train = train_y[self.AR:]
        y = test_y[self.AR:]

        self.residuals = y - y_hat.ravel()

        self.rMSE = np.std(self.residuals)
        print("rMSE: {:.3f} and {:.3f}%".format(self.rMSE, self.rMSE/np.mean(y)*100 ))


        plot.figure(figsize=(15,5))
        plot.subplot(131)
        plot.plot([min(y), max(y)], [min(y), max(y)],
                  color='black', linestyle='solid')
        plot.scatter(y, y_hat)
        #plot.scatter(test_y[ANN.AR:], y_hat)
        plot.ylabel("y_hat")
        plot.xlabel("y")
        plot.tight_layout()

        plot.subplot(132)
        plot.scatter(y_hat, self.residuals)
        plot.hlines(0,min(y_hat), max(y_hat), color='black', linestyle='solid')
        plot.ylabel("Residuals")
        plot.xlabel("y_hat")
        plot.tight_layout()


        plot.subplot(133)
        plot.scatter(y_hat, self.residuals/np.mean(y_hat))
        plot.hlines(0,min(y_hat), max(y_hat), color='black', linestyle='solid')
        plot.ylabel("Residuals (fraction)")
        plot.xlabel("y_hat")
        plot.tight_layout()

        if save_fig:
            plot.savefig('twinstat_AR-ANN_residuals.jpg',dpi=600)

        t = range(y_hat.shape[0])
        plot.figure(figsize=(15,7))
        plot.title("Test Set")
        plot.scatter(t, y, label='Data')
        plot.plot(t,y_hat, color='red', linestyle='dotted', label = 'Fitted Line')
        plot.ylabel("y")
        plot.xlabel("Time")
        plot.legend()
        plot.tight_layout()

        t = range(y_hat_train.shape[0])
        plot.figure(figsize=(15,7))
        plot.title("Training Set")
        plot.scatter(t, y_train, label='Data')
        plot.plot(t,y_hat_train, color='red', linestyle='dotted', label = 'Fitted Line')
        plot.ylabel("y")
        plot.xlabel("Time")
        plot.legend()
        plot.tight_layout()

        if save_fig:
            plot.savefig('twinstat_AR-ANN_timeseries.jpg',dpi=600)



    def save_model(self, filename:str) -> None:
        ''' save the model for later use'''
        self.model.save(filename+'.h5')
        dict_to_save = {key:value for key,value in vars(self).items() if key not in ['model', 'history']}
        np.savez(filename+'.npz', **dict_to_save)

    def load_model(self, filename:str) -> None:
        ''' load the model weights and config values'''
        npzfile = np.load(filename+'.npz')
        for key,value in npzfile.items():
            #print(key, len(value.shape))
            if len(value.shape) > 0:
                vars(self)[key] = value
            else:
                vars(self)[key] = value.item()
        #moved to object specific, see inherted objects
        #self.model.load_weights(filename+'.h5')

    #TODO: add this in
    def _determine_variable_sensitivity(self):
        ''' run shapely '''