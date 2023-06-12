# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


import matplotlib.pyplot as plot
from autogluon.tabular import TabularPredictor
import pandas
from sklearn.model_selection import train_test_split
import numpy as np

class autogluon(object):
    '''
    Enables low effort utilization of autogluon for echelon model
    development and hyperparameter tuning.  Specifically setup for
    only tabular data.

    See https://auto.gluon.ai/dev/index.html for more options and
    customization.


    '''
    def __init__(self, endogenous_variable:str,
                       save_path:str = 'twinstat_autogluon',
                       num_trials:int = 10,
                       search_strategy:str='auto',
                       eval_metric:str='rmse',
                       problem_type:str ='regression',
                       validation_frac:float=0.1,
                       preload:bool=True):
        '''


        Parameters
        ----------
        endogenous_variable : str
            Variable name of the output variable.
        save_path : str, optional
            File directory path where the Autogluon files will be saved.
            The default is 'twinstat_autogluon'.
        num_trials : int, optional
            Number of hyperparameter tuning iterations. The default is 10.
        search_strategy : str, optional
            Hyperparameter tuning method. The default is 'auto'.
        eval_metric : str, optional
            Metric used . The default is 'rmse'.
        problem_type : str, optional
             The default is 'regression'.
        validation_frac : float, optional
            Fraction of data held in the validation test. The default is 0.1.
        preload : bool, optional
            At inference time, autogluon can be quite slow when making predictions.
            This option will preload all of the models into RAM enabling significantly
            faster predictions, albeit at the cost of more RAM usage.
            The default is True.

        Returns
        -------
        None.

        '''

        self.save_path='./AutogluonModels/' + save_path
        self.endogenous_variable = endogenous_variable
        self.validation_frac = validation_frac
        self.preload = preload

        self.hyperparameter_tune_kwargs = {
                                             'num_trials': num_trials,
                                             'scheduler' : 'local',
                                             'searcher': search_strategy,
                                          }

        self.models = TabularPredictor( label=endogenous_variable,
                                        eval_metric=eval_metric,
                                        path=self.save_path,
                                        problem_type =problem_type )


    def train(self, data:pandas.DataFrame,
                    num_gpus:int=0) -> None:
        '''

        Training include Monte Carlo hyperparameter search
        for each model used in training.

        Parameters
        ----------
        data : pandas.DataFrame
            Data includes both the model inputs and outputs
        num_gpus : int, optional
            The default is 0.

        Returns
        -------
        None

        '''

        #split data, autogluon has seperate timeseries automation
        #not being used here
        y = data[self.endogenous_variable]
        X = data.drop([self.endogenous_variable],axis=1)

        train_X, test_X, train_y, test_y = train_test_split(np.array(X), np.array(y),
                                                            test_size = self.validation_frac )

        #autogluon wants one dataframe
        train_data = pandas.DataFrame(train_X)
        train_data.columns = X.columns
        train_data[self.endogenous_variable] = train_y

        test_data = pandas.DataFrame(test_X)
        test_data.columns = X.columns
        test_data[self.endogenous_variable] = test_y

        self.train_data = train_data
        self.test_data = test_data

        #run the automl which both performs a monte carlo search and seeks to combine
        #many models for better accuracy.  All models are save to the save_path location
        self.models.fit(num_gpus=num_gpus,
                        train_data=train_data,
                        hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs,
                        presets='best_quality')

        #Evaluate how the models performed
        y_hat = self.models.predict(test_data)
        perf = self.models.evaluate_predictions(y_true=test_data[self.endogenous_variable],
                                                y_pred=y_hat,
                                                auxiliary_metrics=True)
        self.models.fit_summary(show_plot=False)

        for key, value in perf.items():
            #autogluon multiplies the loss metrics by -1 to enable
            #minimizing with their solvers.  Here we are flipping the
            #sign back to make it easier to read
            if key =='r2' or key == 'pearsonr':
                flip = 1
            else:
                flip = -1
            print("{0:<25} {1:>15.3f}".format(key, value*flip))

        if self.preload:
             #to speed up inference, preload all models into RAM
             self.models._learner.persist_trainer(low_memory=False)

    def predict(self, data:pandas.DataFrame) -> pandas.core.series.Series:
        return self.models.predict(data)

    def load_models(self) -> None:
        self.models = TabularPredictor.load(self.save_path)

        if self.preload:
            #to speed up inference, preload all models into RAM
            self.models._learner.persist_trainer(low_memory=False)

    def determine_variable_sensitivity(self) -> None:
        '''
        Performs permutation importance sampling to determine
        model sensitivity to inputs.

        A matplotlib figure is generated with the results.

        Returns
        -------
        None

        '''

        self.test_data.columns = list(map(str,self.test_data.columns))
        sensitivity = self.models.feature_importance(self.test_data)

        plot.figure(figsize=(16,7))
        plot.bar(sensitivity.index, sensitivity['importance'])
        plot.hlines(0, sensitivity.index[0], sensitivity.index[-1], color='black')
        plot.ylabel('Sensitivity')
        plot.xlabel('Variables')
        plot.tight_layout()
        plot.savefig("autogluon_model_sensitivity.png")



#%% main
if __name__ == '__main__':

    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    data = X.copy()
    data['y'] = y
    ag = autogluon('y', num_trials=2)

    ag.train(data)

    yhat = ag.predict(data)
    residual = y - yhat

    plot.figure()
    plot.scatter(yhat, residual)
    plot.hlines(0,np.min(yhat), np.max(yhat), color='black')
    plot.ylabel("Residual")
    plot.xlabel("yhat")
    plot.tight_layout()

    ag.determine_variable_sensitivity()








