# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

import os
import numpy as np
import shap
import pandas
from tqdm import tqdm
from twinstat.timeseries_forecast.GP_model import GPModel
import matplotlib.pyplot as plot
import seaborn as sns
sns.set(style="darkgrid")


def shapely_sensitivity(inputs:list[str], outputs:list[str], df:pandas.DataFrame):
    '''
    Determine the shapely sensitivities of each 'output' variable.

    A TwinStat Gaussian Process is used to map the inputs to outputs and shapely
    sampling is performed on the Gaussian Process.

    The package shap is used for the shap sampling.
    https://shap.readthedocs.io/en/latest/index.html

    Parameters
    ----------
    inputs : list[str]
        List of the names of inputs.
    outputs : list[str]
        List of the names of outputs.  The sensivitiy of the inputs to
        each output will be calculated and a new gaussian process made
        for each input/output combination.
    data : pandas.DataFrame

    Returns
    -------
    dictionary
        keys: output variables
        values: list of shapely values normalized to sum to 1.0

    '''

    n_inputs = len(inputs)
    n_eval = df.shape[0]

    if not os.path.isdir('images'):
        os.mkdir('images')

    samples = df[inputs].to_numpy()
    normalized_samples = (samples - np.min(samples,axis=0)) / (np.max(samples,axis=0) - np.min(samples,axis=0))

    plot.ioff()
    sensitivities = {}
    for output in tqdm(outputs):
        target_output = df[output].to_numpy()
        target_output = target_output / np.max(target_output)

        #setup a model to map the change in inputs to the output pdf
        GP = GPModel( normalized_samples, target_output,
                      lr=0.1,
                      base_kernel='rbf',
                      auto_stop_tol=1e-4,
                      mean_function="constant"
                      )
        try:
            #some large number, hopefully the auto stop will terminate this much sooner
            GP.train(15000)

            # depending on the results, might cause training to crash,
            # e.g. all same value, etc
        except:
            sensitivities[output] = [0]*n_inputs
            continue

        # some formatting needed to work with shap package
        def f(X):
            return np.array(GP.get_estimate(X).mean)

        #TODO: do we need to make the 100 and 500 as optional?

        #needs a represenative sample, should be a random sample
        idx = np.random.choice(range(normalized_samples.shape[0]), 100)
        explainer = shap.KernelExplainer(f, normalized_samples[idx, :])

        mu = np.mean(normalized_samples, axis=0)
        shap_values = explainer.shap_values(mu, nsamples=500)
        #normalize values for improved understanding
        shap_values = shap_values / np.sum(np.abs(shap_values))
        sensitivities[output] = shap_values

        series = pandas.Series(shap_values)
        plot.figure(figsize=(12,4))
        plot.title('Output:' + output)
        ax = series.plot(kind="bar")
        plot.hlines(0,-1,n_inputs, color='black')
        ax.set_xlabel("Input Variable")
        ax.set_ylabel("Shapely Sensitivity")
        ax.set_xticklabels(inputs)
        plot.tight_layout()
        plot.savefig("./images/sensitivity_{}.png".format(output))
        plot.close()

    plot.ion()
    return sensitivities

#--------------------------------------------------------------------------


#%% main
if __name__ == '__main__':

    ''' Example.'''