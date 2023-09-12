# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


import os
import random
import chaospy as cp
import copy
import numpy as np
from scipy import stats
import matplotlib.pyplot as plot
import seaborn as sns
sns.set(style="darkgrid")

from twinstat.core.util import pdf_to_cdf
from twinstat.core.sensitivity_analysis import shapely_sensitivity

from twinmodules.AWSModules.AWS_secrets import get_secret
from twinmodules.core.sql_databases import SQLHandler, get_dataframe_results


def uncertainty_propagation(evaluate_data,
                            config:dict,
                            pce_degree:int = 6,
                            method:str = 'monte_carlo',
                            sampling_method:str = 'latin_hypercube',
                            seed: int = 0) -> None:
    '''

    chaospy provides utility functions for generating joint
    probablity sampling distributions.

    The actual creation of a the PCE becomes exponentially slower with
    the number of variables and thus basic MC might be needed for higher
    dimension count.

    Also, if the objective function runtime is fast, no need for PCE.

    Function assumes the TwinModules function 'setup_uncertainty_propagation_db'
    was used to setup an RDS database for data storage.

    Upon completion of the posterior distribution calculation, this function
    will automatically run the 'shapely_sensitivity' to determine which
    inputs have the greatest impact on the outputs.

    Parameters
    ----------
    evaluate_data : function
        Python function that accepts a vector X
    config : dict
        Contains:
            'num_samples': int, number of sobel samples
            'sample_dist_input_#': list, [str, float]
                [sampling distribution, inputs into sampling distribution]
            'result_#': str, outputs of the evaluate_data function
            'secret_name' : str, AWS secret name providing security credentials
            'region_name' : str, AWS region of the RDS database
            'mysql_db_endpoint' : str, location of SQL database such as AWS RDS endpoint

            Example:

            .. code-block:: python

                  {
                    'num_samples': int, number of sobel samples
                    'sample_dist_input_0': ["TruncNormal", 1e-3, 100, 0, 0.01],
                    "sample_dist_input_1" : ["Uniform", 1e-3, 0.3],
                    "result_0" : "deg",
                    "result_1" : "ms",
                    "result_2" : "LoadCellTension1_N_1",
                    }

    pce_degree : int, optional
        PCE polynomial order.
        The default is 6.
    method : str, optional
        Either 'monte_carlo' or 'pce'.

        Both methods will use use sobol sampling of the user provided sampling
        distributions.

        'pce' : polynomial chaos expansion requires appreciably less samples than
                traditional monte carlo methods to determine the posterior distribution.
                However, the complexity of the mapping polynomial scales poorly. Depending
                on the user defined sampling distribution, pce may become prohibitively
                slow for higher dimensions, e.g. > ~20

        'monte_carlo' :

        The default is 'monte_carlo'.

    sampling_method : str, optional
        Sets how will the input distributions be sampled.

        Using purely random sampling is typically not advisable due to the inefficiency
        associated with natural clustering.

        A space filling method is recommended, but there are many options.

        References discussing Sobol and Latin Hyper Cube:

        Mainly discussing integration; concludes Sobol is best for
        integration:

            https://arxiv.org/ftp/arxiv/papers/1505/1505.02350.pdf

        Specifically discussing uncertainty propagation; concludes
        Latin Hyper Cube generally is best for uncertainty quantification:

            https://www.osti.gov/servlets/purl/806696

        For a sample size of 100,000 with 9 dimensions

        Latin Hyper Cube is ~21.0% slower than brute Monte Carlo
        Sobol is ~438.0% slower than brute Monte Carlo

        The default is 'latin_hypercube'.

    seed : int, optional
        The default is 0.

    Returns
    -------
    None

    '''

    bounds = [x for x in config.keys() if "sample_dist" in x]
    if len(bounds) == 0:
        raise ValueError("ERROR: sample_dist bounds must be provided for inputs")
    variables = [getattr(cp, config[g][0] )(*config[g][1:]) for g in bounds ]
    joint = cp.J(*variables)

    num_samples = config['num_samples']

    if not os.path.isdir('images'):
        os.mkdir('images')

    # determine the user defined inputs and outputs
    inputs = [config[x] for x in config.keys() if "input" in x and 'sample_' not in x]
    outputs = [config[x] for x in config.keys() if "result" in x ]

    unc_prop_results = {}

    local_config = copy.deepcopy(config)
    local_config['eval_table'] = 'UncertaintyPropagationSamples'
    local_config['uid'] = random.getrandbits(24)

    if method.lower() == 'pce':
        print("Generating PCE expansion.")
        pce = cp.generate_expansion(pce_degree, joint)

        X = joint.sample(num_samples, seed=seed, rule=sampling_method)
        evaluate_data(X.T, local_config)
        df = get_dataframe_results(config, 'UncertaintyPropagationSamples')

        for output in outputs:
            y = df[output]

            nx = 300
            if np.std(y) < 1e-8:
                unc_prop_results[output]['pdf'] = [0]*nx
                unc_prop_results[output]['cdf'] = [0]*nx
                unc_prop_results[output]['xtest'] = [0]*nx
            else:
                #Fit the polynomial chaos expansion
                pce_model = cp.fit_regression(pce, X, y)

                xtest = np.linspace( np.min(y), np.max(y), num=nx)
                qoi_dist = cp.QoI_Dist(pce_model, joint)
                pdf = qoi_dist.pdf(xtest)

                cdf = pdf_to_cdf(xtest,  pdf )
                #expected = cp.E(pce_model, joint)
                unc_prop_results[output] = {}
                unc_prop_results[output]['pdf'] = pdf
                unc_prop_results[output]['cdf'] = cdf
                unc_prop_results[output]['xtest'] = xtest
    else:
        #sobol ensures more efficient sample space coverage relative to brute force
        X = joint.sample(num_samples, seed=seed, rule=sampling_method)
        X = X.T

        evaluate_data(X, local_config)

        df = get_dataframe_results(config, 'UncertaintyPropagationSamples')

        for output in outputs:
            y = df[output]
            unc_prop_results[output] = {}

            nx = 300
            if np.std(y) < 1e-8:
                unc_prop_results[output]['pdf'] = [0]*nx
                unc_prop_results[output]['cdf'] = [0]*nx
                unc_prop_results[output]['xtest'] = [0]*nx
            else:
                pdf = stats.gaussian_kde(y)
                xtest = np.linspace( np.min(y), np.max(y), num=nx)
                pdfi = pdf(xtest)
                cdfi = pdf_to_cdf(xtest,  pdfi )
                unc_prop_results[output]['pdf'] = pdfi
                unc_prop_results[output]['cdf'] = cdfi
                unc_prop_results[output]['xtest'] = xtest

    #process results-------------------------

    #generate uncertainty figures

    plot.ioff()
    for output in outputs:
        plot.figure()
        plot.plot(unc_prop_results[output]['xtest'],  unc_prop_results[output]['pdf'] )
        plot.ylabel("Probability Density")
        plot.xlabel(output)
        plot.tight_layout()
        plot.savefig(f"./images/{output}_pdf.png")
        plot.close()


        plot.figure()
        plot.plot(unc_prop_results[output]['xtest'],  unc_prop_results[output]['cdf'] )
        plot.ylabel("Cummulative Distribution")
        plot.xlabel(output)
        plot.tight_layout()
        plot.savefig(f"./images/{output}_cdf.png")
        plot.close()
    plot.ion()


    #perform sensitivity analysis
    if df.shape[0] > 1000:
        sensitivities = shapely_sensitivity(inputs, outputs, df.sample(1000) )
    else:
        sensitivities = shapely_sensitivity(inputs, outputs, df )

    #TODO: need to make an option that is not specific to AWS
    # record results
    secret = get_secret(config['secret_name'], config['region_name'])

    with SQLHandler(secret, config['mysql_db_endpoint'], config['database_name']) as sql:

        for col in unc_prop_results.keys():
            data = np.array([unc_prop_results[col]['pdf'],
                             unc_prop_results[col]['xtest']])
            data = data.T
            columns = [col+'_pdf', col+'_xtest']
            sql.send_sql_data('UncertaintyPropagationPDF', columns, data )

            data = np.array([unc_prop_results[col]['cdf'],
                             unc_prop_results[col]['xtest']])
            data = data.T
            columns = [col+'_cdf', col+'_xtest']
            sql.send_sql_data('UncertaintyPropagationCDF', columns, data )


        for col in sensitivities.keys():
            columns = ['output_variable']
            columns.extend(inputs)

            data = [col]
            data.extend(sensitivities[col])
            data = [data]
            sql.send_sql_data('UncertaintyPropagationSensitivity', columns, data )