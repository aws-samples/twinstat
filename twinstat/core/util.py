# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


from scipy.integrate import cumtrapz

def pdf_to_cdf(x,p):
    '''
    Convert a pdf to cdf

    Parameters
    ----------
    x : np.array
        The random variable output.
    p : np.array
        The probability density of x

    Returns
    -------
    cdf : np.array
        The cumulative probability

    '''
    cdf = cumtrapz(p, x, initial=0)
    cdf /= cdf[-1]
    return cdf