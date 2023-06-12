# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:55:47 2022

@author: rpivovar
"""

import numpy as np
from twinstat.statespace_models.estimators import particle_filter


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


def test_particles():

    y = get_random_data()

    pobj = particle_filter(y=y, n_particles=50000)
    state_means, _ = pobj.get_estimate(y)
    print(state_means)


    good_arr = np.array([0.0005785965375011505, 0.11719976275366761, 0.25538406088516347, 0.2811239856154304, 0.4022647236286647,
                        0.45091168430371453, 0.4969006180121733, 0.4947956998154076, 0.5985895888072588, 0.643314952581112,
                        0.6613862110212009, 0.7505527491426432, 0.8916610962592685, 0.9754625341621693, 0.96396060077085,
                        1.0556330732608656, 1.0706615680734646, 1.084772644336855, 1.107151543089811,
                        1.1206594093198285, 1.1392474414150555, 1.2014392219004246, 1.2215832485029914,
                        1.2237517854453959, 1.2656907308432268, 1.3637307308866806, 1.4154438248085082,
                        1.4471052589318274, 1.5386605177219805, 1.6010045936537074, 1.5990515096523612,
                        1.5204742072382798, 1.4649065346535797, 1.4220317858530205, 1.385927944760157,
                        1.3871036996621646, 1.272031877563537, 1.2688274975843308, 1.2641966903902428,
                        1.3245762322751917, 1.4344882912759755, 1.4391654794032895, 1.4242451400183753,
                        1.4496763183288512, 1.4746106390922111, 1.4375356515868691, 1.425369746167067,
                        1.4569915064413963, 1.4163468619488284, 1.340192520071694, 1.2464954193602387])

    #the particles package appears to not support seeding without some internal modification
    #thus the approach here is to use a large number of particles and loose tolerance
    compare = np.allclose(state_means,good_arr, rtol=1e-2,atol=1e-2)
    assert compare == True


