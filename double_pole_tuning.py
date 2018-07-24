from datetime import datetime
import double_pole
import math
import multiprocessing
import numpy as np
import os
import tuning
import util

def double_pole_spread(kwargs, var, values, f=None, prefix=None):
    kw2 = dict(kwargs)
    kw2["func"] = double_pole.solve_double_pole
    kw2["func_name"] = "Pole"
    kw2["pool"] = pool
    kw2["n_runs"] = n_runs
    return tuning.func_spread(kw2, var, values, f, prefix)


def run1():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    #kwargs["start_with_connected_progenitor"] = True
    #kwargs["perturb_mutation_weight"] = 10
    #kwargs["mc_base"] = 1.2
    kwargs["pruning_discount_factor"] = 0.9
    kwargs["dead_end_threshold"] = 400
    kwargs["subprogenitor_discount_factor"] = 0.9
    kwargs['enable_homomorphic_propagation'] = False
    kwargs["neuron_cost"] = 0.05
    kwargs["seed_base"] = 1000
    f = open("dp_tuning.txt", "w")
    r = [10]
    #double_pole_spread(kwargs, "perturb_mutation_weight", r, f=f)
    kwargs["start_with_connected_progenitor"] = True
    double_pole_spread(kwargs, "perturb_mutation_weight", r, f=f)


pool = None
n_runs = 1000
if __name__ == '__main__':
    multiprocessing .freeze_support()
    pool = multiprocessing.Pool(3)
    run1()

