from datetime import datetime
import math
import multiprocessing
import numpy as np
import os
import tuning
import util
import xor

def xor_spread(kwargs, var, values, f=None, prefix=None):
    kw2 = dict(kwargs)
    kw2["func"] = xor.solve_xor
    kw2["func_name"] = "Xor"
    kw2["pool"] = pool
    kw2["n_runs"] = n_runs
    return tuning.func_spread(kw2, var, values, f, prefix)

def run1():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    #xor_avg({'max_steps':20000})
    f = open("xor_tuning.txt", "w")
    r = range(100, 201, 100)
    xor_spread(kwargs, "seed_base", r, f=f, prefix="Prune_HM")
    kwargs['enable_homomorphic_propagation'] = False
    #xor_spread(kwargs, "seed_base", r, f=f, prefix="Prune_NHM")
    
def run_epsilon():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    f = open("xor_tuning.txt", "w")
    r = [0.1, 0.03, 0.01, 0.003, 0.001]
    xor_spread(kwargs, "fitness_epsilon", r, f=f, prefix="HM")
def run_mc_base():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    f = open("xor_tuning.txt", "w")
    r = [1.5, 1.75, 2.0, 2.25, 2.5]
    xor_spread(kwargs, "mc_base", r, f=f, prefix="HM")
def run_dmg():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    #kwargs['enable_dmg'] = False
    #kwargs["start_with_connected_progenitor"] = True
    #kwargs["perturb_mutation_weight"] = 10
    kwargs["pruning_discount_factor"] = 0.8
    kwargs["dead_end_threshold"] = 170
    kwargs["subprogenitor_discount_factor"] = 0.9
    kwargs["enable_homomorphic_propagation"] = False
    
    f = open("xor_tuning.txt", "w")
    r = [False, True]
    #xor_spread(kwargs, "enable_dmg", r, f=f)
    xor_spread(kwargs, "start_with_connected_progenitor", r, f=f)
    #xor_spread(kwargs, "seed_base", [0], f=f)
    
    
def run_pruning():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    f = open("xor_tuning.txt", "w")
    r = range(140, 186, 5)
    xor_spread(kwargs, "dead_end_threshold", r, f=f, prefix="spf_orig")
    kwargs1 = dict(kwargs)
    kwargs1["subprogenitor_discount_factor"] = 0.9
    xor_spread(kwargs1, "dead_end_threshold", r, f=f, prefix="spf_reduced")
    
    r = np.arange(0.8, 0.99, 0.02)
    xor_spread(kwargs, "subprogenitor_discount_factor", r, f=f, prefix="det_orig")
    kwargs1 = dict(kwargs)
    kwargs1["dead_end_threshold"] =150
    xor_spread(kwargs1, "subprogenitor_discount_factor", r, f=f, prefix="det_reduced") 
    

def run_pdf():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    f = open("xor_tuning.txt", "w")
    r = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    xor_spread(kwargs, "perturb_mutation_weight", r, f=f, prefix="pdf")
    r = [0.7, 0.8, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.01, 1.03, 1.10, 1.30]
    xor_spread(kwargs, "pruning_discount_factor", r, f=f, prefix="pdf")


def run3():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    kwargs["start_with_connected_progenitor"] = True
    kwargs["perturb_mutation_weight"] = 10
    kwargs["pruning_discount_factor"] = 0.8
    #kwargs["dead_end_threshold"] = 100
    kwargs["subprogenitor_discount_factor"] = 0.9
    kwargs['enable_homomorphic_propagation'] = False
    kwargs["mc_base"] = 1.2
    kwargs["neuron_cost"] = 0.05
    f = open("xor_tuning.txt", "w")
    r = [100] 
    xor_spread(kwargs, "dead_end_threshold", r, f=f)

def run_pivot_mutations():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    kwargs["start_with_connected_progenitor"] = True
    kwargs["perturb_mutation_weight"] = 10
    kwargs["pruning_discount_factor"] = 0.8
    kwargs["dead_end_threshold"] = 1000
    kwargs["subprogenitor_discount_factor"] = 0.9
    kwargs['enable_homomorphic_propagation'] = False
    kwargs["mc_base"] = 1.2
    kwargs["neuron_cost"] = 0.05
    f = open("xor_tuning.txt", "w")
    #r = [None, [1,1,1], [1,2,1], [2,1,1]]
    r = [[1,1,1]] 
    xor_spread(kwargs, "pivot_mutation_weights", r, f=f)



def run4():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    kwargs["start_with_connected_progenitor"] = True
    kwargs["perturb_mutation_weight"] = 10
    kwargs["pruning_discount_factor"] = 0.8
    kwargs["dead_end_threshold"] = 1000
    kwargs["subprogenitor_discount_factor"] = 0.9
    kwargs['enable_homomorphic_propagation'] = False
    kwargs["mc_base"] = 1.2
    kwargs["neuron_cost"] = 0.05
    kwargs["pivot_mutation_weights"] = [1,1,1]
    f = open("xor_tuning.txt", "w")
    r = [1,2,3,4,5,7,9,12,15]
    xor_spread(kwargs, "mc_a", r, f=f)



def run5():    
    kwargs = {}
    kwargs['max_steps'] = 20000
    kwargs["start_with_connected_progenitor"] = True
    kwargs["perturb_mutation_weight"] = 10
    kwargs["pruning_discount_factor"] = 0.8
    kwargs["dead_end_threshold"] = 1000
    kwargs["subprogenitor_discount_factor"] = 0.9
    kwargs['enable_homomorphic_propagation'] = False
    #kwargs["mc_base"] = 1.2
    kwargs["neuron_cost"] = 0.05
    kwargs["fitness_threshold"] = 13.0
    kwargs["pivotize_threshold"] = 20
    kwargs["pivot_mutation_weights"] = [1,1,1]
    f = open("xor_tuning.txt", "w")
    r = [1.2, 2.0]
    xor_spread(kwargs, "mc_base", r, f=f)

pool = None
n_runs = 20
if __name__ == '__main__':
    multiprocessing .freeze_support()
    pool = multiprocessing.Pool(3)
    #run_pruning()
    #run3()
    #run_pivot_mutations()
    run5()

