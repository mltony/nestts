import multiprocessing
import numpy as np
import tuning
import util
import xor

def xor_investigate(
        seed=None,
        dmg1=False,
        dmg2=False,
        te_mc=40.0,
        te_mc_mu=5.0,
        **kwargs):
    #kwargs = {}
    explorer = xor.solve_xor(
        do_explore=False,
        seed=seed,
        **kwargs)
    if False:
        pivot = explorer.homomorphic_optimize(explorer.progenitor)
        print("Pivot = %.2f" % pivot.fitness)
        print(str(pivot.network))

        max_mc = 20
        attempts = 100
        for mc in range(10, 51, 10):
            successes = 0
            scores = []
            for attempt in range(attempts):
                sp = explorer.explore_topology(pivot, mc)
                spo = explorer.homomorphic_optimize(sp)
                if spo.is_winner:
                    successes += 1
                scores.append(spo.fitness)
            print("%d %d" % (mc, successes))
            #print(sorted(scores, reverse= True))
    if True:
        while True:
            while True:
                explorer.progenitor.explored_count = 0
                pivot = explorer.homomorphic_optimize(explorer.progenitor, dmg=dmg1)
                if pivot is not explorer.progenitor:
                    break 
            for i in range(10):
                mc = explorer.random.gauss(te_mc, te_mc_mu)
                mc = int(mc)
                mc = max(mc, 1)
                sp = explorer.explore_topology(pivot, mc)
                spo = explorer.homomorphic_optimize(sp, dmg=dmg2)
                if spo.is_winner:
                    steps = explorer.step
                    #print("Steps = %d" % steps)
                    return (steps, spo.get_size())
                
    
def xor_investigate_average():
    n_runs = 100
    x = []
    for i in range(n_runs):
        result =xor_investigate(i) 
        print("xor(%d) = %d" % (i, result))
        x.append(result)
    #x = [xor_investigate() for i in range(n_runs)]
    print("Average=%.2f" % util.avg(x))
    print("Median=%.2f" % util.median(x))


def xor_investigate_spread(kwargs, var, values, f=None, prefix=None):
    kw2 = dict(kwargs)
    kw2["func"] = xor_investigate
    kw2["func_name"] = "Xor"
    kw2["pool"] = pool
    kw2["n_runs"] = n_runs
    return tuning.func_spread(kw2, var, values, f, prefix)
# run1 yields 3600 on many values, but 3480 with mc_a=8
def run1():    
    kwargs = {}
    #kwargs["dmg2"] = True
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
    f = open("xor_investigate.txt", "w")
    r = range(1, 11)
    xor_investigate_spread(kwargs, "mc_a", r, f=f)


# yields 3200 on mc_a=8 and te_mc = 25
def run2():    
    kwargs = {}
    #kwargs["dmg2"] = True
    kwargs["te_mc"] = 25
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
    f = open("xor_investigate.txt", "w")
    r = [1,6,8]
    xor_investigate_spread(kwargs, "mc_a", r, f=f)

class MyConstantMcClass(object):
    def __init__(self, i):
        self.i = i
    def __call__(self, org, random):
        return self.i
    def __str__(self):
        return "const_mc_%d" % self.i
    #def my_mc_func(org, random):
    #    return mc
def constant_mc_func(mc):
    return MyConstantMcClass(mc)

class MyProportionalMcClass(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, org, random):
        network = org.network
        connections = sum([len(n.connections) + len(n.incoming_connections) for n in network.neurons.values()]) / 2
        #print(connections)
        result = int(self.p * connections)
        result = max(result, 2)
        #print(result)
        return result
    def __str__(self):
        return "proportional_mc_%.2f" % self.p


def run3():    
    kwargs = {}
    #kwargs["dmg2"] = True
    kwargs["te_mc"] = 25
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
    kwargs["seed_base"] = 1000
    f = open("xor_investigate.txt", "w")
    r = map(constant_mc_func, range(3,4,1))
    xor_investigate_spread(kwargs, "mc_func", r, f=f)
    #kwargs["seed_base"] = 100
    #xor_investigate_spread(kwargs, "mc_func", r, f=f,prefix="sb100")


def run4():    
    kwargs = {}
    #kwargs["dmg2"] = True
    kwargs["te_mc"] = 25
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
    f = open("xor_investigate.txt", "w")
    r = map(MyProportionalMcClass, np.arange(.05, .16, .025))
    xor_investigate_spread(kwargs, "mc_func", r, f=f)


pool = None
n_runs =10
if __name__ == '__main__':
    multiprocessing .freeze_support()
    pool = multiprocessing.Pool(3)
    run3()
