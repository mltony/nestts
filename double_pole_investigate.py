import double_pole
import multiprocessing
import numpy as np
import tuning
import util


def double_pole_investigate(
        seed=None,
        dmg1=False,
        dmg2=False,
        te_mc=40.0,
        te_mc_mu=5.0,
        cut_discount=None,
        **kwargs):
    #kwargs = {}
    explorer = double_pole.solve_double_pole(
        do_explore=False,
        seed=seed,
        **kwargs)
    if True:
        nodes = [explorer.progenitor]
        while True:
            for node in nodes:
                node.explored_count = 0
            pivot = explorer.homomorphic_optimize(nodes[-1])
            new_nodes = []
            nn = pivot
            while nn is not nodes[-1]:
                new_nodes.append(nn)
                nn = nn.parent
            nodes.extend(new_nodes)
            if pivot.is_winner:
                return (explorer.step, pivot.get_size())
            if len(nodes) > 1:
                if cut_discount is None:
                    cut = explorer.random.randint(1, len(nodes) - 1)
                else:
                    cut = 1+ util.discounted_random(len(nodes) - 1, cut_discount, explorer.random)
                nodes = nodes[:cut]
            
        
    if False:
        while True:
            counter = 0
            while True:
                counter += 1
                explorer.progenitor.explored_count = 0
                pivot = explorer.homomorphic_optimize(explorer.progenitor, dmg=dmg1)
                if pivot.is_winner:
                    print (counter)
                    steps = explorer.step
                    #print("Steps = %d" % steps)
                    return (steps, pivot.get_size())
                
                #if pivot is not explorer.progenitor:
                #    break
            # Never get here 
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


def double_pole_investigate_spread(kwargs, var, values, f=None, prefix=None):
    kw2 = dict(kwargs)
    kw2["func"] = double_pole_investigate
    kw2["func_name"] = "Pole"
    kw2["pool"] = pool
    kw2["n_runs"] = n_runs
    return tuning.func_spread(kw2, var, values, f, prefix)
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
    #kwargs["fitness_threshold"] = 13.0
    #kwargs["pivotize_threshold"] = 40
    kwargs["pivot_mutation_weights"] = [1,1,1]
    kwargs["mc_func"]= constant_mc_func(3)
    f = open("dp_investigate.txt", "w")
    r = [40]
    double_pole_investigate_spread(kwargs, "pivotize_threshold", r, f=f)
    #double_pole_investigate(**kwargs)

def run4():    
    kwargs = {}
    kwargs["seed_base"] = 1000
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
    #kwargs["fitness_threshold"] = 13.0
    kwargs["pivotize_threshold"] = 40
    kwargs["pivot_mutation_weights"] = [1,1,1]
    kwargs["mc_func"]= constant_mc_func(3)
    f = open("dp_investigate.txt", "w")
    r = [1.01, 1.03, 1.1]
    double_pole_investigate_spread(kwargs, "cut_discount", r, f=f)
    #double_pole_investigate(**kwargs)


pool = None
n_runs =10
if __name__ == '__main__':
    multiprocessing .freeze_support()
    pool = multiprocessing.Pool(2)
    run4()
