import ann
import ann_mutations
import math
import os
import topology
import util

def solve_xor(**kwargs):
    tests = [
            ([1, i, j], int(i!=j))
            for i in range(2)
            for j in range(2)]
    return solve_binary_function(tests=tests,**kwargs)

def solve_or(**kwargs):
    tests = [
            ([1, i, j], int(i + j >= 1))
            for i in range(2)
            for j in range(2)]
    return solve_binary_function(tests=tests,**kwargs)


def solve_binary_function(
    tests,
    seed=None,
    max_steps=10000,
    connect_mutation_weight=1.0,
    split_mutation_weight=1.0,
    perturb_mutation_weight=1.0,
    pivot_mutation_weights=None, #Connect, Split, Perturb
    start_with_connected_progenitor=False,
    neuron_cost=0.1,
    fitness_threshold =13.0, 
    out_file_name=None,
    orgs_file_name=None,
    do_explore=True,
    **kwargs
    ):
    this_tests = tests
    this_fitness_threshold = fitness_threshold  
    class XorProblem(ann.BooleanFunctionTopology):
        fitness_threshold = this_fitness_threshold 
        NEURON_COST = neuron_cost 
        tests = this_tests   
    xor_progenitor = XorProblem()
    xor_progenitor.network = ann.ArtificialNeuralNetwork()
    pro_network = xor_progenitor.network
    for i in range(3):
        pro_network.inputs.append(pro_network.create_neuron())
    pro_network.outputs.append(pro_network.create_neuron())
    if start_with_connected_progenitor:
        for input in pro_network.inputs:
            c = pro_network.connect(input.id, pro_network.outputs[0].id, False)
            c.weight.value = 0.0
            
    
    class XorExplorer(topology.TopologyExplorer):
        def __init__(self):
            super().__init__()
            self.mutations = util.Distribution()
            self.mutations.add(connect_mutation_weight, ann_mutations.ConnectMutation(6.0))
            #self.mutations.add(1, ann_mutations.DisconnectMutation())
            self.mutations.add(split_mutation_weight, ann_mutations.SplitMutation(6.0))
            self.mutations.add(perturb_mutation_weight, ann_mutations.PerturbMutation(3.0))
            if pivot_mutation_weights is not None:
                self.pivot_mutations_distribution= util.Distribution()
                self.pivot_mutations_distribution.add(pivot_mutation_weights[0], ann_mutations.ConnectMutation(6.0))
                self.pivot_mutations_distribution.add(pivot_mutation_weights[1], ann_mutations.SplitMutation(6.0))
                self.pivot_mutations_distribution.add(pivot_mutation_weights[2], ann_mutations.PerturbMutation(3.0))
            if seed is not None:
                self.random.seed(seed)
            if out_file_name is not None:
                self.out = open(out_file_name, "w")
            if orgs_file_name is not None:
                self.orgs_file_name = orgs_file_name
            for k,v in kwargs.items():
                setattr(self, k, v)

        def stop_condition(self):
            return (
                (self.step >= max_steps) 
                or (self.get_best_topo().is_winner)
                )
    
    explorer = XorExplorer()
    explorer.add_progenitor(xor_progenitor)
    if not do_explore:
        return explorer
    explorer.explore()
    if not explorer.get_best_topo().is_winner:
        # failed to find a solution
        return (math.inf, math.nan)
    return (explorer.step, explorer.get_best_topo().get_size())

if __name__ == "__main__":
    solve_xor(
        seed=0, 
        max_steps=10000,
        enable_homomorphic_propagation=False
        )
    #dead_end_threshold=150, start_with_connected_progenitor=True,
    #solve_xor()
