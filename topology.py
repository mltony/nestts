import copy
import itertools
import math
import operator
import random
from sortedcontainers import SortedDict
import sys
import time
import threading
import util


class MutationError(Exception):
    """Can be thrown by Mutation.apply() methods.
    However if this is thrown, org/topology must be unchanged.
    If you need to change org/topology, then you should rather throw a ValidationError in Topology.validate() method.
    """
    pass

class ValidationError(Exception):
    pass

class DeadEndError(Exception):
    pass 

class Topology:
    def __init__(self):
        self.timestamp = time.process_time() 
        self.id = 0
        self.parent = None
        self.children = []
        self.is_winner = False
        self.explored_count = 0
        self.pivot_hits = 0
        self.progenitor_hits = 0
        self.mutations = []
        self.fitness = -math.inf
        self.original_fitness = -math.inf 
        self.is_progenitor = False
        self.is_pivot = False
        self.population = None
        self.progenitor = None
        self.parent_progenitor = None
        self.is_homomorphic_to_parent = False
        self.is_disabled = False
    
    def numerical_optimize(self):
        raise NotImplementedError()
    
    def get_fitness(self, random=None):
        raise NotImplementedError()
    
    def validate(self): # might throw TopologyValidationError
        raise NotImplementedError()
    
    def clone(self):
        # No shallow copying here because of risc of cross-referencing in subclasses.
        # Subclasses should override this to copy extra fields.
        my_type = type(self)
        other = my_type()
        other.network = copy.deepcopy(self.network)
        #other.parent = self
        #self.children.append(other) 
        return other
    
class TopologyExplorer:
    def __init__(self):
        self.random = random.Random()
        self.out = sys.stdout
        self.orgs_file_name = "orgs.txt"
        self.mutations = None # Normal mutation distribution
        self.pivot_mutations_distribution = None
        self.domain = None
        self.orgs = {}
        self.dead_ends = []
        self.progenitor = None
        self.lock = threading.RLock()
        self.mutation_stat = util.IntegerStatistics()
        self.step = 0
        self.id_counter = 1
        self.leaders_count = 1
        #self.mutation_quota = 2
        self.mc_func = None
        self.mc_base = 2.0
        self.mc_a = 1
        self.mc_b = 1
        self.mc_max = 20
        self.subspecies_threshold = 10
        self.subprogenitor_width_factor = 1.0
        self.subprogenitor_discount_factor = 0.99
        self.pruning_discount_factor = 0.99
        self.pivotize_threshold = 10
        self.fitness_epsilon = 0.01
        self.enable_homomorphic_propagation = True
        self.enable_dmg = True # Enable feature "DisableMinorGain"
        self.dead_end_threshold = 500
        
    def print(self, s):
        print(s, file=self.out)
    
    def generate_id(self):
        with self.lock: 
            result = self.id_counter
            self.id_counter += 1
        return result

    def select_subspecies(self, topology):
        if len(topology.children) < self.subspecies_threshold:
            return topology
        if self.random.random() >= self.subspecies_ratio:
            return topology 
        subspecies = sorted(topology.children, key=lambda t:t.fitness)[::-1]
        assert(all([self.is_known_topology(t) for t in subspecies]))
        l = min(len(subspecies), self.subspecies_leaders_count)
        weights = [0.9 ** i for i in range(l)]
        index = util.weighted_random(weights, self.random)
        return subspecies[index]
        


    def select_topology_from_progenitor(self, progenitor):
        assert(progenitor.is_progenitor)
        progenitor.progenitor_hits += 1
        population = progenitor.population
        return population.best_topo()
        # Ignore from this point
        l = population.count_topologies()
        l = min(l, self.leaders_count)
        index = util.discounted_random(l, 0.9, self.random)
        for i, topo in enumerate(population.topologies()):
            if i == index:
                return topo 
    
    def select_topology(self):
        progenitor = self.progenitor
        while True:
            topo = self.select_topology_from_progenitor(progenitor)
            if topo.is_pivot:
                eligible_children = [c for c in topo.children if not c.population.is_empty()]
                # eligible children excludes disabled children
                threshold = 1.0 / (1 + len(eligible_children )) * self.subprogenitor_width_factor 
            if (not topo.is_pivot) or (self.random.random() < threshold):
                #topo.explored_count += 1
                # We will increase EC after we apply mutations and after ascertaining that mutations are valid.  
                return topo
            # Select one of the children of the pivot to be the next progenitor whose tree to explore
            topo.pivot_hits += 1
            if topo.pivot_hits > self.dead_end_threshold:
                if topo is not self.progenitor:
                    error = DeadEndError()
                    error.org = topo
                    raise error
            subprogenitors = sorted(eligible_children, key=lambda p:p.population.best_fitness(), reverse=True)
            index = util.discounted_random(len(subprogenitors), self.subprogenitor_discount_factor, self.random)
            progenitor = subprogenitors[index]

    def progenitorize_recursive(self, progenitor, topo, recursing_into_subprogenitor):
        assert(topo.progenitor != progenitor)
        
        # The condition next line checks that 
        # one of the parents is progenitorized after the child has been progenitorized.
        if topo.progenitor == topo:
            recursing_into_subprogenitor = True
            topo.parent_progenitor = progenitor
        
        if not recursing_into_subprogenitor :
            topo.progenitor = progenitor
        progenitor.population.save_topology(topo)
        for child in topo.children:
            self.progenitorize_recursive(progenitor, child, recursing_into_subprogenitor )

    def progenitorize(self, topo):
        assert(not topo.is_progenitor)
        #assert(not topo.is_pivot)
        topo.is_progenitor = True
        topo.population = util.Population()
        topo.parent_progenitor = topo.progenitor
        self.progenitorize_recursive(topo, topo, False)
        
    def pivotize(self, topo):
        assert(not topo.is_pivot)
        topo.is_pivot = True
        if self.is_TEP(topo):
            topo.children.clear()
            topo.explored_count = 0
            self.update_progenitor_populations()
        for child in topo.children:
            self.progenitorize(child)
        

            
    def mutations_count_old(self, topology):
        '''How many times we should apply a mutation operator to this topology.
        We start with a single mutation and gradually increase if no progress is made.
        '''
        c = topology.explored_count
        for i in itertools.count():
            m = i + 1
            quota = self.mutation_quota ** (m * 0.9)
            if c < quota:
                # Up to m mutations
                if m >= 10:
                    print("Damn")
                assert( m < 10) # Doesn't make any sense to have too many mutations
                weights = [0.9 ** (-i) for i in range(m)]
                i = util.weighted_random(weights, self.random)
                result = i + 1
                self.mutation_stat.record(result)
                return result
            else:
                c -= quota

    def mutations_count(self, topology):
        if self.mc_func is not None:
            return self.mc_func(topology, self.random)
        c = topology.explored_count
        result =self.mc_a + math.log(c + self.mc_b, self.mc_base)
        result = min(result, self.mc_max)
        result = int(result) 
        result = 1 + util.discounted_random(result, 0.99, self.random)
        self.mutation_stat.record(result)
        return result

    def handle_mutation_error(self, old_topology, topology, e, e_type):
        if False:
            print(e_type + ": " + str(e))
            for m in topology.mutations:
                print(m)
        raise e

    def is_TEP(self, org):
        '''TEP means Topology-exploring pivot.
        That is any pivot that is non-root.'''
        if org.is_pivot and (self.pivot_mutations_distribution is not None) and (org != self.progenitor):
            return  True
        return False

    def apply_mutations(self, topology, mc=None):
        #assert(self.is_known_topology(topology))
        debug_undo_func = False
        old_topology = topology
        topology = old_topology.clone()
        assert(old_topology.network.neurons is not topology.network.neurons)
        mutation_distribution = self.mutations
        if self.is_TEP(old_topology):
            mutation_distribution = self.pivot_mutations_distribution
        if mc is not None:
            m = mc
        else:
            m = self.mutations_count(old_topology)
        homomorphic = True
        for i in range(m):
            success = False
            for attempt in range(100):
                undo_func = None
                try:
                    mutation = mutation_distribution.draw(self.random)
                    if debug_undo_func:
                        topo_str = str(topology)
                    undo_func = mutation.apply(topology, self.random)
                    assert(undo_func is not None)
                    topology.validate()
                    homomorphic = homomorphic and mutation.is_homomorphic()
                    success = True
                    break # successful attempt
                except ValidationError as e:
                    assert(undo_func is not None)
                    undo_func()
                    if debug_undo_func:
                        assert(topo_str == str(topology))
                except MutationError as e:
                    # topology is unchanged
                    if debug_undo_func:
                        assert(topo_str == str(topology))
                    pass
            if not success:
                raise RuntimeError("Failed to mutate org")
        '''
        except ValidationError as e:
            self.handle_mutation_error(old_topology, topology, e, "VE")
        except MutationError as e:
            self.handle_mutation_error(old_topology, topology, e, "ME")
        '''
        return (topology,homomorphic)
    
    def apply_to_progenitors(self, topology, f):
        progenitor = topology.progenitor
        while progenitor is not None:
            f(progenitor)
            progenitor = progenitor.parent_progenitor
            
    def update_topology_fitness(self, topology, original_fitness):
        self.apply_to_progenitors(topology, lambda g:g.population.update_topology(topology, original_fitness))
        
    def homomorphic_gain_old(self, topo, topo_gain):  
        original_fitness = topo.fitness
        fitness = topo_gain.fitness
        topo.fitness = fitness 
        topo.network = topo_gain.network
        topo.mutations.extend(topo_gain.mutations)
        self.update_topology_fitness(topo, original_fitness)
        return
        while topo.is_homomorphic_to_parent:
            parent = topo.parent
            if parent.fitness >= fitness:
                break
            parent_original_fitness = parent.fitness
            parent.fitness = fitness
            parent.network = topo.network
            parent.mutations.extend(topo.mutations)
            self.update_topology_fitness(parent, parent_original_fitness )
            topo = parent
            
    def homomorphic_gain(self, topo, topo_gain):
        '''
        Finds the highest fitness org in the homomorphic chain of topo.
        Increases its fitness and assigns the network of topo_gain to it.
        '''
        #original_fitness = topo.fitness
        original_topo = topo
        fitness = topo_gain.fitness
        homomorphic_chain = [topo]
        while topo.is_homomorphic_to_parent:
            topo = topo.parent
            homomorphic_chain.append(topo)
        homomorphic_chain = [t for t in homomorphic_chain if t.fitness < fitness]
        if len(homomorphic_chain) == 0:
            print("Damn kjcvbuhdfbg")
        topo = homomorphic_chain[-1]
        original_fitness = topo.fitness 
        topo.fitness = fitness 
        topo.network = topo_gain.network
        topo.outputs = topo_gain.outputs
        #topo.mutations.extend([m for t in homomorphic_chain[1:] for m in t.mutations])
        #topo.mutations.extend(topo_gain.mutations)
        topo.mutations.append("Homomorphic propagate from mutation on %d" % original_topo.id)
        self.update_topology_fitness(topo, original_fitness)
        
    def disable_topology_if_minor_gain(self, topology):
        if not self.enable_dmg:
            return
        fitness = topology.fitness
        p = topology.parent
        while p is not None:
            if p.fitness > fitness:
                return
            if 0 <= fitness - p.fitness <= self.fitness_epsilon:
                topology.is_disabled = True
                return 
            p = p.parent

    def update_progenitor_populations(self):
        def clear_progenitor_population(org):
            if org.is_progenitor:
                org.population.clear()
        self.apply_to_orgs(clear_progenitor_population)
        self.apply_to_orgs(lambda org:self.save_org(org))

    def prune(self, org):
        #best = self.progenitor.population.best_topo()
        self.dead_ends.append(org)
        best = org
        old_size = self.count_subtree(self.progenitor)
        trace = self.trace_topology(best)
        if len(trace) < 2:
            print("Damn lszdfh")
        assert(len(trace) >= 2)
        #prune_point = self.random.randint(1, len(trace) - 1)
        prune_point = 1 + util.discounted_random(len(trace) - 1, self.pruning_discount_factor, self.random)
        prune_org = trace[prune_point]
        parent = prune_org.parent
        assert(parent is not None)
        assert(prune_org in parent.children)
        parent.children.remove(prune_org)
        new_size = self.count_subtree(self.progenitor)
        pruned_size = self.count_subtree(prune_org)
        assert(old_size == new_size + pruned_size)
        self.update_progenitor_populations()
        old_best_fitness = best.fitness
        new_best_fitness = self.progenitor.population.best_topo().fitness
        self.print("Dead end at org %d with fitness %.2f." % (best.id, best.fitness))
        self.print(
            "Pruning at org %d with fitness %.2f. Depth: %d out of %d. Pruned away %d out of %d orgs." %
            (prune_org.id, prune_org.fitness,
             prune_point, len(trace),
             pruned_size, old_size))
        self.print("After pruning best fitness dropped from %.2f down to %.2f. Delta=%f." % (old_best_fitness, new_best_fitness, (new_best_fitness-old_best_fitness))) 

    def count_subtree(self, org):
        return 1 + sum([self.count_subtree(c) for c in org.children]) 
        
    def apply_to_orgs(self, f, top_down=True, org=None):
        if org is None:
            org = self.progenitor
        if top_down:
            f(org) 
        for c in org.children:
            self.apply_to_orgs(f, top_down=top_down, org=c)
        if not top_down:
            f(org)

    def save_org(self, org):
        self.apply_to_progenitors(org, lambda g:g.population.save_topology(org))
    
    def explore_step(self):
        best_fitness = self.get_best_fitness()
        parent_topology = self.select_topology()
        (topology,homomorphic) = self.apply_mutations(parent_topology)
        topology.validate()
        parent_topology.explored_count += 1
        topology.numerical_optimize()
        fitness = topology.get_fitness(self.random)
        topology.fitness = fitness
        topology.original_fitness = fitness
        if homomorphic:
            # Disable homomorphic propagation?
            #if self.enable_homomorphic_propagation and (0 < (fitness - parent_topology.fitness) < self.fitness_epsilon):
            if (
                self.enable_homomorphic_propagation and 
                (0 < (fitness - parent_topology.fitness)) and
                (fitness - parent_topology.original_fitness) < self.fitness_epsilon):
                if fitness > best_fitness:
                    self.print("Org %d increased fit %f" % (parent_topology.id, fitness))
                self.homomorphic_gain(parent_topology, topology)
                return
            # If homomorphic, but fitness has droppedor increased by more than epsilon, still create a new offspring in the tree
        topology.id = self.generate_id()
        topology.is_homomorphic_to_parent = homomorphic
        topology.parent = parent_topology 
        topology.progenitor = parent_topology.progenitor
        parent_topology.children.append(topology)
        self.disable_topology_if_minor_gain(topology)         
        self.save_org(topology)
        self.orgs[topology.id] = topology
        if (fitness > best_fitness) and not topology.is_disabled:
            self.print("Org %d fit %f size %d" % (topology.id, fitness, topology.get_size()))
        if parent_topology.is_pivot:
            self.progenitorize(topology)
        else:
            if len(parent_topology.children) >= self.pivotize_threshold:
                #self.print("Pivotize org %d" % parent_topology.id)
                self.pivotize(parent_topology)

    def explore_step_with_validation(self):
        self.step += 1
        
        max_attempts = 1000
        for i in range(max_attempts):
            try:
                self.explore_step()
                return
            except ValidationError:
                pass
            except MutationError:
                pass
            except DeadEndError as e:
                self.prune(e.org)
        raise RuntimeError("Couldn't produce a valid mutation after %d attempts." % max_attempts)
        
    def stop_condition(self):
        return self.step >= 2500
    
    def explore(self):
        while not self.stop_condition():
            self.explore_step_with_validation()
        self.finalize()

    def homomorphic_optimize(self, org, dmg=False):
        assert(not org.is_pivot)
        max_ho_steps = 10000
        epsilon = 0.0
        if dmg:
            epsilon = self.fitness_epsilon
        for step in range(max_ho_steps):
            org.explored_count += 1
            if org.explored_count >= self.pivotize_threshold:
                return org
            (org2,homomorphic) = self.apply_mutations(org)
            org2.validate()
            self.step += 1
            fitness = org2.get_fitness(self.random)
            org2.fitness = fitness
            if org2.fitness > org.fitness + epsilon:  
                #print("Fitness %.2f" % org2.fitness)
                org2.parent = org
                org = org2
                if org.is_winner:
                    return org
        raise RuntimeError("homomorphic_optimize() failed to converge after %d steps" % max_ho_steps )
    
    def explore_topology(self, org, mc):
        org.is_pivot = True
        #print(org.is_pivot)
        #print(self.pivot_mutations_distribution is not None)
        #print(org != self.progenitor)
        assert(self.is_TEP(org))
        (org2,homomorphic) = self.apply_mutations(org, mc=mc)
        org2.fitness = org2.get_fitness(self.random)
        self.step += 1
        return org2
    
    def trace_topology(self, topo):
        trace = []
        while topo is not None:
            trace.append(topo)
            topo = topo.parent
        return trace[::-1]
        
        
    def time_report(self):
        block_size = 1000
        times = []
        for i in range(1 + block_size, max(self.orgs.keys()) + 1, block_size):
            times.append(self.orgs[i].timestamp - self.orgs[i - block_size].timestamp)
        self.print("Block times in seconds: %s" % str(times))
        

    def finalize(self):
        self.print("Best=%f, evals=%d." % (self.get_best_fitness(), self.step))
        best_topology = self.get_best_topo()
        trace_ids = [t.id for t in self.trace_topology(best_topology)]
        self.print("Trace best: %s" % str(trace_ids))
        topologies = list(self.orgs.values())
        indexed_topologies = dict([(t.id, t) for t in topologies]) 
        assert(None not in topologies)
        ec_stat = util.IntegerStatistics()
        with open(self.orgs_file_name , "w") as f:
            print("###Best:", file=f)
            for t in self.trace_topology(best_topology):
                t.print(f)
            print("###All:", file=f)
            for topology in sorted(topologies, key=lambda t:t.id):
                topology.print(f)
                ec_stat.record(topology.explored_count)
            print("###Dead ends:", file=f)
            for t in self.dead_ends:
                t.print(f)
        self.print("Mutations: %s" % self.mutation_stat.hist_str())
        self.print("Explored count: %s" % ec_stat.hist_str(reverse=True))
        top_explored = sorted(topologies, key=lambda t:t.explored_count, reverse=True)
        try:
            top_explored = top_explored[:5]
        except IndexError:
            pass
        self.print("Top explored: %s" % ", ".join(["%d:%d" % (t.id, t.explored_count) for t in top_explored]))
        self.time_report()
        
    def get_best_topo(self):
        return self.progenitor.population.best_topo()
    
    def get_best_fitness(self):
        return self.get_best_topo().fitness   
        
    def add_progenitor(self, progenitor):
        progenitor.validate()
        progenitor.id = self.generate_id()
        self.progenitor = progenitor
        fitness = progenitor.get_fitness(self.random)
        progenitor.fitness = fitness
        #progenitor.original_fitness = fitness
        progenitor.progenitor = progenitor
        progenitor.is_progenitor = True
        progenitor.population = util.Population()
        progenitor.population.save_topology(progenitor)
        self.orgs[progenitor.id] = progenitor
