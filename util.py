
import math
import operator
import random
from sortedcontainers import SortedDict
import statistics

def is_finite(x):
    return not math.isnan(x) and not math.isinf(x)

def avg(a):
    if len(a) == 0:
        return math.nan
    return statistics.mean(a)
    #return sum(a) / len(a)

def median(a):
    if len(a) == 0:
        return math.nan
    return statistics.median(a)
    a = sorted(a)
    l = len(a)
    if l % 2 == 0:        
        return avg(a[l/2-1:l/2+1])
    else:
        return a[l/2]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def discounted_random(n, discount, random):
    assert(n > 0)
    # total_weight = discount**0 + discount**1 + discount**2 + ... + discount**(n-1)
    total_weight = (1 - discount ** n) / (1 - discount)
    r = random.random() * total_weight
    c = 1.0
    i = 0
    while True:
        if i >= n:
            print("Damn sdfluhlku")
        assert(i < n)
        if r < c:
            return i 
        r -= c
        i += 1
        c *= discount
    

def weighted_random(weights, random):
    assert(all([w >= 0 for w in weights]))
    total_weight = sum(weights)
    r = random.random() * total_weight
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i    
    return RuntimeError()

class Distribution:
    def __init__(self):
        self.values = []
        self.totalWeight = 0.0 
    
    def update(self):
        self.totalWeight = sum([x[0] for x in self.values])

    def add(self, weight, value):
        assert(weight >= 0)
        self.values.append((weight, value))
        self.update()
        
    def draw(self, random=random.Random):
        assert(len(self.values) > 0)
        assert(self.totalWeight > 0)
        r = random.random() * self.totalWeight
        c = 0.0
        for w, v in self.values:
            c += w
            if c >= r:
                return v  
        raise RuntimeError("Inconsistent state in Distribution")
    
class IntegerStatistics:
    def __init__(self):
        self.a = []
        self.hist = SortedDict()
        
    def record(self, x):
        self.a.append(x)
        try:
            self.hist[x] += 1
        except KeyError:
            self.hist[x] = 1
        
    def hist_str(self, reverse=False):
        items =self.hist.items()
        if reverse:
            items = list(items)[::-1] 
        return ", ".join(["%d:%d" % (k, v) for k,v in items if v > 0])
    
class PopulationOld(SortedDict):
    def __init__(self):
        super().__init__(operator.neg)
        self.disabled_topologies = []
        
    def save_topology(self, topology, fitness=None):
        if topology.disabled:
            self.disabled_topologies.append(topology)
            return
        if fitness is None:
            fitness = topology.fitness
        if fitness in self:
            self[fitness].append(topology)
        else:
            self[fitness] = [topology]
            
    def remove_topology(self, topology, fitness=None):
        if fitness is None:
            fitness = topology.fitness
        assert(topology in self[fitness])
        self[fitness].remove(topology)
        if len(self[fitness]) == 0:
            del self[fitness]

    def update_topology(self, topology, old_fitness):
        self.remove_topology(topology, old_fitness)
        self.save_topology(topology)
        
            
    def topologies(self, include_disabled=False):
        for tt in self.values():
            for t in tt:
                yield t
        if include_disabled:
            for t in self.disabled_topologies:
                yield t
                
    def count_topologies(self):
        return sum([len(v) for v in self.values()])
    
    def is_empty(self):
        return len(self) == 0
    
    def validate(self):
        for fitness, topos in self.items():
            for topo in topos:
                assert(fitness == topo.fitness)

    def best_topo(self):
        for topo in self.topologies():
            return topo

    def best_fitness(self):
        for topo in self.topologies():
            return topo.fitness
        raise RuntimeError() 
        
class Population:
    def __init__(self):
        self.best = None
        
    def save_topology(self, topology, fitness=None):
        if topology.is_disabled:
            return
        if fitness is None:
            fitness = topology.fitness
        if (self.best is None) or (fitness > self.best.fitness):
            self.best = topology

    def update_topology(self, topology, old_fitness):
        if (self.best is None) or (topology.fitness > self.best.fitness):
            self.best = topology
            
    def is_empty(self):
        return (self.best is None)
    
    def clear(self):
        self.best = None
    
    def validate(self):
        pass
    
    def best_topo(self):
        return self.best

    def best_fitness(self):
        return self.best_topo().fitness
    
