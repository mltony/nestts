# NESTTS
NESTTS stands for NeuroEvolution through Stochastic Topology Tree Search.

NESTTS aims to be a Neuroevolution framework. Although it doesn't perform a genetic or evolutionary algorithm per se, in the sense that it doesn't have a population of orgs and it doesn't select the fittest from every generation, it explores the space of topologically diverse artificial neural networks. NESTTS views the space of topologically diverse artificial neural networks as an infinite tree and it performs a special tree search similar to depth-first search on this tree while constructing parts of this tree at the same time.

Sample problems that NESTTS can solve are Xor problem, single and double pole balancing carts.

## Key concepts
NESTTS works with organisms - that is directly represented artificial neural networks with sigmoid activation function. For every org we evaluate its fitness on a problem we are trying to solve (such as Xor or Double-pole balancing cart). 

Mutation operators can be applied to orgs. Currently supported mutations are: 
* Perturb - perturbs the weight of a connection
* Connect - adds a new connection between two existing neurons
* Split - Split an existing connection by adding a new neuron in the middle
A new org can be obtained by applying one or more mutations to an existing org.

We also introduce two higher-level optimization functions.

### Homomorphic optimize
This function takes and org and tries to optimize the weights of the connections wihtout changing its topology.

```python
def homomorphic_optimize(org):
    while True:
        m = <<compute mutation count>>
        org2 = <<Apply m Perturb mutations to org>>
        if org2.fitness > org.fitness:
            org = org2
        org.explored_count += 1
        if org.explored_count >= threshold:
            return org
```

In other words, `homomorphic_optimize` tries to apply homomorphic mutations to `org` repeatedly. On every step, if mutations increase the fitness, then we keep them, otherwise we  ignore them. If we fail to increase fitness after `threshold` attempts, we consider this optimization to have converged and return the highest fitness org we found.

### Topological explore
This function takes an org as an input and tries to alter its topology to explore the topology space. It basically applies `m` mutations and returns the mutated org as result.

```python
def topological_explore(org):
    m = <<compute mutation count>>
    org2 = <<Apply m  mutations to org>>
    return org2
```

Here mutations are sampled from the set of all possible mutations, including Perturb, Connect and Split.

## Main loop
The basic idea of the algorithm looks like this:

```python
def evolve(progenitor_org):
    pivot_org = homomorphic_optimize(progenitor_org)
    while True:
        candidate_org = homomorphic_optimize(topological_explore(pivot_org))
        if candidate_org.fitness > winner_threshold:
            return candidate_org
```

In other words, we first homomorphically optimize the progenitor org and call it pivot. For Xor problem pivot can guess 3 out of 4 results correctly. Then we explore other topologies mutated from  pivot and optimize them homomorphically until we find a winner org.

Markovian double pole balancing problem can be solved even wihtout the second step. `homomorphic_optimize` alone can find a winning solution efficiently.

## Results
Xor: 2747 evaluations on average
Markovian double pole balancing: 1039 evaluations on average

## Old version of the algorithm (v7)
v7 is an old version of NESTTS algorithm. It doesn't perform as well as the current one, but I decided to include its description here for the record.

We perform search on a topology tree starting from the root progenitor.

Nodes of this tree are orgs, i.e. artificial neural networks with defined values for all connection weights. We can build the tree by applying mutations to any node in the tree. The resulting mutated org will be added to the tree as a child.

The basic idea of v7 algorithm can be expressed in pseudocode like this:

```python
def evolve_v7(root):
    while True:
        org = <<pick highest fitness org in the tree>>
        if org.is_winner:
            return org
        org2 = apply_mutations(org)
        org.children.append(org2)
```

This simplistic pseudocode wouldn't work, because it only looks at the org with the highest fitness in the tree. However, sometimes a mutation can lead to an org with lower fitness, but that org can eventually evolve into a winning solution. So sometimes we need to explore orgs that have lower fitness than the current champion in the tree. In order to achieve this, we introduce pivots.

### Pivots
Every org has a counter `explored_count` that indicates how many times this node has been visited so far. If the value of this counter exceeds a predefined threshold, then we convert this org into a pivot.

When a pivot is visited, the behavior is going to be different. With small probability `p` we will fall back to non-pivot behavior, that is apply some mutations to the pivot and append the outcome as another child of the pivot. 

With probability `1-p` however, we will explore one of the children (direct or grandchildren) of the pivot. We can assume that at this point the fitness of any children and grandchildren of pivot is less than the fitness of pivot, otherwise pivot would have not been visited. So by doing that we allowing some detrimental mutations to happen and we still give the detrimental offspring a chance to evolve into a higher fitness orgs. So we first select one of direct children of the pivot - whose subtree to explore, and then we visit the highest fitness org in that subtree.

This arrangement can be roughly compared to speciation idea in NEAT. 

Note that there can be multiple pivot in the tree at the same time. One pivot can be either direct or indirect child of another. When this happens, we allow two or more detrimental mutations to happen at the same time and we still give offspring a chance to recover.

Pivots can have a different distribution of mutations. For example, it would make sense to set normal mutation distribution to be 90% or 100% Perturb mutations, whereas pivot mutation distribution can be set to explore new topologies more aggressively, that is to have a substantial share of Connect and Split mutations.

