
import topology

class ANNMutation:
    def is_homomorphic(self):
        return False
    
    def select_random_connection(self, topo, random):
        network = topo.network
        connections = [c for n in network.neurons.values() for c in n.connections]
        if len(connections) == 0:
            raise topology.MutationError("No connections in this network")
        c = random.choice(connections)
        return c


class AbstractConnectMutation(ANNMutation):
    def get_all_connections(self, network):
        return [c for n in network.neurons.values() for c in n.connections+n.incoming_connections]

class ConnectMutation(AbstractConnectMutation):
    def __init__(self, mu):
        self.mu = mu

    def sorted_tuple(self, t):
        a, b = t
        if a < b:
            return t        
        else:
            return (b, a)
        
    def apply(self, topo, random):
        network = topo.network
        assert(len(network.neurons) > 1)
        unconnected_pairs = set()
        for nid, n in network.neurons.items():
            ids = [nid] + [id for c in n.connections + n.incoming_connections for id in [c.from_id, c.to_id]]
            unconnected_ids = network.neurons.keys() - set(ids)
            unconnected_pairs = unconnected_pairs .union(set([self.sorted_tuple((nid, uid)) for uid in unconnected_ids]))
        if len(unconnected_pairs) == 0:
            raise topology.MutationError("Graph is too dense to add another connection")
        #[id1, id2] = random.sample(network.neurons.keys(), 2)
        (id1, id2) = random.choice(list(unconnected_pairs))
        #reverse = bool(random.getrandbits(1))
        reverse = False
        weight = random.gauss(0, self.mu)
        c = network.connect(id1, id2, reverse)
        c.weight.value = weight
        topo.mutations.append("connect %s" % c)
        def undo_func():
            del topo.mutations[-1]
            network.disconnect(c)
        return undo_func

class DisconnectMutation(AbstractConnectMutation):
    def apply(self, topo, random):
        network = topo.network
        all_connections = self.get_all_connections(network)
        if len(all_connections) == 0:
            raise topology.MutationError("No connections to disconnect")
        c = random.choice(all_connections)
        network.disconnect(c)
        topo.mutations.append("disconnect %s" % c)

class SplitMutation(ANNMutation):
    def __init__(self, mu):
        self.mu = mu
        
    def apply(self, topo, random):
        network = topo.network
        c = self.select_random_connection(topo, random)
        (n, c1, c2) = network.split(c)
        c2.weight.value = random.gauss(0, self.mu)
        topo.mutations.append("split %s >> (%f,%f)" % (c, c1.weight.value, c2.weight.value))
        def undo_func():
            del topo.mutations[-1]
            network.delete(n)
            network.connect(c)
        return undo_func

        
class PerturbMutation(ANNMutation):
    '''Perturbs one of the connection weights.'''
    def __init__(self, mu):
        self.mu = mu

    def is_homomorphic(self):
        return True
    
    def apply(self, topo, random):
        c = self.select_random_connection(topo, random)
        delta = random.gauss(0, self.mu)
        original_weight = c.weight.value
        c.weight.value += delta
        topo.mutations.append("perturb %s was %.2f" % (c, original_weight))
        def undo_func():
            del topo.mutations[-1]
            c.weight.value = original_weight 
        return undo_func
        
        
