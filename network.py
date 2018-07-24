
import topology
from variable import Variable

class Connection:
    to_id = None
    from_id = None
    reverse = False
    weight = Variable(0.0)
    def get_variables(self):
        return [self.weight]

class Node 
    id = None
    connections = [] # outgoing connections
    incoming_connections = []
    def get_outputs(self):
        outgoing = [c for c in self.connections if not c.reversed]
        incoming = [c for c in self.incoming_connections if c.reversed]
        return map(
            [(c.to_id, c) for c in incoming] +
            [(c.from_id, c) for c in outgoing)]
            )
            
    def get_variables(self):
        return [v for v in c.get_variables() for c in self.connections)]
                
    

class TopologyNetwork(topology.Topology):
    id_counter = 1
    def get_next_node_id(self):
        result = self.id_counter
        self.id_counter += 1
        return result
    
    def create_node(self):
        # Implementation specific
        return NotImplementedError())