
import copy
import numbers
import sys
import topology
from topology import ValidationError
import util
from variable import Variable


class TopologyError(Exception):
    '''Topology error. Might indicate that there is a loop in the network.'''
    pass
class Connection:
    '''Represents a connection between two neurons.
    It is considered to belong to {to_id} neuron.
    The direction of data flow depends on {reserve} field.
    If reserve is False, then the information flows from {from_id} to {to_id}.
    '''
    
    def __init__(self):
        self.to_id = None
        self.from_id = None
        self.reverse = False
        self.weight = Variable()
    
    def get_variables(self):
        return [self.weight]
    
    def __str__(self):
        direction = "->"
        if self.reverse:
            direction = "<-"
        return "%d%s%d:%.2f" % (self.from_id, direction, self.to_id, self.weight.value)

class Neuron:
    def __init__(self):
        self.id = None
        self.network = None
        self.x = 0.0
        self.connections = [] # outgoing connections
        self.incoming_connections = []
    
    def get_inputs(self):
        outgoing = [c for c in self.connections if c.reverse]
        incoming = [c for c in self.incoming_connections if not c.reverse]
        return dict(
            [(c.from_id, c) for c in incoming] +
            [(c.to_id, c) for c in outgoing]
            )
    
    
    def get_outputs(self):
        outgoing = [c for c in self.connections if not c.reverse]
        incoming = [c for c in self.incoming_connections if c.reverse]
        return dict(
            [(c.from_id, c) for c in incoming] +
            [(c.to_id, c) for c in outgoing]
            )
            
    def get_variables(self):
        return [v for v in c.get_variables() for c in self.connections]
    
    def precompute(self):
        self.inputs = self.get_inputs()
        self.outputs = self.get_outputs()
    
    def compute(self):
        m = self.inputs
        self.x = 0
        for id,c in m.items():
            self.x += self.network.neurons[id].x * c.weight.value
        self.x = util.sigmoid(self.x)
        
        
        

    

class ArtificialNeuralNetwork:
    def __init__(self):
        self.id_counter = 1
        self.neurons = {}
        self.inputs = []
        self.outputs = []
        
    def get_next_id(self):
        result = self.id_counter
        self.id_counter += 1
        return result
    
    def create_neuron(self):
        id = self.get_next_id()
        neuron = Neuron()
        neuron.id = id
        neuron.network = self
        self.neurons[id] = neuron 
        return neuron
    
    def precompute(self):
        [n.precompute() for n in self.neurons.values()]
        order = self.topological_sort()
        input_ids = set([input.id for input in self.inputs])
        self.order = [i for i in order if i not in input_ids]
        
        
    def topological_sort(self):
        outputs = dict([
            (neuron.id, set(neuron.outputs.keys()))
            for neuron in self.neurons.values()
            ])
        empty = [id for id,d in outputs.items() if len(d) == 0] 
        order = []
        while len(empty) > 0:
            i = empty.pop()
            order.append(i)
            for j in self.neurons[i].inputs.keys():
                outputs[j].discard(i)
                if len(outputs[j]) == 0:
                    empty.append(j)
        if len(order) != len([neuron for neuron in self.neurons if neuron is not None]):
            raise TopologyError("Topological sort failed")
        return order[::-1]
    
    def compute(self, input):
        assert(len(input) == len(self.inputs))
        assert(all([isinstance(x, numbers.Number) for x in input]))
        for i in range(len(input)):
            self.inputs[i].x = input[i]
        # Assume precompute() has been invoked
        for id in self.order:
            self.neurons[id].compute()
        return [output.x for output in self.outputs]
    
    def validate(self):
        try:
            self.precompute()
        except TopologyError as e:
            raise ValidationError(e)
        for id,neuron in self.neurons.items():
            assert(id == neuron.id)
            if len(neuron.inputs) == 0:
                if neuron not in self.inputs:
                    pass
                    #raise ValidationError("Neuron %d has no inputs." % neuron.id)
            if len(neuron.outputs) == 0:
                if neuron not in self.inputs and neuron not in self.outputs:
                    pass
                    #raise ValidationError("Neuron %d has no outputs." % neuronlid)
            connected_ids = [c.to_id for c in neuron.connections]
            connected_ids += [c.from_id for c in neuron.incoming_connections]
            if len(connected_ids) != len(set(connected_ids)):
                # There are duplicates
                raise ValidationError("Duplicated connections for neuron %d" % neuron.id)
            for c in neuron.connections:
                if neuron.id != c.from_id:
                    raise RuntimeError("Invalid graph state: neuron %d connection %d >> %d" % (neuron.id, c.from_id, c.to_id))
                if c not in self.neurons[c.to_id].incoming_connections:
                    raise RuntimeError("Invalid graph state: connection %d >> %d" % (neuron.id, c.to_id))
            for c in neuron.incoming_connections:
                if neuron.id != c.to_id:
                    raise RuntimeError("Invalid graph state: neuron %d connection %d << %d" % (neuron.id, c.from_id, c.to_id))
                if c not in self.neurons[c.from_id].connections:
                    raise RuntimeError("Invalid graph state: connection %d << %d" % (neuron.id, c.to_id))
        for neuron in self.inputs:
            if len(neuron.inputs) > 0:
                raise ValidationError("Input neuron %d has an input connection" % neuron.id)
        for neuron in self.outputs:
            if len(neuron.outputs) > 0:
                # Not sure if this should be a valid configuration
                #raise ValidationError("Output neuron %d has output connections" % neuron.id))
                pass

    def connect(self, from_id, to_id, reverse, weight=None):
        assert(from_id in self.neurons.keys())
        assert(to_id in self.neurons.keys())
        assert(from_id != to_id)
        c = Connection()
        c.from_id = from_id
        c.to_id = to_id
        c.reverse = reverse
        if weight is None:
            c.weight.value = 1.0
        else:
            c.weight = copy.deepcopy(weight)
        
        self.neurons[from_id].connections.append(c)
        self.neurons[to_id].incoming_connections.append(c)
        return c

    def disconnect(self, connection):
        assert(connection in self.neurons[connection.from_id].connections)
        self.neurons[connection.from_id].connections.remove(connection)
        assert(connection not in self.neurons[connection.from_id].connections)
        assert(connection in self.neurons[connection.to_id].incoming_connections)
        self.neurons[connection.to_id].incoming_connections.remove(connection)
        assert(connection not in self.neurons[connection.to_id].incoming_connections)
        
    def split(self, connection):
        self.disconnect(connection)
        neuron = self.create_neuron()
        c1 = self.connect(connection.from_id, neuron.id, connection.reverse, weight=connection.weight)
        c2 = self.connect(neuron.id, connection.to_id, connection.reverse)
        return (neuron, c1, c2)

    def delete(self, neuron):
        assert(neuron in self.neurons.values())
        for c in neuron.connections + neuron.incoming_connections:        
            self.disconnect(c)
        del self.neurons[neuron.id]

    def get_size(self):
        return len(self.neurons) - len(self.inputs) - len(self.outputs)
        
    def print(self, f, prefix=""):
        print(self.__str__(prefix), file=f)
    
    def __str__(self, prefix=""):
        result = ""
        #input_ids = [neuron.id for neuron in self.inputs]
        for id in self.order:
            m = self.neurons[id].inputs
            ss = ["%d=%.2f" % (id, c.weight.value) for id,c in m.items()]
            result += prefix + "%d: %s" % (id, ",".join(ss))
            result += "\n"
        return result
            
            
class ANNTopology(topology.Topology):
    def __init__(self):
        super().__init__()
        self.network = None
    
    def numerical_optimize(self):
        pass 
    
    def validate(self):
        self.network.validate()

    def get_size(self):
        return self.network.get_size()
        
    def print(self, f=sys.stdout, prefix=""):
        try:
            parent_id = self.parent.id
            parent_id = str(parent_id)
        except AttributeError:
            parent_id = "None"
        attrs = ""
        if self.is_winner:
            attrs += "WINNER "
        if self.is_pivot:
            attrs += "pivot "
        if self.is_progenitor:
            attrs += "progenitor "
        print(prefix + "Org %d, parent %s %s EC=%d" % (self.id, parent_id, attrs.strip(), self.explored_count), file=f)
        prefix += "  "
        print(prefix + "Size=%d, Children=%d, PivotHits=%d, ProgenitorHits=%d" % (self.get_size(), len(self.children), self.pivot_hits, self.progenitor_hits), file=f)
        print(prefix + "Network:", file=f)
        self.network.print(f, prefix=prefix+"  ")
        print(prefix + "Mutations:", file=f)
        mprefix = prefix + "  "
        for m in self.mutations:
            print(mprefix + m, file=f)

class BooleanFunctionTopology(ANNTopology):
    '''Represents a problem of a boolean function, such as XOR.'''
    
    # list of (input, output) tuples.
    # input and output are vectors of possible inputs and expected outputs.
    
    def __init__(self):
        super().__init__()

    def compute(self, input):
        outputs = self.network.compute(input)
        assert(len(outputs) == 1) 
        output = outputs[0]
        assert(0 <= output <= 1)
        #return int(round(output))
        return output 
            
    
    def get_fitness(self, random=None):
        self.outputs = []
        fitness = 0.0
        for test_input, test_output in self.tests:
            output = self.compute(test_input)
            self.outputs.append(output)
            fitness +=abs(output - test_output)                 
        fitness = 4.0- fitness
        assert(0 <= fitness <= 4)
        fitness **= 2
        fitness -= self.NEURON_COST *len(self.network.neurons)
        if fitness >= self.fitness_threshold:
            self.is_winner = True 
        
        return fitness
    
    def print(self, f=sys.stdout, prefix=""):
        super().print(f, prefix=prefix)
        prefix += "  "
        if self.parent is not None:
            parent_fitness = self.parent.fitness
            fitness_increment = self.fitness - self.parent.fitness
            fitness_increment_str = "increment_p: %f" % fitness_increment
        else:
            fitness_increment_str = ""
        fitness_increment_original_str = "increment_o: %f" % (self.fitness - self.original_fitness)
        print(prefix + "Fitness: %.2f %s %s" % (self.fitness, fitness_increment_str, fitness_increment_original_str ), file=f)
        outputs_str = ["%.2f" % x for x in self.outputs]
        print(prefix + "Outputs: %s" % ", ".join(outputs_str), file=f)

