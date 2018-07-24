import ann
import ann_mutations
import math
import sys
import topology
import util


class SinglePoleTopology(ann.ANNTopology):
    def __init__(self):
        super().__init__()

    def compute(self, input):
        outputs = self.network.compute(input)
        assert(len(outputs) == 2)
        return outputs 
            
    
    def get_fitness(self, random):
        max_steps = 100000
        fitness = self.go_cart(max_steps, random)
        hidden_neurons_count = len(self.network.neurons) - len(self.network.inputs) - len(self.network.outputs)
        fitness -= 1000 *hidden_neurons_count
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
        #outputs_str = ["%.2f" % x for x in self.outputs]
        #print(prefix + "Outputs: %s" % ", ".join(outputs_str), file=f)



    # Ported from NEAT-1.2.1 C++ code
    #     cart_and_pole() was take directly from the pole simulator written
    #     by Richard Sutton and Charles Anderson.
    def go_cart(self, max_steps, random):
        #float x,            /* cart position, meters */
        #     x_dot,            /* cart velocity */
        #     theta,            /* pole angle, radians */
        #     theta_dot;        /* pole angular velocity */
        steps=0
        random_start=True
        input = [0.0] * 5 #Input loading array
        twelve_degrees=0.2094384
        if random_start:
            self.x = random.random() * 4.8 - 2.4
            self.x_dot = random.random() * 2.0 - 1
            self.theta = random.random() * 0.4 - .2
            self.theta_dot = random.random() * 3.0 - 1.5
        else: 
            self.x = self.x_dot = self.theta = self.theta_dot = 0.0
       
        while steps < max_steps:
            steps += 1
            input[0]=1.0 #Bias
            input[1]=(self.x + 2.4) / 4.8;
            input[2]=(self.x_dot + .75) / 1.5
            input[3]=(self.theta + twelve_degrees) / .41
            input[4]=(self.theta_dot + 1.0) / 2.0
            outputs = self.compute(input)
            #/*-- decide which way to push via which output unit is greater --*/
            if outputs[0] > outputs[1]:
                self.y = 0
            else:
                self.y = 1
            self.cart_pole()
            if (abs(self.x) > 2.4) or (abs(self.theta) > twelve_degrees): 
                return steps             
        return steps
    
    #//     cart_and_pole() was take directly from the pole simulator written
    #//     by Richard Sutton and Charles Anderson.
    #//     This simulator uses normalized, continous inputs instead of 
    #//    discretizing the input space.
    #/*----------------------------------------------------------------------
    #   cart_pole:  Takes an action (0 or 1) and the current values of the
    # four state variables and updates their values by estimating the state
    # TAU seconds later.
    #----------------------------------------------------------------------*/
    def cart_pole(self):
        GRAVITY=9.8
        MASSCART=1.0
        MASSPOLE=0.1
        TOTAL_MASS=(MASSPOLE + MASSCART)
        LENGTH=0.5      #/* actually half the pole's length */
        POLEMASS_LENGTH=(MASSPOLE * LENGTH)
        FORCE_MAG=10.0
        TAU=0.02      #/* seconds between state updates */
        FOURTHIRDS=4/3
    
        if self.y > 0:
            force = FORCE_MAG
        else:
            force = -FORCE_MAG
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)
        temp = (force + POLEMASS_LENGTH * self.theta_dot * self.theta_dot * sintheta) / TOTAL_MASS
        thetaacc = (
            (GRAVITY * sintheta - costheta* temp)
            / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
            / TOTAL_MASS))
            )
        xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS
        #/*** Update the four state variables, using Euler's method. ***/
        self.x  += TAU * self.x_dot
        self.x_dot += TAU * xacc
        self.theta += TAU * self.theta_dot
        self.theta_dot += TAU * thetaacc

def solve_single_pole(
    seed=None,
    max_steps=10000,
    connect_mutation_weight=1.0,
    split_mutation_weight=1.0,
    perturb_mutation_weight=1.0,
    start_with_connected_progenitor=False,
    out_file_name=None,
    orgs_file_name=None,
    **kwargs
    ):
    class MySinglePoleTopology(SinglePoleTopology):
        fitness_threshold = 80000
    progenitor = MySinglePoleTopology()
    progenitor.network = ann.ArtificialNeuralNetwork()
    pro_network = progenitor.network
    for i in range(5):
        pro_network.inputs.append(pro_network.create_neuron())
    for i in range(2):
        pro_network.outputs.append(pro_network.create_neuron())
    if start_with_connected_progenitor:
        for input in pro_network.inputs:
            for output in pro_network.outputs:
                c = pro_network.connect(input.id, output.id.id, False)
                c.weight.value = 0.0
            
    
    class SinglePoleExplorer(topology.TopologyExplorer):
        def __init__(self):
            super().__init__()
            self.mutations = util.Distribution()
            self.mutations.add(connect_mutation_weight, ann_mutations.ConnectMutation(6.0))
            #self.mutations.add(1, ann_mutations.DisconnectMutation())
            self.mutations.add(split_mutation_weight, ann_mutations.SplitMutation(6.0))
            self.mutations.add(perturb_mutation_weight, ann_mutations.PerturbMutation(3.0))
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
    
    explorer = SinglePoleExplorer()
    explorer.add_progenitor(progenitor)
    explorer.explore()
    if not explorer.get_best_topo().is_winner:
        # failed to find a solution
        return (math.inf, math.nan)
    return (explorer.step, explorer.get_best_topo().get_size())

if __name__ == "__main__":
    solve_single_pole(seed=0, max_steps=10000)
