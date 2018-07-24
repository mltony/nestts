import ann
import ann_mutations
import math
from math import sin, cos
import sys
import topology
import util

# Converted from C++ NEAT-1.2.1 code
class CartPole:
    def __init__(self, randomize):
        #self.state = [0.0] * 6
        self.maxFitness = 100000
        self.MIN_INC = 0.001
        self.POLE_INC = 0.05
        self.MASS_INC = 0.01
        self.LENGTH_2 = 0.05
        self.MASSPOLE_2 = 0.01
        self.NUM_INPUTS=7
        self.status = [0.0] * 6
        self.TAU= 0.01 # seconds between state updates
        self.cartpos_sum= 0.0
        self.balanced_sum = 0
        self.GRAVITY= -9.8
        self.MUP = 0.000002
        self.MUC = 0.0005
        self.MASSCART= 1.0
        self.MASSPOLE_1= 0.1
        self.LENGTH_1= 0.5          #/* actually half the pole's length */
        self.FORCE_MAG= 10.0
        
    
    #Faustino Gomez wrote this physics code using the differential equations from 
    #Alexis Weiland's paper and added the Runge-Kutta himself.
    def evalNet(self, eval_func, debug=False):
        steps=0
        input = [0.0] * self.NUM_INPUTS
        self.init(0)
        while  steps < self.maxFitness:
            steps += 1
            input[0] = self.state[0] / 4.8
            input[1] = self.state[1] /2
            input[2] = self.state[2]  / 0.52
            input[3] = self.state[3] /2
            input[4] = self.state[4] / 0.52
            input[5] = self.state[5] /2
            input[6] = 1.0
            output = eval_func(input)
            if debug:
                print("\t".join(map(str, input + [output])))
            self.performAction(output,steps)
            if self.outsideBounds():
                break
        return steps
    
    def init(self, randomize):
        self.balanced_sum=0 #Always count # balanced
        self.last_hundred=False
        self.state = [0.0] * 6
        self.state[2] = 0.07 # one_degree; 
    
    def performAction(self, output, stepnum):
        dydx = [0.0] * 6
        state = self.state
        RK4=True #Set to Runge-Kutta 4th order integration method
        EULER_TAU= self.TAU/4
        if RK4:
            for i in range(2):
                dydx[0] = state[1]
                dydx[2] = state[3]
                dydx[4] = state[5]
                self.step(output,state,dydx)
                self.rk4(output,state,dydx,state)
        else:
            for i in range(8):
                self.step(output,state,dydx)
                state[0] += EULER_TAU * dydx[0]
                state[1] += EULER_TAU * dydx[1]
                state[2] += EULER_TAU * dydx[2]
                state[3] += EULER_TAU * dydx[3]
                state[4] += EULER_TAU * dydx[4]
                state[5] += EULER_TAU * dydx[5]
        #self.cartpos_sum+= abs(state[0])
        #self.cartv_sum+=abs(state[1])
        #polepos_sum+=fabs(state[2]);
        #polev_sum+=fabs(state[3]);
        #if stepnum<=1000
        #    self.jigglestep[stepnum-1]=abs(state[0])+abs(state[1])+abs(state[2])+abs(state[3])
        if not self.outsideBounds():
            self.balanced_sum += 1
    
    def step(self, action, st, derivs):
        force =  (action - 0.5) * self.FORCE_MAG * 2
        costheta_1 = cos(st[2])
        sintheta_1 = sin(st[2])
        gsintheta_1 = self.GRAVITY * sintheta_1
        costheta_2 = cos(st[4])
        sintheta_2 = sin(st[4])
        gsintheta_2 = self.GRAVITY * sintheta_2
        
        ml_1 = self.LENGTH_1 * self.MASSPOLE_1
        ml_2 = self.LENGTH_2 * self.MASSPOLE_2
        temp_1 = self.MUP * st[3] / ml_1
        temp_2 = self.MUP * st[5] / ml_2
        fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) + \
               (0.75 * self.MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1))
        fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) + \
               (0.75 * self.MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2))
        mi_1 = self.MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1))
        mi_2 = self.MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2))
        derivs[1] = (force + fi_1 + fi_2) \
                     / (mi_1 + mi_2 + self.MASSCART)
        derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1) \
                     / self.LENGTH_1
        derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2) \
                      / self.LENGTH_2
    
    def rk4(self, f, y, dydx, yout):
        dym = [0.0] * 6
        dyt = [0.0] * 6
        #yt = [0.0] * 6
        hh=self.TAU*0.5
        h6=self.TAU/6.0
        yt = [y[i]+hh*dydx[i] for i in range(6)]
        self.step(f,yt,dyt)
        dyt[0] = yt[1]
        dyt[2] = yt[3]
        dyt[4] = yt[5]
        yt = [y[i]+hh*dyt[i] for i in range(6)]
        self.step(f,yt,dym)
        dym[0] = yt[1]
        dym[2] = yt[3]
        dym[4] = yt[5]
        for i in range(6):
            yt[i]=y[i]+self.TAU*dym[i]
            dym[i] += dyt[i]
        self.step(f,yt,dyt)
        dyt[0] = yt[1]
        dyt[2] = yt[3]
        dyt[4] = yt[5]
        for i in range(6):
            yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i])
    
    def outsideBounds(self):
        thirty_six_degrees= 0.628329
        failureAngle = thirty_six_degrees
        state = self.state 
        return ( 
        (abs(state[0]) > 2.4)               or 
        (abs(state[2]) > failureAngle)      or
        (abs(state[4]) > failureAngle)
        )  
    
debug_counter = 0
class DoublePoleTopology(ann.ANNTopology):
    def __init__(self):
        super().__init__()
        self.cart = CartPole(1)

    def compute(self, input):
        outputs = self.network.compute(input)
        assert(len(outputs) == 1)
        return outputs[0] 
    
    def get_fitness(self, random):
        max_steps = 100000
        global debug_counter
        debug_counter+=1
        debug = False
        if debug_counter == 201:
            #debug = True
            pass
        fitness = self.cart.evalNet(lambda input:self.compute(input), debug)
        hidden_neurons_count = len(self.network.neurons) - len(self.network.inputs) - len(self.network.outputs)
        fitness -= 100 *hidden_neurons_count
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

def solve_double_pole(
    seed=None,
    max_steps=10000,
    connect_mutation_weight=1.0,
    split_mutation_weight=1.0,
    perturb_mutation_weight=1.0,
    start_with_connected_progenitor=False,
    do_explore=True,
    out_file_name=None,
    orgs_file_name=None,
    **kwargs
    ):
    class MyDoublePoleTopology(DoublePoleTopology):
        fitness_threshold = 80000
    progenitor = MyDoublePoleTopology()
    progenitor.network = ann.ArtificialNeuralNetwork()
    pro_network = progenitor.network
    for i in range(7):
        pro_network.inputs.append(pro_network.create_neuron())
    for i in range(1):
        pro_network.outputs.append(pro_network.create_neuron())
    if start_with_connected_progenitor:
        for input in pro_network.inputs:
            for output in pro_network.outputs:
                c = pro_network.connect(input.id, output.id, False)
                c.weight.value = 0.0
    class DoublePoleExplorer(topology.TopologyExplorer):
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
    
    explorer = DoublePoleExplorer()
    explorer.add_progenitor(progenitor)
    if not do_explore:
        return explorer
    explorer.explore()
    if not explorer.get_best_topo().is_winner:
        # failed to find a solution
        return (math.inf, math.nan)
    return (explorer.step, explorer.get_best_topo().get_size())

if __name__ == "__main__":
    solve_double_pole(
        seed=0, 
        max_steps = 20000
        ,pruning_discount_factor= 0.8
        ,dead_end_threshold= 400
        ,subprogenitor_discount_factor= 0.9
        ,enable_homomorphic_propagation= False
        ,neuron_cost= 0.05
        ,start_with_connected_progenitor= True
        ,perturb_mutation_weight=10)
        
        
        
