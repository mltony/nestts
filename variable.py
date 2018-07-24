
import math

class Variable:
    value = 0.0
    def get_bounds(self):
        return (-math.inf, math.inf)
    
class PositiveVariable:
    def get_bounds(self):
        return (0, math.inf)
    
class IntegerVariable:
    #TBD
    pass
