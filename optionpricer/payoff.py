"""payoff Module
series of classes for different payoff types

in C++ would have an abstract base class, and inherited payoff classes
but due to Python's duck typing simply have many payoff classes, all with a
getpayoff method

get_payoff method could instead be __call__ method, but then any abject with __call__
would be able to be used like a payoff, which could cause unexpected errors.
"""

import numpy as np
#import error

class CallPayOff:
    def __init__(self,strike):
        if not isinstance(strike,(float,int)):
            raise TypeError("CallPayOff object initialization: strike should be a float or an integer")
        self._strike = strike

    def get_payoff(self,spot):
        return np.maximum((spot-self._strike), 0.0)
