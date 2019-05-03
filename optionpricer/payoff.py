"""payoff Module
series of classes for different payoff types

in C++ would have an abstract base class, and inherited payoff classes
but due to Python's duck typing simply have many payoff classes, all with a
getpayoff method

get_payoff method could instead be __call__ method, but then any abject with __call__
would be able to be used like a payoff, which could cause unexpected errors.


Payoffs need to have a clone method
"""
import copy
import numpy as np
#import error

class CallPayOff:
    def __init__(self,strike):
        if not isinstance(strike,(float,int)):
            raise TypeError("CallPayOff object initialization: strike should be a float or an integer")
        self._strike = strike
        self._name   = "a call pay off with strike:"+str(self._strike)
        self._assets = 1 # number of assets this payoff relates to

    def get_payoff(self,spot):
        return np.maximum((spot-self._strike), 0.0)

    def get_strike(self):
        return self._strike

    def clone(self):
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__


class PutPayOff:
    def __init__(self,strike):
        if not isinstance(strike,(float,int)):
            raise TypeError("PutPayOff object initialization: strike should be a float or an integer")
        self._strike = strike
        self._name   = "a put pay off with strike:"+str(self._strike)
        self._assets = 1 # number of assets this payoff relates to

    def get_payoff(self,spot):
        return np.maximum((self._strike-spot), 0.0)

    def get_strike(self):
        return self._strike

    def __str__(self):
        return self._name

    __repr__ = __str__


class DigitalPayOff:
    def __init__(self,strike):
        if not isinstance(strike,(float,int)):
            raise TypeError("DisigitalPayOff object initialization: strike should be a float or an integer")
        self._strike = strike
        self._name   = "a digital pay off with strike:"+str(self._strike)
        self._assets = 1 # number of assets this payoff relates to

    def get_payoff(self,spot):
        if isinstance(spot, (float, int)):
            if spot>self._strike:
                return 1.0
            else:
                return 0.0
        elif isinstance(spot, np.ndarray):
            payoff = np.zeros_like(spot)
            payoff[np.where(spot>self._strike)] = 1.0
            return payoff
        else:
            raise TypeError("spot supplied to DigitalPayOff is of unsupported type")


    def get_strike(self):
        return self._strike

    def __str__(self):
        return self._name

    __repr__ = __str__


class DoubleDigitalPayOff:
    def __init__(self,strike_lo,strike_hi):
        condition = isinstance(strike_lo,(float,int)) and \
                    isinstance(strike_hi,(float,int))
        if not condition:
            raise TypeError("DoubleDigitalPayOff object initialization: strikes should be float or integer")
        self._strike_lo = strike_lo
        self._strike_hi = strike_hi
        self._name   = "a double digital pay off with strikes:"+str(self._strike_lo)+", "+str(self._strike_hi)
        self._assets = 1 # number of assets this payoff relates to

    def get_payoff(self,spot):
        if isinstance(spot, (float, int)):
            if spot>self._strike_lo and spot<self._strike_lo:
                return 1.0
            else:
                return 0.0
        elif isinstance(spot, np.ndarray):
            payoff = np.zeros_like(spot)
            payoff[np.where(spot>self._strike_lo)] = 1.0
            payoff[np.where(spot>self._strike_hi)] = 0.0
            return payoff
        else:
            raise TypeError("spot supplied to DoubleDigitalPayOff is of unsupported type")


    def get_strike(self,lo=True):
        if lo:
            return self._strike_lo
        else:
            return self._strike_hi

    def __str__(self):
        return self._name

    __repr__ = __str__


# class BadPayOff:
#     def __init__(self,strike):
#         if not isinstance(strike,(float,int)):
#             raise TypeError("CallPayOff object initialization: strike should be a float or an integer")
#         self._strike = strike
