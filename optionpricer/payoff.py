""" Module for payoff classes

series of classes for different payoff types

in C++ would have an abstract base class, and inherited payoff classes
but due to Python's duck typing simply have many payoff classes, all with a
getpayoff method

get_payoff method could instead be __call__ method, but then any abject with __call__
would be able to be used like a payoff, which could cause unexpected errors.


All Payoffs should have (at minimum) the following methods:

get_payoff()
get_strike()
clone()
__str__()

"""
import copy
import numpy as np
#import error

class CallPayOff:
    """ Payoff for a call option """
    def __init__(self,strike):
        if not isinstance(strike,(float,int)):
            raise TypeError("CallPayOff object initialization: strike should be a float or an integer")
        self._strike = strike
        self._name   = "a call pay off with strike:"+str(self._strike)
        self._assets = 1 # number of assets this payoff relates to

    def get_payoff(self,spot):
        """ returns the payoff for a given spot
        Args:
            - spot: the spot to get the payoff for
        Returns:
            - payoff: The payoff for the given spot
        """
        return np.maximum((spot-self._strike), 0.0)

    def get_strike(self):
        """ return the strike of this payoff object
        Returns:
            - strike: the strike of the payoff
        """
        return self._strike

    @property
    def n_assets(self):
        """ return the numnber of assets (underlyings) of this payoff object
        Returns:
            - number of underlyings
        """
        return self._assets

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this object
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__


class PutPayOff:
    """ Payoff for a put option """
    def __init__(self,strike):
        if not isinstance(strike,(float,int)):
            raise TypeError("PutPayOff object initialization: strike should be a float or an integer")
        self._strike = strike
        self._name   = "a put pay off with strike:"+str(self._strike)
        self._assets = 1 # number of assets this payoff relates to

    def get_payoff(self,spot):
        """ returns the payoff for a given spot
        Args:
            - spot: the spot to get the payoff for
        Returns:
            - payoff: The payoff for the given spot
        """
        return np.maximum((self._strike-spot), 0.0)

    def get_strike(self):
        """ return the strike of this payoff object
        Returns:
            - strike: the strike of the payoff
        """
        return self._strike

    @property
    def n_assets(self):
        """ return the numnber of assets (underlyings) of this payoff object
        Returns:
            - number of underlyings
        """
        return self._assets

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this object
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__


class DigitalPayOff:
    """ Payoff for a digital option """
    def __init__(self,strike):
        if not isinstance(strike,(float,int)):
            raise TypeError("DisigitalPayOff object initialization: strike should be a float or an integer")
        self._strike = strike
        self._name   = "a digital pay off with strike:"+str(self._strike)
        self._assets = 1 # number of assets this payoff relates to

    def get_payoff(self,spot):
        """ returns the payoff for a given spot
        Args:
            - spot: the spot to get the payoff for
        Returns:
            - payoff: The payoff for the given spot
        """
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
        """ return the strike of this payoff object
        Returns:
            - strike: the strike of the payoff
        """
        return self._strike

    @property
    def n_assets(self):
        """ return the numnber of assets (underlyings) of this payoff object
        Returns:
            - number of underlyings
        """
        return self._assets

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this object
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__


class DoubleDigitalPayOff:
    """ Payoff for a double digital option """
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
        """ returns the payoff for a given spot
        Args:
            - spot: the spot to get the payoff for
        Returns:
            - payoff: The payoff for the given spot
        """
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
        """ return the strike of this payoff object
        Keyword Args:
            - lo: if True, return the lower strike, else return the higher strike

        Returns:
            - strike: the strike of the payoff according to the selection of
                      the lo keyword argument
        """
        if lo:
            return self._strike_lo
        else:
            return self._strike_hi

    @property
    def n_assets(self):
        """ return the numnber of assets (underlyings) of this payoff object
        Returns:
            - number of underlyings
        """
        return self._assets

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this object
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__


# class BadPayOff:
#     def __init__(self,strike):
#         if not isinstance(strike,(float,int)):
#             raise TypeError("CallPayOff object initialization: strike should be a float or an integer")
#         self._strike = strike

class ExchangePayOff:
    """ Payoff for an exchange option """
    def __init__(self):
        self._name   = "an exchange option"
        self._assets = 2 # number of assets this payoff relates to

    def get_payoff(self,spot):
        """ returns the payoff for a given spot
        Args:
            - spot: the spot(s) to get the payoff for, of size [cases, 2]
        Returns:
            - payoff: The payoff for the given spot
        """
        if isinstance(spot, np.ndarray):
            if (spot.ndim==2):
                if(np.size(spot,1)==self._assets):
                    payoff = np.fmax(spot[:,0] - spot[:,1],np.zeros_like(spot[:,0]))
                    return payoff
                else:
                    raise TypeError("spot supplied to ExchangePayOff is of wrong size")
            else:
                raise TypeError("spot supplied to ExchangePayOff is of wrong size")
        else:
            raise TypeError("spot supplied to ExchangePayOff is of unsupported type")


    def get_strike(self):
        """ return the strike of this payoff object
        Returns:
            - strike: None, as no strike for exchange object
        """
        return None

    @property
    def n_assets(self):
        """ return the numnber of assets (underlyings) of this payoff object
        Returns:
            - number of underlyings
        """
        return self._assets

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this object
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__


class SpreadPayOff:
    """ Payoff for a spread option """
    def __init__(self,strike):
        self._name   = "a spread option"
        self._strike  = strike
        self._assets = 2 # number of assets this payoff relates to

    def get_payoff(self,spot):
        """ returns the payoff for a given spot
        Args:
            - spot: the spot(s) to get the payoff for, of size [cases, 2]
        Returns:
            - payoff: The payoff for the given spot
        """
        if isinstance(spot, np.ndarray):
            if (spot.ndim==2):
                if(np.size(spot,1)==self._assets):
                    payoff = np.fmax(spot[:,0] - spot[:,1] - self._strike,np.zeros_like(spot[:,0]))
                    return payoff
                else:
                    raise TypeError("spot supplied to ExchangePayOff is of wrong size")
            else:
                raise TypeError("spot supplied to ExchangePayOff is of wrong size")
        else:
            raise TypeError("spot supplied to ExchangePayOff is of unsupported type")


    def get_strike(self):
        """ return the strike of this payoff object
        Returns:
            - strike: None, as no strike for exchange object
        """
        return self._strike

    @property
    def n_assets(self):
        """ return the numnber of assets (underlyings) of this payoff object
        Returns:
            - number of underlyings
        """
        return self._assets

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this object
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__


class DoubleCallPayOff:
    """ Payoff for an exchange option """
    def __init__(self,strikes):
        self._name   = "a double call option"
        self.strikes = strikes
        self._assets = 2 # number of assets this payoff relates to

    def get_payoff(self,spot):
        """ returns the payoff for a given spot
        Args:
            - spot: the spot(s) to get the payoff for, of size [cases, 2]
        Returns:
            - payoff: The payoff for the given spot
        """
        if isinstance(spot, np.ndarray):
            if (spot.ndim==2):
                if(np.size(spot,1)==self._assets):
                    payoff = np.fmax(spot[:,0] - self.strikes[0],np.zeros_like(spot[:,0]))
                    payoff += np.fmax(spot[:,1] - self.strikes[1],np.zeros_like(spot[:,0]))
                    return payoff
                else:
                    raise TypeError("spot supplied to ExchangePayOff is of wrong size")
            else:
                raise TypeError("spot supplied to ExchangePayOff is of wrong size")
        else:
            raise TypeError("spot supplied to ExchangePayOff is of unsupported type")


    def get_strike(self):
        """ return the strike of this payoff object
        Returns:
            - strikes:
        """
        return self.strikes

    @property
    def n_assets(self):
        """ return the numnber of assets (underlyings) of this payoff object
        Returns:
            - number of underlyings
        """
        return self._assets

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this object
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__ = __str__
