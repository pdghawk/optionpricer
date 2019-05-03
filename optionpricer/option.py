""" Module for Option classes

all option classes should have methods:

- get_option_payoff()
- clone()

"""

from optionpricer import error
import copy

class VanillaOption:
    """ Option with a single expiry time/maturity

    VanillaOption should be a initialized with a optionpricer.payoff object
    and an expiry time.

    """
    def __init__(self,payoff,expiry):
        if not isinstance(expiry,(float,int)):
            raise TypeError("VanillaOption object initialization: expiry should be time to expiry (in years) as a float or int")
        self._expiry = expiry
        self._payoff = payoff.clone() # should be a clone
        self._name   = "a vanilla option with " + str(self._payoff) + ", and expiry: " + str(self._expiry)
        print(self._name)

    def get_option_payoff(self,spot):
        """ returns the payoff of the option for a given spot
        Args:
            - spot: the spot to get the payoff for
        Returns:
            - payoff: The payoff for the given spot
        """
        try:
            return self._payoff.get_payoff(spot)
        except AttributeError as e:
            raise error.BadPayOffError(e)

    def get_expiry(self):
        """ return the option's expiry time
        Returns:
            - expiry: the expiry of the option
        """
        return self._expiry

    def clone(self):
        """ get a clone (deep copy) of this object
        Returns:
            - a deep copy of this VanillaOption
        """
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__=__str__
