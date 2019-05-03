"""option classes
"""

from optionpricer import error
import copy

class VanillaOption:
    def __init__(self,payoff,expiry):
        if not isinstance(expiry,(float,int)):
            raise TypeError("VanillaOption object initialization: expiry should be time to expiry (in years) as a float or int")
        self._expiry = expiry
        self._payoff = payoff.clone() # should be a clone
        self._name   = "a vanilla option with " + str(self._payoff) + ", and expiry: " + str(self._expiry)
        print(self._name)

    def get_option_payoff(self,spot):
        try:
            return self._payoff.get_payoff(spot)
        except AttributeError as e:
            raise error.BadPayOffError(e)

    def get_expiry(self):
        return self._expiry

    def clone(self):
        return copy.deepcopy(self)

    def __str__(self):
        return self._name

    __repr__=__str__
