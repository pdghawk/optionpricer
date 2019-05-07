""" Module for Option classes

all option classes should have methods:

- get_option_payoff()
- clone()

"""

# These should inherit from single asset option maybe

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
        self._name   = "a vanilla option with " + str(self._payoff) + ", and expiry: " + "{:.1f}".format(self._expiry*252.0) + "days"
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

    @property
    def the_expiry(self):
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


# class ExoticOption:
#     """ Option with path dependent payoff
#
#     VanillaOption should be a initialized with a optionpricer.payoff object
#     and an expiry time.
#
#     """
#     def __init__(self,payoff,key_times,payoff_times):
#         self.key_times    = key_times
#         self.payoff_times = payoff_times
#         self._payoff = payoff.clone() # should be a clone
#         self._name   = "an exotic Asian option with " + str(self._payoff) + ", and expiry: " + "{:.1f}".format(self._key_times[-1]*252.0) + "days"
#         print(self._name)
#
#     def get_option_payoff(self,spots):
#         """ returns the payoff of the option for a given spot
#         Args:
#             - spots: the spots at the key times for this option
#         Returns:
#             - payoff: The payoff for the given spot
#         """
#         # 1: check if payoff_times is single time or many times
#         # 2: if single: get payoff based on function (ie avg) applied to sopt at each key_time
#         # 3: if many times, then for each time
#         # N.B if was for a multiasset exotic, this would fail bc spot should be len>1 in all cases then
#             try:
#                 return self._payoff.get_payoff(avg_spot)
#             except AttributeError as e:
#                 raise error.BadPayOffError(e)
#         else:
#
#
#     def get_expiry(self):
#         """ return the option's expiry time
#         Returns:
#             - expiry: the expiry of the option
#         """
#         return self._key_times[-1]
#
#     @property
#     def the_expiry(self):
#         """ return the option's expiry time
#         Returns:
#             - expiry: the expiry of the option
#         """
#         return self._key_times[-1]
#
#
#     def clone(self):
#         """ get a clone (deep copy) of this object
#         Returns:
#             - a deep copy of this VanillaOption
#         """
#         return copy.deepcopy(self)
#
#     def __str__(self):
#         return self._name
#
#     __repr__=__str__
