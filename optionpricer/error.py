import sys

class Error(Exception):
    pass

class BadPayOffError(Error):
    """raised when a payoff doesn`t have a get_payoff() method"""
    pass
