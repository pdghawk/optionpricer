""" Module for analytic expressions of option prices
"""

from scipy.stats import norm
import numpy as np

def black_scholes_call_price(spot,strike,expiry,interest_rate,volatility):
      """ Analytic Black schole price for a European call

      Args:
        - spot: current spot
        - strike: call option's strike
        - expiry: expiry/maturity of the option (in years)
        - interest_rate: annualized interest rate
        - volatility: annualized volatility

      Returns:
        - price: the price of the European call
      """
      k = strike
      factor_plus  = (interest_rate+0.5*volatility**2)
      factor_minus = (interest_rate-0.5*volatility**2)
      d1 = (np.log(spot/k) + factor_plus*expiry) \
           /volatility/np.sqrt(expiry)
      d2 = (np.log(spot/k) + factor_minus*expiry) \
           /volatility/np.sqrt(expiry)
      price  = spot*norm.cdf(d1) - \
               k*np.exp(-interest_rate*expiry)*norm.cdf(d2)
      return price

def margrabe_option_price(spots,expiry,covariance_matrix):
    """ Analytic Magrabe price for an exchange option

    """
    sigma = np.sqrt(np.trace(covariance_matrix) \
                    - np.trace(np.fliplr(covariance_matrix)))
    if len(np.shape(spots))>1:
        s_plus  = spots[:,0]
        s_minus = spots[:,1]
        d1 = (np.log(s_plus/s_minus)+sigma**2*expiry*0.5)/sigma/np.sqrt(expiry)
        d2 = d1 - sigma*np.sqrt(expiry)
        price  = s_plus*norm.cdf(d1) - s_minus*norm.cdf(d2)
    else:
        s_plus  = spots[0]
        s_minus = spots[1]
        d1 = (np.log(s_plus/s_minus)+sigma**2*expiry)*0.5/sigma/np.sqrt(expiry)
        d2 = d1 - sigma*np.sqrt(expiry)
        price  = s_plus*norm.cdf(d1) - s_minus*norm.cdf(d2)
    return price
