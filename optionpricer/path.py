import numpy as np


def get_path_constants(time0,time1,r_param, vol_param):
    """ get path constants within a time-window
    Args:
        - time0: beginning of time window
        - time1: end of time window
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
    Returns:
         - (r,var,mu,discount), where:
             - r: interest rate in this time region
             - var: variance in this time region
             - mu: r-0.5*var in this time region
             - discount: discount factor for time-vlaue of money in this time region
    """
    r = r_param.integral(time0,time1)
    # variance
    var = vol_param.square_integral(time0,time1)
    # risk neutral movement position
    mu = r - 0.5*var
    # discount to be applied due to time-value of money
    discount = np.exp(-r)
    return r,var,mu,discount

def single_path(spot, generator, time0, time1, r_param, vol_param):
    """ calculate a future spot value at a later time

    Args:
        - spot: current spot
        - generator: an optionpricer.generator object for generating statistical path
        - time0: initial time
        - time1: time at which to get future_spot
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
        Returns:
            - future_spot: value for spot at time1
    """
    r,var,mu,discount = get_path_constants(time0, time1,r_param, vol_param)
    rand_val = generator.get_samples(1)
    future_spot = spot*np.exp(mu)
    future_spot *= np.exp(np.sqrt(var)*rand_val)
    return future_spot

def single_timed_path(spot,generator,times,r_param,vol_param):
    """ calculate a future spot value on a list of futre times

    Args:
        - spot: current spot
        - generator: an optionpricer.generator object for generating statistical path
        - times: times at which to get spot (from initial time to a final time)
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
    Returns:
        - future_spots: value for spot at times in 'times'
    """
    future_spots    = np.zeros_like(times)
    future_spots[0] = spot
    if isinstance(generator, Antithetic):
        print("Warning ( optionpricer.path.single_timed_path() ): generating a \
               timed sequence with antithetic generator")
    for i range(1,len(times)):
        r,var,mu,discount = get_path_constants(times[i-1], times[i],r_param, vol_param)
        rand_vals = generator.get_samples(1)
        future_spots[i] = future_spots[i-1]*np.exp(mu)
        future_spots[i] *= np.exp(np.sqrt(var)*rand_vals)
    #future_spots *= discount
    return future_spots

def many_single_asset_paths(n_paths, spot, generator, time0, time1, r_param, vol_param):
    """ calculate many future spot value at a later time

    Args:
        - n_paths: number of paths to calculate
        - spot: current spot
        - generator: an optionpricer.generator object for generating statistical path
        - time0: initial time
        - time1: time at which to get future_spot
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
    Returns:
        - future_spots: values for spot at time1
    """
    assert(n_paths>0)
    if n_paths==1:
        return single_path(spot, generator, time0, time1, r_param, vol_param)
    r,var,mu,discount = get_path_constants(time0, time1,r_param, vol_param)
    rand_vals = generator.get_samples(n_paths)
    #print("rands = ", rand_vals)
    future_spots = spot*np.exp(mu)
    future_spots *= np.exp(np.sqrt(var)*rand_vals)
    #future_spots *= discount
    return future_spots


def get_multipath_constants(time0,time1,r_param,cholesky_param):
    """ get path constants within a time-window
    Args:
        - time0: beginning of time window
        - time1: end of time window
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
    Returns:
         - (r,var,mu,discount), where:
             - r: interest rate in this time region
             - var: variance in this time region
             - mu: r-0.5*var in this time region
             - discount: discount factor for time-vlaue of money in this time region
    """
    r = r_param.integral(time0,time1)
    # variance
    vars = cholesky_param.diag_square_integral(time0,time1)
    # risk neutral movement position
    mu = r - 0.5*vars
    # discount to be applied due to time-value of money
    discount = np.exp(-r)
    return r,vars,mu,discount

def single_multi_asset_path(spots,generator,time0,time1,r_param, cholesky_param):
    r,vars,mu,discount = get_multipath_constants(time0, time1,r_param, cholesky_param)
    # get samples that are not antithetic or decorated
    rand_vals_standard = generator.get_simple_samples(len(spots))
    rand_vals = np.dot(cholesky_param.mean(time0,time1),rand_vals_standard)
    future_spots = spots*np.exp(mu)
    future_spots *= np.exp(np.sqrt(vars)*rand_vals)
    return future_spots
