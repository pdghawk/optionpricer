import numpy as np


def get_path_constants(r_param, vol_param,time0,time1):
    r = r_param.integral(time0,time1)
    # variance
    var = vol_param.square_integral(time0,time1)
    # risk neutral movement position
    mu = r - 0.5*var
    # discount to be applied due to time-value of money
    discount = np.exp(-r)
    return r,var,mu,discount

def single_path(spot, generator, time0, time1, r_param, vol_param):
    r,var,mu,discount = get_path_constants(r_param, vol_param, time0, time1)
    rand_val = generator.get_samples(1)
    future_spot = spot*np.exp(mu)
    future_spot *= np.exp(np.sqrt(var)*rand_val)
    future_spot *= discount
    return future_spot

def single_timed_path(spot,generator,times,r_param,vol_param):
    r,var,mu,discount = get_path_constants(r_param, vol_param, time0, time1)
    rand_vals = generator.get_samples(len(times))
    future_spots = spot*np.exp(mu)
    future_spots *= np.exp(np.sqrt(var)*rand_vals)
    future_spots *= discount
    return future_spots

def many_paths(n_paths, spot, generator, time0, time1, r_param, vol_param):
    r,var,mu,discount = get_path_constants(r_param, vol_param, time0, time1)
    rand_vals = generator.get_samples(n_paths)
    #print("rands = ", rand_vals)
    future_spots = spot*np.exp(mu)
    future_spots *= np.exp(np.sqrt(var)*rand_vals)
    future_spots *= discount
    return future_spots
