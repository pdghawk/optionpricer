""" Module for calculating Monte Carlo Prices """
from optionpricer import path
from optionpricer import option
import numpy as np


def montecarlo_sa_vanilla_price(option_,n_paths,spot,generator,r_param,vol_param,dt=0.0):
    """ get the price of a vanilla option based on n_paths

    Args:
        - option_: the option to price
        - n_paths: number of paths to use
        - spot: current spot
        - generator: (of optionpricer.generator type) a stastical genertor for the path
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
    Returns:
        - price: the statistically expected price
    """
    if type(option_) == option.VanillaOption:
        expiry = option_.the_expiry
        future_spots = path.many_single_asset_paths(n_paths,spot,generator,0.0,expiry,r_param,vol_param)
        _,_,_,discount = path.get_path_constants(0.0,expiry,r_param,vol_param)
        payoffs = option_.get_option_payoff(future_spots)
        payoffs *= discount
    else:
        raise NotImplementedError("only VanillaOption's can be  \
                                   priced by montecarlo_sa_vanilla_price")
    #print(np.cumsum(payoffs)/(1+np.arange(len(payoffs))))
    #running_mean = np.cumsum(payoffs)/(1+np.arange(len(payoffs)))
    return np.mean(payoffs)

def montecarlo_sa_barrier_price(option_,n_paths,spot,generator,r_param,vol_param,dt=1.0/252.2):
    """ get the price of a vanilla option based on n_paths

    Args:
        - option_: the option to price
        - n_paths: number of paths to use
        - spot: current spot
        - generator: (of optionpricer.generator type) a stastical genertor for the path
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
    Returns:
        - price: the statistically expected price
    """
    #TODO: Browninan bridge (prob of crossing a barrier between two points for
    # geometric brownian motion). would be more efficient
    if type(option_) == option.BarrierOption:
        expiry = option_.the_expiry
        times  = np.linspace(0.0,expiry,int(expiry/dt))
        future_spots = path.many_single_asset_timed_paths(n_paths,spot,generator,times,r_param,vol_param)
        _,_,_,discount = path.get_path_constants(0.0,expiry,r_param,vol_param)
        payoffs = np.zeros((n_paths,))
        for pathway in range(n_paths):
            option_.reset_validity()
            #print(option_.validity)
            option_.update_option_validity(future_spots[:,pathway])
            #print(option_.validity)
            payoffs[pathway] = option_.get_option_payoff(future_spots[-1,pathway])
            #print(payoffs[pathway],future_spots[-1,pathway],option_.barrier)
        payoffs *= discount

    else:
        raise NotImplementedError("only BarrierOption's can be  \
                                   priced by montecarlo_sa_barrier_price")
    return np.mean(payoffs)

class SAMonteCarlo:
    """ Single Asset Monte Carlo pricing

    For a given option and stistical generator, get the price of the option
    """
    def __init__(self,option_,generator):
        self._option   = option_.clone()
        self._generator = generator.clone()
        self.price = 0.0
        self.eps   = 1.0e20
        self._iter_count = 0

    def solve_price(self,spot,r_param,vol_param,eps_tol=0.001,max_paths=1e6,paths_per_iter=200):
        """ get the price of the option

        Args:
            - spot: the current spot
            - r_param: (of optionpricer.parameter type) interest rate parameter
            - vol_param: (of optionpricer.parameter type) volatility parameter
        Keyword Args:
            - eps_tol: (=0.001) tolerance for convergence
            - max_paths (=1e6): maximal number of paths to calculate price with
            - paths_per_iter: (=200) number of paths to use at each iteration
        Return:
            - price: price of the option
        """
        self.reset()
        if type(self._option) == option.VanillaOption:
            self.iterate_price(montecarlo_sa_vanilla_price,
                               spot,
                               r_param,
                               vol_param,
                               eps_tol=eps_tol,
                               max_paths=max_paths,
                               paths_per_iter=paths_per_iter)
            return self.price
        elif(type(self._option) == option.BarrierOption):
            #print("pricing Barrier")
            self.reset()
            halve_count = 0 # how many times have halved the time grid spacing
            dt = (1.0/1.0)*1.0/252.0
            # halve the time-grid spacing until converged.
            while(self.eps>eps_tol and halve_count<6):
                old_price=self.price
                self.reset()
                self.iterate_price(montecarlo_sa_barrier_price,
                                   spot,
                                   r_param,
                                   vol_param,
                                   eps_tol=eps_tol*0.1,
                                   max_paths=max_paths,
                                   paths_per_iter=paths_per_iter,
                                   dt=dt)

                self.eps = np.abs(old_price-self.price)
                dt /= 2.0
                halve_count+=1
                print("prices",old_price,self.price,self.eps,eps_tol)
            print(halve_count,dt,int(self._option.get_expiry()/dt))
            return self.price
        else:
            raise NotImplementedError("Currently the option type passed to \
                                       SAMonteCarlo cannot be priced - it is \
                                       not supported.")

    def iterate_price(self,pricing_function,spot,r_param,vol_param,eps_tol=0.001,max_paths=1e6,paths_per_iter=200,dt=1.0/252.0):
        while (self.eps>eps_tol and self._iter_count<max_paths):
            new_price = pricing_function(self._option,
                                         paths_per_iter,
                                         spot,
                                         self._generator,
                                         r_param,
                                         vol_param,
                                         dt=dt)
            old_price = self.price
            if self._iter_count>0:
                self.price = old_price + (new_price-old_price)/float(self._iter_count)
            else:
                self.price = new_price
            self.eps = np.abs(old_price-new_price)
            self._iter_count +=paths_per_iter
        if(self._iter_count>=max_paths and self.eps>eps_tol):
            print("\n\nWarning: SAMonteCarlo price stopped before convergence")

        return None

    def reset(self):
        """ reset the pricer
        """
        self.price = 0.0
        self.eps   = 1.0e20
        self._iter_count = 0

    def get_iteration_count(self):
        """ get the number of iterations that were performed
        Returns:
            - iteration_count
        """
        return self._iter_count

    def clone(self):
        return copy.deepcopy(self)

# ------------------------------------------------------------------------------
# multiasset functions and classes
# ------------------------------------------------------------------------------

def montecarlo_ma_vanilla_price(option_,n_paths,spots,generator,r_param,cholesky_param):
    """ get the price of a vanilla option based on n_paths

    Args:
        - option_: the option to price
        - n_paths: number of paths to use
        - spot: current spot
        - generator: (of optionpricer.generator type) a stastical genertor for the path
        - r_param: (of optionpricer.parameter type) interest rate parameter
        - vol_param: (of optionpricer.parameter type) volatility parameter
    Returns:
        - price: the statistically expected price
    """
    if type(option_) == option.VanillaOption:
        #expiry = option_.get_expiry()
        expiry = option_.the_expiry
        future_spots = path.many_multi_asset_paths(n_paths,spots,generator,0.0,expiry,r_param,cholesky_param)
        _,_,_,discount = path.get_multipath_constants(0.0,expiry,r_param,cholesky_param)
        payoffs = option_.get_option_payoff(future_spots.T)
        payoffs *= discount
    else:
        raise NotImplementedError("only VanillaOption's can be  \
                                   priced by montecarlo_sa_vanilla_price")

    return np.mean(payoffs)

class MAMonteCarlo:
    """ Multi Asset Monte Carlo pricing

    For a given option and stistical generator, get the price of the option
    """
    def __init__(self,option_,generator):
        self._option   = option_.clone()
        if self._option._payoff.n_assets<2:
            raise TypeError("option supplied to MAMonteCarlo has a payoff only \
                             on a single asset, should be a multi-asset payoff")
        self._generator = generator.clone()
        self.prices = 0.0
        self.eps   = 1.0e20
        self._iter_count = 0

    def solve_price(self,spots,r_param,cholesky_param,eps_tol=0.001,max_paths=1e6,paths_per_iter=200):
        """ get the price of the option

        Args:
            - spot: the current spots of all assets the option depends on
            - r_param: (of optionpricer.parameter type) interest rate parameter
            - vol_param: (of optionpricer.parameter type) volatility parameter
        Keyword Args:
            - eps_tol: (=0.001) tolerance for convergence
            - max_paths (=1e6): maximal number of paths to calculate price with
            - paths_per_iter: (=200) number of paths to use at each iteration
        Return:
            - price: price of the option
        """
        self.reset()
        if type(self._option) == option.VanillaOption:
            while (self.eps>eps_tol and self._iter_count<max_paths):
                new_prices = montecarlo_ma_vanilla_price(self._option,
                                                        paths_per_iter,
                                                        spots,
                                                        self._generator,
                                                        r_param,
                                                        cholesky_param)
                old_prices = self.prices
                #print(old_prices,new_prices)
                if self._iter_count>0:
                    self.prices = old_prices + (new_prices-old_prices)/float(self._iter_count)
                else:
                    self.prices = new_prices
                self.eps = np.abs(old_prices-new_prices)
                self._iter_count +=paths_per_iter
            if(self._iter_count>=max_paths and self.eps>eps_tol):
                print("\n\nWarning: SAMonteCarlo price stopped before convergence")
            return self.prices
        else:
            raise NotImplementedError("Currently only VanillaOption's \
                                       can be priced by MAMonteCarlo")
    def reset(self):
        """ reset the pricer
        """
        self.prices = 0.0
        self.eps   = 1.0e20
        self._iter_count = 0

    def get_iteration_count(self):
        """ get the number of iterations that were performed
        Returns:
            - iteration_count
        """
        return self._iter_count

    def clone(self):
        return copy.deepcopy(self)
