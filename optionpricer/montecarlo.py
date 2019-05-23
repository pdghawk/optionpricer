""" Module for calculating Monte Carlo Prices """
from optionpricer import path
from optionpricer import option
import numpy as np
# monte carlo class perhaps
# members: price, stats object, epsilon
# so that as run, the stats object gets updated

def montecarlo_sa_vanilla_price(option_,n_paths,spot,generator,r_param,vol_param):
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


class SAMonteCarlo:
    """ Single Asset Monte Carlo pricing

    For a given option and stistical generator, get the price of the option
    """
    def __init__(self,option_,generator):
        self._option   = option_.clone()
        self._generator = generator.clone()
        #self.options   = {'eps':0.1,'max_steps':1e6,'steps_per_iter':200}
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
            while (self.eps>eps_tol and self._iter_count<max_paths):
                new_price = montecarlo_sa_vanilla_price(self._option,
                                                        paths_per_iter,
                                                        spot,
                                                        self._generator,
                                                        r_param,
                                                        vol_param)
                old_price = self.price
                if self._iter_count>0:
                    self.price = old_price + (new_price-old_price)/float(self._iter_count)
                else:
                    self.price = new_price
                self.eps = np.abs(old_price-new_price)
                # print(self.eps)
                # print(old_price,new_price,self.price)
                self._iter_count +=paths_per_iter
            if(self._iter_count>=max_paths and self.eps>eps_tol):
                print("\n\nWarning: SAMonteCarlo price stopped before convergence")
            return self.price
        else:
            raise NotImplementedError("Currently only VanillaOption's \
                                       can be priced by SAMonteCarlo")
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
# multiasset funstions and classes
# ------------------------------------------------------------------------------
def montecarlo_ma_vanilla_price(option_,n_paths,spots,generator,r_param,vol_param):
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
        future_spots = path.many_many_asset_paths(n_paths,spot,generator,0.0,expiry,r_param,cholesky_param)
        _,_,_,discount = path.get_multipath_constants(0.0,expiry,r_param,cholesky_param)
        payoffs = option_.get_option_payoff(future_spots)
        payoffs *= discount
    else:
        raise NotImplementedError("only VanillaOption's can be  \
                                   priced by montecarlo_sa_vanilla_price")
    #print(np.cumsum(payoffs)/(1+np.arange(len(payoffs))))
    #running_mean = np.cumsum(payoffs)/(1+np.arange(len(payoffs)))
    return np.mean(payoffs)

class MAMonteCarlo:
    """ Multi Asset Monte Carlo pricing

    For a given option and stistical generator, get the price of the option
    """
    def __init__(self,option_,generator):
        self._option   = option_.clone()
        if self._option._payoff,n_assets<2:
            raise TypeError("option supplied to MAMonteCarlo has a payoff only \
                             on a single asset, should be a multi-asset payoff")
        self._generator = generator.clone()
        #self.options   = {'eps':0.1,'max_steps':1e6,'steps_per_iter':200}
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
                                                        spot,
                                                        self._generator,
                                                        r_param,
                                                        cholesky_param)
                old_prices = self.prices
                if self._iter_count>0:
                    self.prices = old_prices + (new_prices-old_prices)/float(self._iter_count)
                else:
                    self.prices = new_price
                self.eps = np.abs(old_prices-new_prices)
                # print(self.eps)
                # print(old_price,new_price,self.price)
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
