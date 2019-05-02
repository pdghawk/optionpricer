from optionpricer import path
from optionpricer import option
import numpy as np
# monte carlo class perhaps
# members: price, stats object, epsilon
# so that as run, the stats object gets updated

def montecarlo_sa_vanilla_price(option,n_paths,spot,generator,r_param,vol_param):
    # should be: if(vanilla) do this, else: some other routine for asian, or american say
    expiry = option.get_expiry()
    future_spots = path.many_paths(n_paths,spot,generator,0.0,expiry,r_param,vol_param)
    _,_,_,discount = path.get_path_constants(0.0,expiry,r_param,vol_param)
    payoffs = option.get_option_payoff(future_spots)
    payoffs *= discount
    #print(np.cumsum(payoffs)/(1+np.arange(len(payoffs))))
    #running_mean = np.cumsum(payoffs)/(1+np.arange(len(payoffs)))
    return np.mean(payoffs)

class SAMonteCarlo:
    def __init__(self,option,generator):
        self._option   = option
        self._generator = generator
        #self.options   = {'eps':0.1,'max_steps':1e6,'steps_per_iter':200}
        self.price = 0.0
        self.eps   = 1.0e20
        self._iter_count = 0

    def solve_price(self,spot,r_param,vol_param,eps_tol=0.001,max_paths=1e6,paths_per_iter=200):
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
                    self.price = (old_price+new_price)*0.5
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
        self.price = 0.0
        self.eps   = 1.0e20
        self._iter_count = 0

    def get_iteration_count(self):
        return self._iter_count
