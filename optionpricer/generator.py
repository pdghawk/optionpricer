import numpy as np
import copy

class normal:
    def __init__(self,mean=0.0,var=1.0):
        self.mean = mean
        self.var  = var

    def get_samples(self,n_samples=1):
        return np.random.normal(self.mean,self.var,n_samples)

    def clone(self):
        return copy.deepcopy(self)


class antithetic:
    def __init__(self,gen):
        self._negate     = False
        self._generator  = gen.clone()
        self.last_sample = None

    def get_samples(self,n_samples=1):
        assert(n_samples>0)
        if (n_samples == 1 and not self._negate):
            samples = self.generator.get_samples(n_samples)
            self._negate = not self._negate
            self.last_sample = samples
            return samples
        elif (n_samples == 1 and self._negate):
            samples = -self.last_sample
            self._negate = not self._negate
            self.last_sample = samples
            return samples
        else:
            # get a certain number, and then repeat them, but negative
            # negate parameter should be swapped depending on n_sampeles even/odd
            samples = np.zeros((n_samples,))
            if n_samples%2==0:
                n_get    = n_samples//2
            else:
                n_get    = n_samples//2+1

            samples[:n_get] = self._generator.get_samples(n_get)
            samples[n_get:] = -samples[:(n_samples-n_get)]

            self.last_sample = None
            self._negate = False

            return samples

    def clone(self):
        return copy.deepcopy(self)
