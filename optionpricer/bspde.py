"""module for solving partial differential equations of Black Scholes form

"""

import numpy as np


class FinDiffGrid1D:

    def __init__(self,min_value,max_value,number_of_nodes):
        self._min_value = min_value
        self._max_value = max_value
        self._number_of_nodes number_of_nodes
        self.grid = None

    def make_linear_grid(self):
        self.grid = np.linspace(min_value,max_value,number_of_nodes)


class BoundaryConditions1D:

    def __init__(self):
        self.lo_isdirichlet =False
        self.lo_isneumann   =False
        self.hi_isdirichlet =False
        self.hi_isneumann   =False
        #self.lo_bc
        #self.hi_bc

    def set_lo_dirichlet(self,value)
        assert(not self.lo_isneumann)
        self.lo_isdirichlet = True
        self.lo_bc = value

    def set_hi_dirichlet(self,value)
        assert(not self._hi_isneumann)
        self.hi_isdirichlet = True
        self.hi_bc = value

    def set_lo_neumann(self,derivitive_value)
        assert(not self._lo_isneumann)
        self.lo_isneumann = True
        self.lo_bc = derivitive_value

    def set_hi_Dirichlet(self,derivitive_value)
        assert(not self._hi_isdeumann)
        self.hi_isneumann = True
        self.hi_bc = derivitive_value


# class HeatEquation1DPde:
#     def __init__(self,diffusion_factor,time_grid):
#         self._diffusion_factor = diffusion_factor
#
#     def solve(self, time_grid, spatial_grid):
