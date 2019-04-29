"""module for solving partial differential equations of Black Scholes form

"""

import numpy as np

class FinDiffGrid1D:
    def __init__(self,min_value,max_value,number_of_nodes):
        self._min_value = min_value
        self._max_value = max_value
        self._number_of_nodes number_of_nodes
        self._grid = np.linspace(min_value,max_value,number_of_nodes) 
