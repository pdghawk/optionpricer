"""module for solving partial differential equations of Black Scholes form

"""

import numpy as np


class FinDiffGrid1D:

    def __init__(self,min_value,max_value,number_of_nodes):
        self._min_value = min_value
        self._max_value = max_value
        self._number_of_nodes = number_of_nodes
        self.grid   = None
        self._delta = None
        self.values = None

    def make_linear_grid(self):
        self.grid   = np.linspace(self._min_value,self._max_value,self._number_of_nodes)
        self._delta = self.grid[1]-self.grid[0]

    def set_values(self,values):
        assert(len(values)==self._number_of_nodes)
        self.values = values

    def get_delta(self):
        return self._delta

    def get_number_of_nodes(self):
        return self._number_of_nodes


class BoundaryConditions1D:

    def __init__(self):
        self.lo_isdirichlet =False
        self.lo_isneumann   =False
        self.hi_isdirichlet =False
        self.hi_isneumann   =False
        #self.lo_bc
        #self.hi_bc

    def set_lo_dirichlet(self,value):
        assert(not self.lo_isneumann)
        self.lo_isdirichlet = True
        self.lo_bc = value

    def set_hi_dirichlet(self,value):
        assert(not self.hi_isneumann)
        self.hi_isdirichlet = True
        self.hi_bc = value

    def set_lo_neumann(self,derivitive_value):
        assert(not self.lo_isneumann)
        self.lo_isneumann = True
        self.lo_bc = derivitive_value

    def set_hi_Dirichlet(self,derivitive_value):
        assert(not self.hi_isdeumann)
        self.hi_isneumann = True
        self.hi_bc = derivitive_value


class HeatEquation1DPde:
    def __init__(self,diffusion_factor,space_grid,time_grid,boundary_conditions,initial_condition,all_time_output=False):
        self._diffusion_factor = diffusion_factor
        self._space = space_grid
        self._time  = time_grid
        self._boundary_conditions = boundary_conditions
        self._initial_condition   = initial_condition
        self._timed_output_flag   = all_time_output

    def solve(self):
        dt = self._time.get_delta()
        dx = self._space.get_delta()
        b = (1.0/dt + self._diffusion_factor/dx**2)
        a = -0.5*self._diffusion_factor/dx**2
        c = -0.5*self._diffusion_factor/dx**2

        bprime = (1.0/dt - self._diffusion_factor/dx**2)
        aprime = 0.5*self._diffusion_factor/dx**2
        cprime = 0.5*self._diffusion_factor/dx**2

        self._space.set_values(self._initial_condition)

        n_space = self._space.get_number_of_nodes()
        n_time = self._time.get_number_of_nodes()

        if self._timed_output_flag:
            timed_solution = np.zeros((n_time,n_space))
            timed_solution[0,:] = self._initial_condition

        for i in range(n_time-1):
            # TODO: if i<2 do forward step not crank nic

            d = aprime*np.concatenate((np.zeros((1,)),self._space.values[:n_space-1]))
            d+= bprime*self._space.values
            d+= cprime*np.concatenate((self._space.values[1:],np.zeros((1,))))

            self._space.set_values(tridiag_constant(a,b,c,d))

            self.apply_boundary_conditions()
            if self._timed_output_flag:
                timed_solution[i+1,:]=self._space.values

        if self._timed_output_flag:
            return timed_solution
        else:
            return self._space.values

    def apply_boundary_conditions(self):
        lo_bc_done = False
        hi_bc_done = False

        if self._boundary_conditions.lo_isdirichlet:
            self._space.values[0] = self._boundary_conditions.lo_bc

        if self._boundary_conditions.hi_isdirichlet:
            self._space.values[self._space.get_number_of_nodes()-1] = self._boundary_conditions.hi_bc

        return None


def tridiag_varying(a,b,c,d,dtype_=np.float64):
    N  = len(d)
    cp = np.zeros((N-1,),dtype=dtype_)
    dp = np.zeros((N,),dtype=dtype_)

    cp[0] = c[0]/b[0];
    dp[0] = d[0]/b[0];

    for i in range(1,N-1):
        cp[i] = c[i]/(b[i] - a[i-1]*cp[i-1])
        dp[i] = (d[i] - a[i-1]*dp[i-1])/(b[i] - a[i-1]*cp[i-1])
    dp[N-1] = (d[N-1] - a[N-2]*dp[N-2])/(b[N-1] - a[N-2]*cp[N-2])

    out = np.zeros((N,),dtype=dtype_)
    out[-1] = dp[-1]
    for i in range(N-2,-1,-1):
        out[i] = dp[i] - cp[i]*out[i+1];

    return out

def tridiag_constant(a,b,c,d,dtype_=np.float64):
    N = len(d)

    cp = np.zeros((N-1,),dtype=dtype_)
    dp = np.zeros((N,),dtype=dtype_)

    cp[0] = c/b;
    dp[0] = d[0]/b;

    for i in range(1,N-1):
        cp[i] = c/(b - a*cp[i-1])
        dp[i] = (d[i] - a*dp[i-1])/(b - a*cp[i-1])
    dp[N-1] = (d[N-1] - a*dp[N-2])/(b - a*cp[N-2])

    out = np.zeros((N,),dtype=dtype_)
    out[-1] = dp[-1]

    for i in range(N-2,-1,-1):
        out[i] = dp[i] - cp[i]*out[i+1];

    return out
