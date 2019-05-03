"""module for solving partial differential equations of Black Scholes form

"""

import numpy as np
from scipy.interpolate import interp1d
import copy

class FinDiffGrid1D:
    """ Defines a 1d grid for finite difference calculations
    """
    def __init__(self,min_value,max_value,number_of_nodes):
        self._min_value = min_value
        self._max_value = max_value
        self._number_of_nodes = number_of_nodes
        self.grid   = None
        self._delta = None
        self.values = None

    def make_linear_grid(self):
        """ creates the grid as a linearly spaced one

        self.grid is updated with linear grid
        self._delta is updated to the spacing between points in the grid
        """
        self.grid   = np.linspace(self._min_value,self._max_value,self._number_of_nodes)
        self._delta = self.grid[1]-self.grid[0]

    def set_values(self,values):
        """ set values of a function at the grid points

        self.values is then updated to hold the values passed to this method

        Args:
            - values: a series of values,
                      should be of length: self.number_of_nodes
        """
        assert(len(values)==self._number_of_nodes)
        self.values = values

    def get_delta(self,absolute=True):
        """ return the grid spacing

        Keyword Args:
            - absolute: if True, return absolute value of grid spacing
                        (defualts to True)

        Returns:
            - the grid spacing
        """
        if absolute:
            return np.abs(self._delta)
        else:
            return self._delta

    def get_number_of_nodes(self):
        """ get number of nodes in the grid
        Returns:
            - self.number_of_nodes
        """
        return self._number_of_nodes

    def clone(self):
        return copy.deepcopy(self)


class BoundaryConditions1D:
    """ Boundary conditions for 1d PDE boundary value problem
    """
    def __init__(self):
        self.lo_isdirichlet =False
        self.lo_isneumann   =False
        self.hi_isdirichlet =False
        self.hi_isneumann   =False

    def set_lo_dirichlet(self,value):
        """ set the lower bound of axis to be a Dirchlet condition

        Args:
            - value: value to set lower bound of axis to be at all times
        """
        assert(not self.lo_isneumann)
        self.lo_isdirichlet = True
        self.lo_bc = value

    def set_hi_dirichlet(self,value):
        """ set the higher bound of axis to be a Dirchlet condition

        Args:
            - value: value to set higher bound of axis to be at all times
        """
        assert(not self.hi_isneumann)
        self.hi_isdirichlet = True
        self.hi_bc = value

    def set_lo_neumann(self,derivitive_value):
        """ set the lower bound of axis to be a Neumann condition

        Args:
            - derivative_value: value to set derivitive at lower bound of axis
                                to be at all times
        """
        assert(not self.lo_isneumann)
        self.lo_isneumann = True
        self.lo_bc = derivitive_value

    def set_hi_Dirichlet(self,derivitive_value):
        """ set the higher bound of axis to be a Neumann condition

        Args:
            - derivative_value: value to set derivitive at higher bound of axis
                                to be at all times
        """
        assert(not self.hi_isdeumann)
        self.hi_isneumann = True
        self.hi_bc = derivitive_value

    def clone(self):
        return copy.deepcopy(self)


class HeatEquation1DPde:
    """ Define the 1d heat equation

    Define the 1d heat equation on specific space and time grids, and with
    specific boundary conditions


    """
    def __init__(self,diffusion_factor,space_grid,time_grid,boundary_conditions,initial_condition,all_time_output=False):
        self._diffusion_factor = diffusion_factor
        self._space = space_grid.clone()
        self._time  = time_grid.clone()
        if type(boundary_conditions) == BoundaryConditions1D:
            self._boundary_conditions = boundary_conditions.clone()
        else:
            rause TypeError("boundary conditions passed to HeatEquation1DPde \
                             should be a  bspde.BoundaryConditions1D object")
        self._initial_condition   = initial_condition
        self._timed_output_flag   = all_time_output

    def solve(self):
        """ Solve the 1d Heat equation
        Returns:
            - values of u at all space grid points and final time grid point,
              or values of u at all space and time grid points (if self._timed_output_flag)
        """

        dt = self._time.get_delta()
        dx = self._space.get_delta()
        # factors for tridiagonal problem (lhs)
        b = (1.0/dt + self._diffusion_factor/dx**2)
        a = -0.5*self._diffusion_factor/dx**2
        c = -0.5*self._diffusion_factor/dx**2
        # factors for tridiagonal problem (rhs)
        bprime = (1.0/dt - self._diffusion_factor/dx**2)
        aprime = 0.5*self._diffusion_factor/dx**2
        cprime = 0.5*self._diffusion_factor/dx**2
        # factors for forward Euler step
        euler_a = dt*self._diffusion_factor/dx**2
        euler_b = (-2.0*euler_a + 1.0)
        # initialise u at starting time
        self._space.set_values(self._initial_condition)

        n_space = self._space.get_number_of_nodes()
        n_time = self._time.get_number_of_nodes()

        if self._timed_output_flag:
            timed_solution = np.zeros((n_time,n_space))
            timed_solution[0,:] = self._initial_condition

        for i in range(n_time-1):
            if i<10: # do forward Euler steps for first few steps
                d = euler_a*np.concatenate((np.zeros((1,)),self._space.values[:n_space-1]))
                d+= euler_b*self._space.values
                d+= euler_a*np.concatenate((self._space.values[1:],np.zeros((1,))))
                self._space.set_values(d)
                self.apply_boundary_conditions()
                if self._timed_output_flag:
                    timed_solution[i+1,:]=self._space.values
            else:
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
        """ Apply Boundary conditions to u
        """
        lo_bc_done = False
        hi_bc_done = False

        if self._boundary_conditions.lo_isdirichlet:
            self._space.values[0] = self._boundary_conditions.lo_bc

        if self._boundary_conditions.hi_isdirichlet:
            self._space.values[self._space.get_number_of_nodes()-1] = self._boundary_conditions.hi_bc

        return None

    def clone(self):
        return copy.deepcopy(self)


class BlackScholesSingleAssetPricer:
    """ Define Black Scholes problem for PDE solution
    """
    def __init__(self,option,interest_rate,volatility,boundary_conditions):
        self._option=option.clone()
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.boundary_conditions = boundary_conditions.clone()

    def solve(self,spot,get_price_on_whole_grid=False):
        """ Solve the pde problem
        Args:
            -spot: spot at which to calculate the price
        Keyword Args:
            -get_price_on_whole_grid: (optional) If True return price for all spots
                                      on the calculation grid.
        """
        expiry = self._option.get_expiry()
        diffusion_coeff = 0.5*self.volatility**2
        r = self.interest_rate
        k = self._option._payoff.get_strike()

        t_grid = FinDiffGrid1D(self._option.get_expiry(),0.0,int(800*self._option.get_expiry()))
        t_grid.make_linear_grid()

        spot_to_x = lambda s:np.log(s/k) #only defined at expiry
        x_to_spot = lambda x,tau:k*np.exp(x - (r - diffusion_coeff)*tau)

        x_max = max(2*np.amax(spot),10.0)
        x_min = min(-10.0,np.amin(spot))
        x_grid = FinDiffGrid1D(x_min,x_max,int((x_max-x_min)/0.025))
        x_grid.make_linear_grid()

        init_cond = self._option.get_option_payoff(k*np.exp(x_grid.grid))

        heat_eq = HeatEquation1DPde(diffusion_coeff,x_grid,t_grid,self.boundary_conditions,init_cond)
        solution_on_x = heat_eq.solve()
        # note that heat eq solution stored in x_grid.values
        option_price_at_all_spots = solution_on_x*np.exp(-r*expiry)
        #print(option_price_at_all_spots)
        if get_price_on_whole_grid:
            return option_price_at_all_spots
        # interpolate to get value at the chosen spot
        interpolator = interp1d(x_to_spot(x_grid.grid,expiry),option_price_at_all_spots)

        return interpolator(spot)

    def clone(self):
        return copy.deepcopy(self)




def tridiag_varying(a,b,c,d,dtype_=np.float64):
    """ Solve tridiagonal problem for non-constant band entries

    Args:
        - a: lower diagonal entries
        - b: diagonal entries
        - c: upper diagonal entries
    Keyword Args:
        - dtype_: (optional) the data type to solve for
    """
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
    """ Solve tridiagonal problem for constant band entries

    Args:
        - a: lower diagonal entries
        - b: diagonal entries
        - c: upper diagonal entries
    Keyword Args:
        - dtype_: (optional) the data type to solve for
    """
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

# for 2d problems where tridiag won't work probably use:
# scipy.sparse.linalg.cg
