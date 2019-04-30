from optionpricer import payoff
from optionpricer import option
from optionpricer import bspde
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

call_payoff = payoff.CallPayOff(10.0)

spots=np.linspace(0.0,20.0,10)
print(call_payoff.get_payoff(spots))
print(call_payoff.get_payoff(spots[0]))
#call_payoff = payoff.CallPayOff([10.0,15.0])

#print(call_payoff)
print(str(call_payoff))

call_option = option.VanillaOption(call_payoff,1.0)
print(call_option.get_option_payoff(spots))
print(call_option.get_expiry())

print(str(call_option))

# badpayoff = payoff.BadPayOff(10.0)
# call_option_bad = option.VanillaOption(badpayoff,1.0)
# print(call_option_bad.get_option_payoff(spots))

################################################################################

# timegrid  = bspde.FinDiffGrid1D(0.0,1.0,7000)
# timegrid.make_linear_grid()
# spacegrid = bspde.FinDiffGrid1D(0.0,np.pi,200)
# spacegrid.make_linear_grid()
# #spacegrid.set_values()
# bcs = bspde.BoundaryConditions1D()
# bcs.set_lo_dirichlet(0.0)
# bcs.set_hi_dirichlet(0.0)
#
# initial_cond = np.sin(spacegrid.grid)**3
#
#
# #print(spacegrid.grid)
# heat_eq = bspde.HeatEquation1DPde(1.0,spacegrid,timegrid,bcs,initial_cond,all_time_output=True)
# output = heat_eq.solve()
#
# print('myvals', spacegrid.values)
#
# plt.plot(spacegrid.grid,initial_cond,'k')
# plt.plot(spacegrid.grid,output[100,:],'b')
# # plt.plot(spacegrid.grid,output[200,:],'b')
# plt.plot(spacegrid.grid,output[-1,:],'m')
# plt.show()
#
# plt.plot(timegrid.grid,output[:,100],'k')
# plt.show()

# print(bspde.tridiag_constant(-1.0,2.0,-1.0,[1.0,2.0,3.0]))
# print(bspde.tridiag_varying([-1.0]*3,[2.0]*3,[-1.0]*3,[1.0,2.0,3.0]))
# print(bspde.tridiag_constant(-2006.0, 1994012.0, -2006.0,initial_cond))

################################################################################
call_payoff = payoff.CallPayOff(50.0)

call_option = option.VanillaOption(call_payoff,3.0)

spotgrid = bspde.FinDiffGrid1D(0.1,120.0,100)
spotgrid.make_linear_grid()

bcs = bspde.BoundaryConditions1D()
#bcs.set_lo_dirichlet(call_option.get_option_payoff(spotgrid.grid[0]))
#bcs.set_hi_dirichlet(call_option.get_option_payoff(spotgrid.grid[-1]))

bspricer = bspde.BlackScholesSingleAssetPricer(call_option,0.1,0.02,bcs)

spots=np.linspace(20.0,70.0,30)
prices=[]
for s in spots:
    prices.append( bspricer.solve(s,spotgrid) )

plt.plot(spots, prices, 'k')
plt.show()
