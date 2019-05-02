from optionpricer import payoff
from optionpricer import option
from optionpricer import bspde
from optionpricer import analytics
from optionpricer import parameter as prmtr
from optionpricer import path
from optionpricer import generator
from optionpricer import montecarlo
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

print(type(call_option) is option.VanillaOption)

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

# strike = 50.0
# expiry = 1.0 #(18.0/12.0)
#
# call_payoff = payoff.CallPayOff(strike)
#
# call_option = option.VanillaOption(call_payoff,expiry)
#
# spotgrid = bspde.FinDiffGrid1D(1.0e-15,120.0,5000)
# spotgrid.make_linear_grid()
#
# bcs = bspde.BoundaryConditions1D()
# #bcs.set_lo_dirichlet(call_option.get_option_payoff(spotgrid.grid[0]))
# #bcs.set_lo_dirichlet(0.0)
# #bcs.set_hi_dirichlet(call_option.get_option_payoff(spotgrid.grid[-1]))
# r = 0.024
# sig = 0.12
# bspricer = bspde.BlackScholesSingleAssetPricer(call_option,r,sig,bcs)
#
# spots=np.linspace(20.0,70.0,30)
# # prices=[]
# # bs_prices=[]
# # for s in spots:
# #     prices.append( bspricer.solve(s,spotgrid) )
# #     bs_prices.append( analytics.black_scholes_call_price(s,strike,expiry,r,sig) )
#
# prices    = bspricer.solve(spots)
# bs_prices = analytics.black_scholes_call_price(spots,strike,expiry,r,sig)
# plt.plot(spots, prices, 'k')
# plt.plot(spots, bs_prices, 'r')
# plt.show()

################################################################################


# strike = 50.0
# expiry = 1.0
# r = 0.024
# sig = 0.12
#
# payoff = payoff.PutPayOff(strike)
# option = option.VanillaOption(payoff,expiry)
# bcs = bspde.BoundaryConditions1D()
# bspricer = bspde.BlackScholesSingleAssetPricer(option,r,sig,bcs)
# spots=np.linspace(20.0,70.0,30)
# prices    = bspricer.solve(spots)
# plt.plot(spots, prices, 'k')
# plt.show()

################################################################################
# expiry    = 1.0/12.0
# Nt        = 200
# times     = np.linspace(0,expiry,Nt)
#
# r0   = 0.024
# sig0 = 0.12
# r_param   = prmtr.SimpleParam(r0)
# vol_param = prmtr.SimpleParam(sig0)
#
# S0 = 100.0
#
# # Npaths  = 1500
# # mypaths = np.zeros((Nt,Npaths))
# # for i in range(Npaths):
# #     mypaths[:,i] = path.single_timed_lognormal_path(S0,times,r_param,vol_param)
# #
# # mypaths0 = np.zeros((Npaths,))
# # for i in range(Npaths):
# #     mypaths0[i] = path.single_lognormal_path(S0,0,times[-1],r_param,vol_param)
#
# # plt.plot(times,mypaths)
# # plt.show()
# #
# # plt.plot(times,np.log(mypaths)-np.log(S0),alpha=0.3)
# # plt.plot(times,(r0-0.5*sig0**2)*times,'k--')
# # plt.plot(times,3*sig0*np.sqrt(times),'r--')
# # plt.plot(times,(r0-0.5*sig0**2)*times+ 3*sig0*np.sqrt(times),'b--')
# # plt.show()
# #
# # vals,bins,ppp = plt.hist(np.log(np.squeeze(mypaths[-1,:]))-np.log(S0)-(r0-0.5*sig0**2)*times[-1],bins=100,alpha=0.5,color='r')
# # plt.hist(np.log(mypaths0)-np.log(S0)-(r0-0.5*sig0**2)*times[-1],bins=100,alpha=0.3,color='b')
# # x_ax = bins[:-1]+0.5*(bins[1]-bins[0])
# # plt.plot(x_ax, np.mean(vals[len(vals)//2-5:len(vals)//2+5])*np.exp(-0.5*(x_ax/(sig0*np.sqrt(times[-1])))**2),'k')
# # plt.show()
#
# Npaths = 500
# future_spots = path.many_antithetic_lognormal_paths(Npaths,S0,0,times[-1],r_param,vol_param)
#
# print(future_spots)
#
# plt.hist(np.log(future_spots)-np.log(S0)-(r0-0.5*sig0**2)*times[-1],bins=50,alpha=0.3,color='b')
# plt.show()

# ------------------------------------------------------------------------------

# test generator

# gen_norm = generator.normal()
# gen_norm_antith = generator.antithetic(gen_norm)
# #
# # print(gen_norm.get_samples(10))
# # print(gen_norm_antith.get_samples(10))
#
# expiry    = 1.0/12.0
# Nt        = 200
# times     = np.linspace(0,expiry,Nt)
#
# r0   = 0.024
# sig0 = 0.12
# r_param   = prmtr.SimpleParam(r0)
# vol_param = prmtr.SimpleParam(sig0)
#
# S0 = 100.0
#
# Npaths = 1500
# future_spots = path.many_paths(Npaths,S0,gen_norm_antith,0.0,times[-1],r_param,vol_param)
#
# #print(future_spots-S0)
#
# vals,bins,ppp = plt.hist(np.log(future_spots)-np.log(S0)-(r0-0.5*sig0**2)*times[-1],bins=100,alpha=0.3,color='b')
# x_ax = bins[:-1]+0.5*(bins[1]-bins[0])
# plt.plot(x_ax, np.mean(vals[len(vals)//2-5:len(vals)//2+5])*np.exp(-0.5*(x_ax/(sig0*np.sqrt(times[-1])))**2),'k')
# plt.show()

# ------------------------------------------------------------------------------

expiry    = 1.0/12.0

strike = 50.0

r0   = 0.094
sig0 = 0.12
r_param   = prmtr.SimpleParam(r0)
vol_param = prmtr.SimpleParam(sig0)

payoff = payoff.CallPayOff(strike)
#payoff = payoff.DoubleDigitalPayOff(strike-5.0,strike+5.0)
option = option.VanillaOption(payoff,expiry)

gen_norm = generator.normal()
gen_norm_antith = generator.antithetic(gen_norm)

spots = np.linspace(40.0,70.0,30)

mc_pricer = montecarlo.SAMonteCarlo(option,gen_norm_antith)
mc_prices = np.zeros_like(spots)
for s_idx,s in enumerate(spots):
    print('spot',s)
    mc_prices[s_idx]  = mc_pricer.solve_price(s,r_param,vol_param)
    print(mc_pricer.get_iteration_count())
    mc_pricer.reset()
    #print(mc_prices[s_idx])

# print('monte carlo price = ', mc_price)
# print(mc_pricer.get_iteration_count())

bs_prices = analytics.black_scholes_call_price(spots,strike,expiry,r0,sig0)

bcs = bspde.BoundaryConditions1D()
bspricer = bspde.BlackScholesSingleAssetPricer(option,r0,sig0,bcs)

pde_prices   = bspricer.solve(spots)

#plt.plot(spots,bs_prices,'k',label='Analytic')
plt.fill_between(spots,bs_prices,facecolor='b',alpha=0.4,label='Analytic')
plt.plot(spots,pde_prices,'k',label='PDE')
plt.plot(spots,mc_prices,'r-.',label='Monte Carlo')
plt.plot([strike, strike],[0,1.2*np.amax(bs_prices)],'b',label='strike')
plt.legend(frameon=False)
plt.xlabel('spot')
plt.ylabel('option price')
plt.title(str(option))

plt.xlim([spots[0], spots[-1]])
plt.ylim([0,1.2*np.amax(bs_prices)])

plt.show()

pde_error = np.abs(pde_prices-bs_prices)/bs_prices
mc_error  = np.abs(mc_prices-bs_prices)/bs_prices
plt.semilogy(spots,pde_error,'k',label='PDE')
plt.semilogy(spots,mc_error,'r-.',label='Monte')
plt.plot([strike, strike],[1e-9,1],'b',label='strike')
plt.legend(frameon=False)
plt.xlabel('spot')
plt.ylabel('|price - analytic|/analytic')
plt.title(str(option))

plt.xlim([spots[0], spots[-1]])
plt.ylim([1e-9, 1])

plt.show()
