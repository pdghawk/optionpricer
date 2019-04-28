from optionpricer import payoff
import numpy as np

call_payoff = payoff.CallPayOff(10.0)

spots=np.linspace(0.0,20.0,10)
print(call_payoff.get_payoff(spots))
print(call_payoff.get_payoff(spots[0]))
#call_payoff = payoff.CallPayOff([10.0,15.0])
