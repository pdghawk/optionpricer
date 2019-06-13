optionpricer - A Package for Pricing Financial Options and Derivatives
------------------------------------------------------------------------

This is a simple option pricing package, that is for pricing financial derivatives.

The package can currently price Vanilla single asset and multi-asset options, using
both Monte Carlo and PDE methods.
Payoffs and Option types can be easily added to the payoff and option modules
respectively if they are not already provided.

There are plans for extension of the package to exotic options, and the modular
design of the package should aid this extension.

The following books were used in the design of the package:

 - C++ Designs Patterns and Derivatives Pricing, M.S. Joshi (Cambridge University Press)
 - Guide to Scientific Computing in C++, Pitt-Francis and Whitely (Springer)

The concepts were translated into python; since I have previously designed some
larger C++ projects during my roles as a scientific researcher, with similar
differential equation solving routines etc in C++ and Boost, and so wanted to
write this package in python for a change. Further, I'm anticipating the potential
for leveraging Tensorflow's GPU libraries, via python, for both PDE and Monte Carlo
pricing methods for high-dimensional multi-asset options where GPU's could be very
useful.

See /notebooks for useful examples of usage.

The directory structure is:

::

  ├── LICENSE
  ├── README.rst
  ├── notebooks
  │   ├── BlackScholesPdeExample.ipynb
  │   ├── MonteCarloVsPdeExample.ipynb
  │   └── MultiAssetExchangeOptions.ipynb
  ├── optionpricer
  │   ├── __init__.py
  │   ├── analytics.py
  │   ├── bspde.py
  │   ├── error.py
  │   ├── generator.py
  │   ├── montecarlo.py
  │   ├── option.py
  │   ├── parameter.py
  │   ├── path.py
  │   └── payoff.py
  ├── requirements.txt
