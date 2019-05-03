
This is a simple option pricing package, that is for pricing financial derivatives.

The package is currently restricted to pricing Vanilla single asset options.
There are plans for extension of the package to exotic and multi-asset options,
and the modular design of the package should aid this extension.

The following books were used in the design of the package:
 - C++ Designs Patterns and Derivatives Pricing, M.S. Joshi (Cambridge University Press)
 - Guide to Scientific Computing in C++, Pitt-Francis and Whitely (Springer)

Although, the ideas were translated into python. I have previously designed some
larger C++ projects during my roles as a scientific researcher, with similar
differential equation solving routines etc in C++, and so wanted to write this
package in python for a change. I'm also anticipating the potential for leveraging
Tensorflow's GPU libraries for both PDE and Monte Carlo pricing methods for
high-dimensional multi-asset options where GPU's could be very useful.
