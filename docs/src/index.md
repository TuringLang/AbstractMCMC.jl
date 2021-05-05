# AbstractMCMC.jl

*Abstract types and interfaces for Markov chain Monte Carlo methods.*

AbstractMCMC defines an interface for sampling and combining Markov chains.
It comes with a default sampling algorithm that provides support of progress
bars, parallel sampling (multithreaded and multicore), and user-provided callbacks
out of the box. Typically developers only have to define the sampling step
of their inference method in an iterator-like fashion to make use of this
functionality. Additionally, the package defines an iterator and a transducer
for sampling Markov chains based on the interface.
