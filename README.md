# MLDs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArndtLab.github.io/MLDs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArndtLab.github.io/MLDs.jl/dev/)
[![Build Status](https://github.com/ArndtLab/MLDs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArndtLab/MLDs.jl/actions/workflows/CI.yml?query=branch%3Amain)

Functions to compute the Match Length Distribution for different demographic models, under the assumption of mutation rate much greater than recombination rate. Two functions are exported: `hid` and `hid_integral` (indefinite integral, useful for computing the volume of bins in an histogram).

Usage:

```julia
TN = [L, N0, T1, N1, T2, N2]    # genome length L, ancestral population size N0, sequence of epoch duration T_i and size N_i
mu = 1e-8
r = 1.0

hid(TN, mu, r)  # compute the mld at value r
hid_integral(TN, mu, r) # indefinite integral evaluated in r
```