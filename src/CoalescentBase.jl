module CoalescentBase
using ForwardDiff

export coalescent, extbps, laplace_n, lineages, cumulative_lineages

"""
    coalescent(t::Int, TN::Vector)

Calculate the probability of coalescence at time `t` generations in
the past.

It is computed for two alleles in the absence of recombinaiton 
and for a demographic scenario encoded in `TN`. The distribution 
of such `t`s is geometric as introduced by Hudson and Kingman.

### References
"""
function coalescent(t::Int, TN::Vector)
    ts = [0;cumsum(reverse(TN[3:2:end-1]))]
    ns = reverse(TN[2:2:end])
    pnt = 1
    c = 0.
    while (pnt < length(ts)) && (ts[pnt] < t)
        gens = ts[pnt+1] >= t ? (t - ts[pnt]) : (ts[pnt+1] - ts[pnt])
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    if ts[pnt] < t
        gens = t - ts[pnt]
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    return exp(-c) / 2ns[pnt-1]
end

"""
    extbps(t::Float64, TN::Vector)

Calculate the the expected number of basepairs that still have to
reach coalescence at time `t` generations in the past. 

The demographic scenario is encoded in `TN`.

### Reference
"""
function extbps(t::Float64, TN::Vector)
    L = Float64(TN[1])
    ts = [0;cumsum(reverse(TN[3:2:end-1]))]
    ns = reverse(TN[2:2:end])
    pnt = 1
    c = 0.
    while (pnt < length(ts)) && (ts[pnt] < t)
        gens = ts[pnt+1] >= t ? (t - ts[pnt]) : (ts[pnt+1] - ts[pnt])
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    if ts[pnt] < t
        gens = t - ts[pnt]
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    return round(L*exp(-c))
end

function laplace_n(TN::Vector, s::Number)
    N = TN[2]
    y = 2 * N^2 / (1 + 2*N*s)
    # stationary solution in first epoch
    for k in 3:2:length(TN) # loop over further epochs
        T  = TN[k]
        Np = TN[k-1]
        N = TN[k+1]
        # step up or down
        gamma = N / Np
        w1 = gamma >= 1 ? gamma^2 - (gamma^2 - 1)/(2 * Np) : gamma^2
        w2 = gamma >= 1 ? (gamma^2 - 1) * Np               : zero(gamma)
        # propagate in time
        v1 = exp((-T/(2*N)) - s*T)
        v2 = (1 - v1) * ((2*N^2) / (1 + 2*N*s))
        # update value
        y = (w1 * y + w2) * v1 + v2
    end
    y
end

function laplace_n(Nv::Vector, Tv::Vector, s::Number)
	# T =        [T1, T2, ...]
	# N = [Nstat, N1, N2, ...]
    Nstat = Nv[1]
    y = 2 * Nstat^2 / (1 + 2*Nstat*s)
    # stationary solution in first epoch
    Np = Nstat
    for (T, N) in zip(Tv, Iterators.drop(Nv, 1)) # loop over further epochs
        # T  = Tv[k-2]
        # Np = Nv[k-1]
        # N =  Nv[k]
        # step up or down
        gamma = N / Np
        w1 = gamma >= 1 ? gamma^2 - (gamma^2 - 1)/(2 * Np) : gamma^2
        w2 = gamma >= 1 ? (gamma^2 - 1) * Np               : zero(gamma)
        # propagate in time
        v1 = exp((-T/(2*N)) - s*T)
        v2 = (1 - v1) * ((2*N^2) / (1 + 2*N*s))
        # update value
        y = (w1 * y + w2) * v1 + v2
        Np = N
    end
    y
end

"""
    lineages(t::Float64, TN::Vector, rho::Float64; k::Int = 0)

Calculate the expected number of genomic segments which are Identical by Descent 
and coalesce at time `t` generations in the past having a genomic length longer
than `k` basepairs. 

The demographic scenario is encoded in `TN` and the recombination rate is `rho`
in unit per bp per generation.
"""
function lineages(t, TN::Vector, rho::Float64; k::Int = 0)
    ts = [0;cumsum(reverse(TN[3:2:end-1]))]
    ns = reverse(TN[2:2:end])
    pnt = 1
    c = 0.
    while (pnt < length(ts)) && (ts[pnt] < t)
        gens = ts[pnt+1] >= t ? (t - ts[pnt]) : (ts[pnt+1] - ts[pnt])
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    if ts[pnt] < t
        gens = t - ts[pnt]
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    return 2TN[1] * rho * t * exp(-2rho * t * k - c) / 2ns[pnt-1]
end

function cumulative_lineages(t, TN::Vector, rho::Float64; k::Int = 0)
    ts = [0;cumsum(reverse(TN[3:2:end-1]))]
    ns = reverse(TN[2:2:end])
    N = TN[end]
    pnt = 1
    c = 0.
    while (pnt < length(ts)) && (ts[pnt] < t)
        gens = ts[pnt+1] >= t ? (t - ts[pnt]) : (ts[pnt+1] - ts[pnt])
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    if ts[pnt] < t
        gens = t - ts[pnt]
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    first_der = ForwardDiff.derivative(s -> laplace_n(TN,s), 2rho*k) / (2 * TN[end]^2)
    ep = findfirst(ts .> t)
    if isnothing(ep)
        TNp = TN[1:2]
    else
        ep -= 1
        remn_ep = sum(TN[end-2ep+1:2:end-1]) - t
        TNp = TN[1:end-2ep+2]
        TNp[end-1] = remn_ep
    end
    first_der_p = ForwardDiff.derivative(s -> laplace_n(TNp,s), 2rho*k) / (2 * TNp[end]^2)
    lap_p = laplace_n(TNp, 2rho*k) / (2 * TNp[end]^2)
    return round(2TN[1] * rho * (exp(-c-2rho*k*t)*(first_der_p - t*lap_p) - first_der))
end

end