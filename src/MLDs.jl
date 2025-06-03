module MLDs

using LinearAlgebra #, Statistics, StatsBase
using ForwardDiff
# using FastGaussQuadrature, Integrals, StaticArrays

include("CoalescentBase.jl")
using .CoalescentBase

export 
    hid, hid_integral

# Computing

include("mathematica-derived.jl")

function secondderivative(f, x)
    dfdx = x -> ForwardDiff.derivative(f, x)
    ForwardDiff.derivative(dfdx, x)
end

function hid(TN::Vector, mu::Float64, r::Number)
	# TN = [L, N0, T1, N1, T2, N2, ...]
	L = TN[1]
	N = TN[end]
	(2*mu^2*L)/(N^2) * secondderivative(s -> laplace_n(TN, s), 2*mu*r)
	# prefactor        pure bliss
end

function hid(L::Number, N::Vector, T::Vector, mu::Float64, r::Number)
	# T = [T1, T2, ...]
	# N = [Nstat, N1, N2, ...]
	Nend = (length(N) == 0 ? Nstat : N[end])
	(2*mu^2*L)/(Nend^2) * secondderivative(s -> laplace_n(N, T, s), 2*mu*r)
	# prefactor           pure bliss
end

function hid_integral(TN::Vector, mu::Float64, r::Number)
    # integral of hid
	# TN = [L, N0, T1, N1, T2, N2, ...]
	L = TN[1]
	N = TN[end]
	(mu*L)/(N^2) * ForwardDiff.derivative(s -> laplace_n(TN, s), 2*mu*r)
	# prefactor    pure bliss
end

function hid_integral(Nv::Vector, Tv::Vector, L::Number, mu::Float64, r::Number)
    # integral of hid
	N = Nv[end]
	(mu*L)/(N^2) * ForwardDiff.derivative(s -> laplace_n(Nv, Tv, s), 2*mu*r)
	# prefactor    pure bliss
end

function truncated_mld(TN::Vector, μ::Number, r::Number, T::Number, sign::Bool, w::Number)
    # Ttot = sum(TN[3:2:end])
    Nsum = sign ? sum([1/2x for x in TN[4:2:end]], init=0) : sum(1/2x for x in TN[2:2:end])
    out = w * exp(-(2μ*r + Nsum) * T) * 
        (hid(TN, μ, r) + 4μ^2*TN[1] * T^2*laplace_n(TN, 2μ*r) - 
        8μ^2*TN[1] * T*ForwardDiff.derivative(s -> laplace_n(TN, s), 2μ*r))
    return out >= 0 ? out : 0
end

function admixed_mld(r::Number, μ::Number, L, N0, N1, N2, N3, T2, T3)
    TN3 = [L, N3]
    TN1 = [L, N1, T3, N3]
    TN2 = [L, N2, T3, N3]
    TN01 = [L, N0, T2, N1, T3, N3]
    TN02 = [L, N0, T2, N2, T3, N3]
    TN00 = [L, N0, T3, N3]

    # admixed = hid(TN3, μ, r) #=truncated_mld(TN3, μ, r, 0, true, 1)=# - truncated_mld(TN3, μ, r, T3, false, 1)
    admixed = truncated_mld(TN1, μ, r, T3, true, (N1/(N1+N2))^2) - truncated_mld(TN1, μ, r, T3+T2, false, (N1/(N1+N2))^2)
    admixed += truncated_mld(TN2, μ, r, T3, true, (N2/(N1+N2))^2) - truncated_mld(TN2, μ, r, T3+T2, false, (N2/(N1+N2))^2)
    admixed += truncated_mld(TN01, μ, r, T3+T2, true, (N1/(N1+N2))^2)
    admixed += truncated_mld(TN02, μ, r, T3+T2, true, (N2/(N1+N2))^2)
    admixed += truncated_mld(TN00, μ, r, T3+T2, true, 2(N1*N2)/(N1+N2)^2)

    return admixed
end

end
