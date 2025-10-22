module MLDs

using LinearAlgebra
using ForwardDiff

include("CoalescentBase.jl")
using .CoalescentBase

include("SMCpIntegrals.jl")
using .SMCpIntegrals

export 
    hid, hid_integral, firstorder, laplacekingman, mldsmcp, mldsmcp!

# Computing

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

function mldsmcp(rs::Vector{<:Real}, edges::Vector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}, order::Int, ndt::Int
)
    res = Array{Float64}(undef, length(rs), order)
    jprt = Array{Float64}(undef, ndt, length(rs))
    temp = Array{Float64}(undef, length(rs), ndt)
    qtt = Array{Float64}(undef, ndt, ndt)
    yth = mldsmcp!(res, jprt, temp, qtt, rs, edges, mu, rho, TN)
    return yth
end

function mldsmcp!(res::AbstractMatrix{<:Real}, jprt::AbstractMatrix{<:Real}, temp::AbstractMatrix{<:Real}, qtt::AbstractMatrix{<:Real},
    rs::Vector{<:Real}, edges::Vector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}
)
    ts = ordts(TN)
    ns = ordns(TN)
    prordn!(res, jprt, temp, qtt, rs, edges, mu+rho, ts, ns)
    yth = zeros(length(rs))
    for i in 1:size(res,2)
        yth .+= res[:,i] * 2 * mu * TN[1] * (rho/(mu+rho))^(i-1) * (mu/(mu+rho))
    end
    return yth
end

function laplacekingman(rs::Vector{<:Real}, mu::Real, TN::AbstractVector{<:Real})
    ts = ordts(TN)
    ns = ordns(TN)
    return map(r -> firstorder(r, mu, ts, ns), rs) * 2 * mu * TN[1]
end

end
