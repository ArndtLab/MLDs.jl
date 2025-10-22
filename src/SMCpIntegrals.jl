module SMCpIntegrals

using FastGaussQuadrature
using LinearAlgebra
using Base.Threads

export prordn!, firstorder

function Nt(t::Real, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real})
    pnt = 1
    while pnt < length(times) && times[pnt+1] <= t
        pnt += 1
    end
    return sizes[pnt]
end

function cumcr(t1::Real, t2::Real, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real})
    @assert t2 >= t1
    @assert t1 >= 0
    pnt = 1
    while pnt < length(times) && times[pnt+1] <= t1
        pnt += 1
    end
    c = 0.
    while pnt < length(times) && times[pnt] < t2
        gens = min(t2, times[pnt+1]) - max(t1, times[pnt])
        c += gens / sizes[pnt]
        pnt += 1
    end
    if times[pnt] < t2
        gens = t2 - max(t1, times[pnt])
        c += gens / sizes[pnt]
        pnt += 1
    end
    return c
end

function firstorder(r::Real, rate::Real, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real})
    s = 0.
    cum = 0.
    pnt = 1
    while pnt < length(times)
        pnt += 1
        t = times[pnt]
        aem = 1/2sizes[pnt-1]
        aep = 1/2sizes[pnt]
        cum += (t - times[pnt-1]) / 2sizes[pnt-1]
        s += (
            t^2*(aep/(aep+2rate*r) - aem/(aem+2rate*r)) 
            + 2t*(aep/(aep+2rate*r)^2 - aem/(aem+2rate*r)^2) 
            + 2*(aep/(aep+2rate*r)^3 - aem/(aem+2rate*r)^3)
        ) * exp(-2rate * r * t - cum)
    end
    s += 8 * sizes[1]^2 / (1 + 4*sizes[1] * rate * r)^3
    return s * 2 * rate
end

function pt(t::Real, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real})
    return exp(-cumcr(0, t, times, sizes)/2) * t / (2 * Nt(t, times, sizes)) # / <t> simplifies when multiplying by number of segments
end

function margrecomb(t::Real, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real})
    s = 0.
    pnt = 1
    while pnt < length(times) && times[pnt+1] < t
        s += (sizes[pnt] - sizes[pnt+1]) * exp(-cumcr(times[pnt+1], t, times, sizes))
        pnt += 1
    end
    return s
end

function ptt(ti::Real, tj::Real, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real}) # ti given tj
    if ti == tj
        return ti - margrecomb(ti, times, sizes) - Nt(ti, times, sizes) + Nt(0, times, sizes) * exp(-cumcr(0, ti, times, sizes)) #/ 2tj
    elseif ti < tj
        return 1 + margrecomb(ti, times, sizes)/Nt(ti, times, sizes) - Nt(0, times, sizes) / Nt(ti, times, sizes) * exp(-cumcr(0, ti, times, sizes)) #/ 2tj
    else
        return exp(-cumcr(tj, ti, times, sizes)/2) * (Nt(tj, times, sizes) + margrecomb(tj, times, sizes) - Nt(0, times, sizes) * exp(-cumcr(0, tj, times, sizes))) / Nt(ti, times, sizes) #/ 2tj
    end
end

function tolaguerre(z, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real})
    epoch = 1
    ce = 0
    ae = 1/2sizes[epoch]
    t = (z - ce)/ae
    while epoch < length(times) && t > times[epoch+1]
        epoch += 1
        ce += (times[epoch] - times[epoch-1]) * ae
        ae = 1/2sizes[epoch]
        t = (z - ce + ae*times[epoch])/ae
    end
    return t, 1/ae
end

function tolegendre(z, times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real})
    y = -1 - 2/(z-1)
    dy = 2/(z-1)^2
    t, dt = tolaguerre(y, times, sizes)
    return t, dt * dy
end

function prordn!(res::Array{<:Real, 2}, jprt::Array{<:Real, 2}, temp::Array{<:Real, 2}, qtt::Array{<:Real, 2},
    rs::Vector{<:Real}, edges::Vector{<:Real}, rate::Real, 
    times::AbstractVector{<:Real}, sizes::AbstractVector{<:Real}
)
    @assert size(res, 1) == length(rs) "results first dimension must match rs length"
    @assert size(jprt, 2) == length(rs) "jprt second dimension must match rs length"
    @assert size(temp, 1) == size(jprt, 2) "temp first dimension must match jprt second dimension"
    @assert size(temp, 2) == size(jprt, 1) "temp second dimension must match jprt first dimension"

    order = size(res, 2)
    n_dt = size(jprt, 1)
    zs, wt = gausslegendre(n_dt)
    res .= 0
    jprt .= 0
    temp .= 0
    ts = similar(zs)
    dts = similar(zs)

    @threads for i in eachindex(rs)
        for j in eachindex(zs)
            t, dt = tolegendre(zs[j], times, sizes)
            ts[j] = t
            dts[j] = dt
            q = pt(t, times, sizes)
            p = rate * exp(-2rate * rs[i] * t)
            jprt[j,i] = p * q
        end
        res[i,1] = firstorder(rs[i], rate, times, sizes)
    end
    @threads for i in 1:n_dt
        for j in 1:n_dt
            w = i == j ? 1. : wt[j] * dts[j]
            p = max(ptt(ts[i], ts[j], times, sizes), 0.)
            qtt[j,i] = p * w
        end
    end
    for o in 1:order-1
        @threads for i in eachindex(rs)
            # transition t integral
            for j in 1:n_dt
                s = 0.
                for k in 1:n_dt
                    s += jprt[k,i] * qtt[k,j]
                end
                temp[i,j] = s
            end
        end # I am modifying jprt in the end, so need to finish all temp first
        @threads for i in eachindex(rs)
            s2 = 0.
            for j in 1:n_dt
                # convolution r integral
                s = 0.
                for k in 1:i-1
                    w = edges[k+1] - edges[k]
                    # if w <= 1
                    #     s += temp[k,j] * exp(-2rate * (edges[i+1]-edges[k+1]) * ts[j]) * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                    # else
                        s += temp[k,j] * exp(-2rate * (rs[i]-edges[k+1]) * ts[j]) * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                    # end
                end
                w = edges[i+1] - edges[i]
                if w <= 1
                    s += temp[i,j] * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                else
                    w = rs[i] - edges[i]
                    s += temp[i,j] * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                end
                jprt[j,i] = s
                # terminal t integral part
                # 2t factor from p(r|t) here does not simplify
                s2 += jprt[j,i] * 2 * ts[j] * wt[j] * dts[j]
            end
            res[i,o+1] = s2
        end
    end
    return nothing
end

end