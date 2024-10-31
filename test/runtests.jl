using MLDs
using Test

@testset "Stationary" begin
    TN = [250_000_000, 10_000]
    μ = 1e-8
    rs = logrange(1,1e6,200)
    y = map(x->hid(TN, μ, x), rs)
end

@testset "Bottleneck" begin
    TN = [250_000_000, 10_000, 2000, 2000, 5000, 10_000]
    μ = 1e-8
    rs = logrange(1,1e6,200)
    y = map(x->hid(TN, μ, x), rs)
end
