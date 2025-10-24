using MLDs
using MLDs.SMCpIntegrals: Nt, cumcr, pt, ptt
using Test

include("mathematica-derived.jl")

@testset "Compare MLD to Mathematica" begin
    st = map(1:1000) do i
        L = rand(1.0e6:1.0e9)
        N0 = N1 = N2 = -1.0
        while true
            N0 = rand(1.0e3:1.0e5)
            N1 = rand(1.0e3:1.0e5)
            N2 = rand(1.0e3:1.0e5)
            if (N0 < N1 > N2) || (N0 > N1 < N2)
                break
            end
        end
        T1 = rand(1.0e1:1.0e3)
        T2 = rand(1.0e1:1.0e3)
        mu = rand(1.0e-9:1.0e-8)
        r = rand(1:1_000_000)

        y1 = hid([L, N0], mu, r)
        y2 = hid([L, N0, T1, N1], mu, r)
        y3 = hid([L, N0, T1, N1, T2, N2], mu, r)

        y1m = hidm(L, N0, mu, r)
        y2m = hidm(L, N0, T1, N1, mu, r)
        y21m = hidm(L, N0, T1, N0, mu, r)
        y3m = hidm(L, N0, T1, N1, T2, N2, mu, r)
        y31m = hidm(L, N0, T1, N0, T2, N0, mu, r)

        @test abs(y1 - y1m) < 1.0e-10
        @test abs(y1 - y21m) < 1.0e-10
        @test abs(y1 - y31m) < 1.0e-10
        @test abs(y2 - y2m) < 1.0e-10
        @test abs(y3 - y3m) < 1.0e-10

        (;
            L, N0, N1, N2, T1, T2, mu, r, y1, y2, y3, y1m, y2m, y3m,
            d1 = y1 - y1m, d2 = y2 - y2m, d3 = y3 - y3m,
            d1r = abs(y1 - y1m) / y1m, d2r = abs(y2 - y2m) / y2m, d3r = abs(y3 - y3m) / y3m,
        )
    end
    @show maximum(abs.(getindex.(st, :d1)))
    @show maximum(abs.(getindex.(st, :d2)))
    @show maximum(abs.(getindex.(st, :d3)))
end

@testset "Coalescent stationary" begin
    N0 = 1_000
    ts = rand(1:40*N0, 10)
    for t in ts
        @test MLDs.coalescent(t, [0, N0]) ≈ exp(-t / (2 * N0)) / (2 * N0)
    end
end

@testset "Extant basepairs stationary" begin
    N0 = 1_000
    L = 3_000_000_000
    ts = rand(1:40*N0, 10)
    for t in ts
        @test MLDs.extbps(t, [L, N0]) ≈ round(L * exp(-t / (2 * N0)))
    end
end

@testset "Lineages stationary" begin
    N0 = 1_000
    ts = rand(1:40*N0, 10)
    for t in ts
        @test abs(MLDs.lineages(t, 1, [1, N0]; k = 1.) - 2 * t * exp(-2 * t - 1/2N0) / 2N0) < eps(Float64)
    end
end

@testset "hid vs fisrt order" begin
    N0 = 1_000
    L = 3_000_000_000
    mu = 1.25e-8
    for r in rand(1:1_000_000, 10)
        y1 = hid([L, N0], mu, r)
        y2 = firstorder(r, mu, [L, N0]) * 2 * mu * L
        y3 = laplacekingman(r, mu, [L, N0])[1]
        @test abs(y1 - y2) < 1.0e-10
        @test y2 ≈ y3
    end
end

@testset "aux functions SMCpIntegrals" begin
    TN = [3_000_000_000.0, 2000.0, 1000.0, 1000.0, 1000.0, 3000.0]

    prev = 0
    for t in sort(rand(0.0:5000.0, 20))
        @test Nt(t, TN) > 0
        @test cumcr(0.0, t, TN) >= 0
        @test cumcr(0.0, t, TN) >= prev
        prev = cumcr(0.0, t, TN)
        @test pt(t, TN) >= 0
        @test ptt(t, 100, TN) >= 0
    end
end

@testset "mld smcp runs" begin
    TN = [3_000_000_000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 10000]
    rs = collect(1:100)
    ed = collect(1:101)
    mu = 1e-8
    rho = 1e-8
    y = zeros(length(rs))
    bag = IntegralArrays(10, 100, length(rs))
    mldsmcp!(y, 1:10, bag, rs, ed, mu, rho, TN)
    @test all(y .> 0)
end