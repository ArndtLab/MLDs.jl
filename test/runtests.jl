using MLDs
using Test

@testset "Compare to Mathematica" begin
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

        y1m = MLDs.hidm(L, N0, mu, r)
        y2m = MLDs.hidm(L, N0, T1, N1, mu, r)
        y21m = MLDs.hidm(L, N0, T1, N0, mu, r)
        y3m = MLDs.hidm(L, N0, T1, N1, T2, N2, mu, r)
        y31m = MLDs.hidm(L, N0, T1, N0, T2, N0, mu, r)

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
