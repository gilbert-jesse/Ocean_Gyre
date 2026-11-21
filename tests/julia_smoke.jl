# Evaluate the unchanged solver definitions without executing the original
# top-level 30-day benchmarks or overwriting their four CSV exports.
using Test
source = read(ARGS[1], String)
sections = split(source, "\n@time "; limit=2)
length(sections) == 2 || error("Original Julia benchmark boundary changed; review this harness")
include_string(Main, first(sections), ARGS[1])

@testset "Julia gyre: short rectangular-grid integration" begin
    eta, u, v, energy, time_days, dt_cfl = run_gyre_model_julia_fast(
        Nx=8, Ny=6, days=600.0/86400.0, dt=10.0)
    @test size(eta) == (6, 8)
    @test size(u) == (6, 9)
    @test size(v) == (7, 8)
    @test all(isfinite, eta) && all(isfinite, u) && all(isfinite, v)
    @test all(isfinite, energy) && all(energy .>= 0)
    @test all(u[:, [1, end]] .== 0)
    @test all(v[[1, end], :] .== 0)
    @test abs(sum(eta) / length(eta)) < 1e-13
    @test length(energy) == 30
    @test time_days[end] ≈ 600.0/86400.0
    @test dt_cfl > 10.0
    eta2, u2, v2, energy2, _, _ = run_gyre_model_julia_fast(
        Nx=8, Ny=6, days=600.0/86400.0, dt=10.0)
    @test eta == eta2 && u == u2 && v == v2 && energy == energy2
    # Export only to pytest's isolated temporary directory for Python comparison.
    for (name, values) in (("eta", eta), ("u", u), ("v", v),
                           ("energy", energy), ("time_days", time_days))
        writedlm(joinpath(ARGS[2], name * ".csv"), values, ',')
    end
end
println("Julia ", VERSION, "; DelimitedFiles ", Base.pkgversion(DelimitedFiles))
