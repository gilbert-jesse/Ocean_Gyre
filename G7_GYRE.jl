using Printf
using DelimitedFiles

const L = 1.0e6
const f0 = 1.0e-4
const beta = 1.0e-11
const g = 10.0
const gamma = 1.0e-6
const rho = 1000.0
const H = 1000.0
const tau0 = 0.2

@inline function apply_bc!(u, v, Nx, Ny)
    @inbounds for j in 1:Ny
        u[j, 1] = 0.0
        u[j, Nx + 1] = 0.0
    end

    @inbounds for i in 1:Nx
        v[1, i] = 0.0
        v[Ny + 1, i] = 0.0
    end

    return nothing
end

@inline function total_energy_fast(u, v, eta, dx, dy)
    eta_sum = 0.0

    @inbounds for x in eta
        eta_sum += x
    end

    eta_mean = eta_sum / length(eta)

    su = 0.0
    sv = 0.0
    se = 0.0

    @inbounds for x in u
        su += x * x
    end

    @inbounds for x in v
        sv += x * x
    end

    @inbounds for x in eta
        d = x - eta_mean
        se += d * d
    end

    return 0.5 * rho * (H * (su + sv) + g * se) * dx * dy
end

function run_gyre_model_julia_fast(; Nx=100, Ny=100, days=30.0, dt=10.0)

    dx = L / Nx
    dy = L / Ny

    eta = zeros(Float64, Ny, Nx)
    u = zeros(Float64, Ny, Nx + 1)
    v = zeros(Float64, Ny + 1, Nx)

    tau_x_u = zeros(Float64, Ny, Nx + 1)
    f_u = zeros(Float64, Ny, Nx + 1)
    f_v = zeros(Float64, Ny + 1, Nx)

    @inbounds for j in 1:Ny
        y = (j - 0.5) * dy
        fy = f0 + beta * y
        taux = -tau0 * cos(pi * y / L)

        for i in 1:(Nx + 1)
            tau_x_u[j, i] = taux
            f_u[j, i] = fy
        end
    end

    @inbounds for j in 1:(Ny + 1)
        y = (j - 1.0) * dy
        fy = f0 + beta * y

        for i in 1:Nx
            f_v[j, i] = fy
        end
    end

    c = sqrt(g * H)
    dt_cfl = 1.0 / (c * sqrt(1.0 / dx^2 + 1.0 / dy^2))

    nt = Int(floor(days * 24.0 * 3600.0 / (2.0 * dt)))

    energy = zeros(Float64, nt)
    time_days = zeros(Float64, nt)

    eta_new = similar(eta)
    u_new = similar(u)
    v_new = similar(v)

    apply_bc!(u, v, Nx, Ny)

    for n in 1:nt

        # First half step: u before v
       

        @inbounds for j in 1:Ny
            for i in 1:Nx
                div = (u[j, i + 1] - u[j, i]) / dx +
                      (v[j + 1, i] - v[j, i]) / dy

                eta_new[j, i] = eta[j, i] - H * dt * div
            end
        end

        # u update
        @inbounds for j in 1:Ny
            u_new[j, 1] = 0.0
            u_new[j, Nx + 1] = 0.0

            for i in 2:Nx
                grad_eta = (eta_new[j, i] - eta_new[j, i - 1]) / dx

                v_on_u = 0.25 * (
                    v[j, i - 1] + v[j, i] +
                    v[j + 1, i - 1] + v[j + 1, i]
                )

                u_new[j, i] =
                    u[j, i] +
                    f_u[j, i] * dt * v_on_u -
                    g * dt * grad_eta -
                    gamma * dt * u[j, i] +
                    (tau_x_u[j, i] / (rho * H)) * dt
            end
        end

        # v update
        @inbounds for i in 1:Nx
            v_new[1, i] = 0.0
            v_new[Ny + 1, i] = 0.0
        end

        @inbounds for j in 2:Ny
            for i in 1:Nx
                grad_eta = (eta_new[j, i] - eta_new[j - 1, i]) / dy

                u_on_v = 0.25 * (
                    u_new[j - 1, i] + u_new[j - 1, i + 1] +
                    u_new[j, i] + u_new[j, i + 1]
                )

                v_new[j, i] =
                    v[j, i] -
                    f_v[j, i] * dt * u_on_v -
                    g * dt * grad_eta -
                    gamma * dt * v[j, i]
            end
        end

        eta, eta_new = eta_new, eta
        u, u_new = u_new, u
        v, v_new = v_new, v

        apply_bc!(u, v, Nx, Ny)

        # Second half step: v before u
        

        @inbounds for j in 1:Ny
            for i in 1:Nx
                div = (u[j, i + 1] - u[j, i]) / dx +
                      (v[j + 1, i] - v[j, i]) / dy

                eta_new[j, i] = eta[j, i] - H * dt * div
            end
        end

        # v update
        @inbounds for i in 1:Nx
            v_new[1, i] = 0.0
            v_new[Ny + 1, i] = 0.0
        end

        @inbounds for j in 2:Ny
            for i in 1:Nx
                grad_eta = (eta_new[j, i] - eta_new[j - 1, i]) / dy

                u_on_v = 0.25 * (
                    u[j - 1, i] + u[j - 1, i + 1] +
                    u[j, i] + u[j, i + 1]
                )

                v_new[j, i] =
                    v[j, i] -
                    f_v[j, i] * dt * u_on_v -
                    g * dt * grad_eta -
                    gamma * dt * v[j, i]
            end
        end

        # u update
        @inbounds for j in 1:Ny
            u_new[j, 1] = 0.0
            u_new[j, Nx + 1] = 0.0

            for i in 2:Nx
                grad_eta = (eta_new[j, i] - eta_new[j, i - 1]) / dx

                v_on_u = 0.25 * (
                    v_new[j, i - 1] + v_new[j, i] +
                    v_new[j + 1, i - 1] + v_new[j + 1, i]
                )

                u_new[j, i] =
                    u[j, i] +
                    f_u[j, i] * dt * v_on_u -
                    g * dt * grad_eta -
                    gamma * dt * u[j, i] +
                    (tau_x_u[j, i] / (rho * H)) * dt
            end
        end

        eta, eta_new = eta_new, eta
        u, u_new = u_new, u
        v, v_new = v_new, v

        apply_bc!(u, v, Nx, Ny)

        energy[n] = total_energy_fast(u, v, eta, dx, dy)
        time_days[n] = n * 2.0 * dt / 86400.0
    end

    return eta, u, v, energy, time_days, dt_cfl
end

@time eta, u, v, energy, time_days, dt_cfl =
    run_gyre_model_julia_fast(Nx=100, Ny=100, days=30.0, dt=30.0)

@time eta, u, v, energy, time_days, dt_cfl =
    run_gyre_model_julia_fast(Nx=100, Ny=100, days=30.0, dt=30.0)

writedlm("eta_julia_fast.csv", eta, ",")
writedlm("u_julia_fast.csv", u, ",")
writedlm("v_julia_fast.csv", v, ",")
writedlm("energy_julia_fast.csv", energy, ",")

