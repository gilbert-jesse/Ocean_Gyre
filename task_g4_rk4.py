import time
import numpy as np
from parameters import L, H, g, gamma, rho, f0, beta, tau0
from diagnostics import run_gyre_model
from plotting import plot_eta_snapshot


def run_gyre_model_rk4(Nx=51, Ny=51, days=200, cfl_factor=0.5, lower_dt=None):
    dx = L / Nx
    dy = L / Ny

    x_eta = (np.arange(Nx) + 0.5) * dx
    y_eta = (np.arange(Ny) + 0.5) * dy

    x_u = np.arange(Nx + 1) * dx
    y_u = y_eta.copy()

    x_v = x_eta.copy()
    y_v = np.arange(Ny + 1) * dy

    X_eta, Y_eta = np.meshgrid(x_eta, y_eta, indexing="xy")
    X_u, Y_u = np.meshgrid(x_u, y_u, indexing="xy")
    X_v, Y_v = np.meshgrid(x_v, y_v, indexing="xy")

    eta = np.zeros((Ny, Nx))
    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    tau_x_u = -tau0 * np.cos(np.pi * Y_u / L)
    tau_y_v = np.zeros_like(Y_v)

    f_u = f0 + beta * Y_u
    f_v = f0 + beta * Y_v

    c = np.sqrt(g * H)
    dt_cfl = 1.0 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))

    if lower_dt is None:
        dt = cfl_factor * dt_cfl
    else:
        dt = lower_dt

    nt = int(days * 24 * 3600 / dt)

    def apply_bc(u, v):
        u = u.copy()
        v = v.copy()
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        v[0, :] = 0.0
        v[-1, :] = 0.0
        return u, v

    def divergence(u, v):
        return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy

    def grad_eta_u(eta):
        out = np.zeros((Ny, Nx + 1))
        out[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / dx
        return out

    def grad_eta_v(eta):
        out = np.zeros((Ny + 1, Nx))
        out[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / dy
        return out

    def interp_v_to_u(v):
        out = np.zeros((Ny, Nx + 1))
        out[:, 1:-1] = 0.25 * (
            v[:-1, :-1] + v[:-1, 1:] +
            v[1:, :-1] + v[1:, 1:]
        )
        return out

    def interp_u_to_v(u):
        out = np.zeros((Ny + 1, Nx))
        out[1:-1, :] = 0.25 * (
            u[:-1, :-1] + u[:-1, 1:] +
            u[1:, :-1] + u[1:, 1:]
        )
        return out

    def rhs(u, v, eta):
        u, v = apply_bc(u, v)

        deta_dt = -H * divergence(u, v)

        du_dt = (
            f_u * interp_v_to_u(v)
            - g * grad_eta_u(eta)
            - gamma * u
            + tau_x_u / (rho * H)
        )

        dv_dt = (
            -f_v * interp_u_to_v(u)
            - g * grad_eta_v(eta)
            - gamma * v
            + tau_y_v / (rho * H)
        )

        du_dt, dv_dt = apply_bc(du_dt, dv_dt)

        return du_dt, dv_dt, deta_dt

    def rk4_step(u, v, eta):
        k1u, k1v, k1e = rhs(u, v, eta)

        k2u, k2v, k2e = rhs(
            u + 0.5 * dt * k1u,
            v + 0.5 * dt * k1v,
            eta + 0.5 * dt * k1e
        )

        k3u, k3v, k3e = rhs(
            u + 0.5 * dt * k2u,
            v + 0.5 * dt * k2v,
            eta + 0.5 * dt * k2e
        )

        k4u, k4v, k4e = rhs(
            u + dt * k3u,
            v + dt * k3v,
            eta + dt * k3e
        )

        u_new = u + (dt / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
        v_new = v + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        eta_new = eta + (dt / 6.0) * (k1e + 2*k2e + 2*k3e + k4e)

        u_new, v_new = apply_bc(u_new, v_new)

        return u_new, v_new, eta_new

    def total_energy(u, v, eta):
        eta_anom = eta - np.mean(eta)
        E_u = 0.5 * rho * H * np.sum(u**2) * dx * dy
        E_v = 0.5 * rho * H * np.sum(v**2) * dx * dy
        E_eta = 0.5 * rho * g * np.sum(eta_anom**2) * dx * dy
        return E_u + E_v + E_eta

    energy = np.empty(nt)
    time_days = np.empty(nt)

    for n in range(nt):
        u, v, eta = rk4_step(u, v, eta)

        energy[n] = total_energy(u, v, eta)
        time_days[n] = (n + 1) * dt / 86400.0

    return {
        "Nx": Nx, "Ny": Ny,
        "dx": dx, "dy": dy,
        "dt": dt,
        "dt_cfl": dt_cfl,
        "x_eta": x_eta, "y_eta": y_eta,
        "x_u": x_u, "y_u": y_u,
        "x_v": x_v, "y_v": y_v,
        "X_eta": X_eta, "Y_eta": Y_eta,
        "X_u": X_u, "Y_u": Y_u,
        "X_v": X_v, "Y_v": Y_v,
        "eta": eta,
        "u": u,
        "v": v,
        "energy": energy,
        "time_days": time_days,
    }


def compare_rk4_fb(Nx=100, Ny=100, days=1, dt=30.0):
    res_rk4 = run_gyre_model_rk4(Nx=Nx, Ny=Ny, days=days, lower_dt=dt)
    res_fb = run_gyre_model(Nx=Nx, Ny=Ny, days=days, lower_dt=dt)

    return {
        "res_rk4": res_rk4,
        "res_fb": res_fb,
        "eta_diff": np.max(np.abs(res_fb["eta"] - res_rk4["eta"])),
        "u_diff": np.max(np.abs(res_fb["u"] - res_rk4["u"])),
        "v_diff": np.max(np.abs(res_fb["v"] - res_rk4["v"])),
    }


def stability_sweep(method="rk4", Nx=100, Ny=100, days=30):
    dt_values = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]
    output = []

    for dt_test in dt_values:
        try:
            if method == "rk4":
                res = run_gyre_model_rk4(
                    Nx=Nx,
                    Ny=Ny,
                    days=days,
                    lower_dt=dt_test
                )
            else:
                res = run_gyre_model(
                    Nx=Nx,
                    Ny=Ny,
                    days=days,
                    lower_dt=dt_test
                )

            eta_max = np.max(np.abs(res["eta"]))
            finite = np.all(np.isfinite(res["eta"]))

            output.append({
                "dt": dt_test,
                "finite": finite,
                "max_eta": eta_max,
            })

        except Exception:
            output.append({
                "dt": dt_test,
                "finite": False,
                "max_eta": np.nan,
            })

    return output


def benchmark_rk4_fb(Nx=100, Ny=100):
    start = time.time()
    res_fb = run_gyre_model(Nx=Nx, Ny=Ny, days=30, lower_dt=60.0)
    fb_time = time.time() - start

    start = time.time()
    res_rk4 = run_gyre_model_rk4(Nx=Nx, Ny=Ny, days=30, lower_dt=100.0)
    rk4_time = time.time() - start

    return {
        "res_fb": res_fb,
        "res_rk4": res_rk4,
        "fb_time": fb_time,
        "rk4_time": rk4_time,
        "runtime_ratio": rk4_time / fb_time,
    }


if __name__ == "__main__":
    comp = compare_rk4_fb()
    print("max eta difference:", comp["eta_diff"])
    print("max u difference:", comp["u_diff"])
    print("max v difference:", comp["v_diff"])

    rk4_stability = stability_sweep(method="rk4")
    fb_stability = stability_sweep(method="fb")

    print("RK4 stability:", rk4_stability)
    print("FB stability:", fb_stability)

    bench = benchmark_rk4_fb()
    print("FB runtime:", bench["fb_time"])
    print("RK4 runtime:", bench["rk4_time"])
    print("RK4/FB runtime ratio:", bench["runtime_ratio"])

    plot_eta_snapshot(
        comp["res_rk4"]["eta"],
        L,
        days=1,
        time=30.0,
        title="RK4 eta after 1 day, dt = 30 s"
    )