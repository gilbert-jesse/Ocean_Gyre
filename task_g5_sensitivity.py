import numpy as np
import matplotlib.pyplot as plt

from parameters import L, H, g, gamma, rho, f0, beta, tau0
from diagnostics import gaussian_eta_bump


def run_gyre_model_g5(
    Nx=100,
    Ny=100,
    days=30,
    cfl_factor=0.5,
    lower_dt=10.0,
    eta_init=None,
    u_init=None,
    v_init=None,
    stochastic_amp=0.0,
    seed=0,
):
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

    if eta_init is not None:
        eta = eta_init.copy()

    if u_init is not None:
        u = u_init.copy()

    if v_init is not None:
        v = v_init.copy()

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

    nt = int(days * 24 * 3600 / (2 * dt))

    rng = np.random.default_rng(seed)

    def apply_bc(u, v):
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

    def total_energy(u, v, eta):
        eta_anom = eta - np.mean(eta)
        E_u = 0.5 * rho * H * np.sum(u**2) * dx * dy
        E_v = 0.5 * rho * H * np.sum(v**2) * dx * dy
        E_eta = 0.5 * rho * g * np.sum(eta_anom**2) * dx * dy
        return E_u + E_v + E_eta

    energy = np.empty(nt)
    time_days = np.empty(nt)

    for n in range(nt):
        # u-first
        u, v = apply_bc(u, v)
        eta_new = eta - H * dt * divergence(u, v)

        u_new = (
            u
            + f_u * dt * interp_v_to_u(v)
            - g * dt * grad_eta_u(eta_new)
            - gamma * dt * u
            + (tau_x_u / (rho * H)) * dt
        )

        u_new, v = apply_bc(u_new, v)

        v_new = (
            v
            - f_v * dt * interp_u_to_v(u_new)
            - g * dt * grad_eta_v(eta_new)
            - gamma * dt * v
            + (tau_y_v / (rho * H)) * dt
        )

        eta, u, v = eta_new, u_new, v_new
        u, v = apply_bc(u, v)

        # v-first
        eta_new = eta - H * dt * divergence(u, v)

        v_new = (
            v
            - f_v * dt * interp_u_to_v(u)
            - g * dt * grad_eta_v(eta_new)
            - gamma * dt * v
            + (tau_y_v / (rho * H)) * dt
        )

        u, v_new = apply_bc(u, v_new)

        u_new = (
            u
            + f_u * dt * interp_v_to_u(v_new)
            - g * dt * grad_eta_u(eta_new)
            - gamma * dt * u
            + (tau_x_u / (rho * H)) * dt
        )

        eta, u, v = eta_new, u_new, v_new
        u, v = apply_bc(u, v)

        # continuous stochastic perturbation
        if stochastic_amp > 0.0:
            eta += stochastic_amp * rng.standard_normal(eta.shape)
            eta -= np.mean(eta)

        energy[n] = total_energy(u, v, eta)
        time_days[n] = (n + 1) * 2.0 * dt / 86400.0

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


def run_g5_experiments(Nx=100, Ny=100, days=30, dt=30.0):
    control = run_gyre_model_g5(
        Nx=Nx,
        Ny=Ny,
        days=days,
        lower_dt=dt
    )

    amps = [0.001, 0.01, 0.05]
    ic_results = {}

    for amp in amps:
        eta0 = gaussian_eta_bump(Nx, Ny, amp=amp)

        ic_results[amp] = run_gyre_model_g5(
            Nx=Nx,
            Ny=Ny,
            days=days,
            lower_dt=dt,
            eta_init=eta0
        )

    eta_a = gaussian_eta_bump(Nx, Ny, amp=0.01)

    rng = np.random.default_rng(123)
    eta_b = eta_a + 1.0e-6 * rng.standard_normal(eta_a.shape)
    eta_b -= np.mean(eta_b)

    res_a = run_gyre_model_g5(
        Nx=Nx,
        Ny=Ny,
        days=days,
        lower_dt=dt,
        eta_init=eta_a
    )

    res_b = run_gyre_model_g5(
        Nx=Nx,
        Ny=Ny,
        days=days,
        lower_dt=dt,
        eta_init=eta_b
    )

    eta_diff_nearby = res_a["eta"] - res_b["eta"]

    ensemble_eta = []

    for seed in range(10):
        res = run_gyre_model_g5(
            Nx=Nx,
            Ny=Ny,
            days=days,
            lower_dt=dt,
            stochastic_amp=1.0e-5,
            seed=seed
        )

        ensemble_eta.append(res["eta"])

    ensemble_eta = np.array(ensemble_eta)
    eta_mean = np.mean(ensemble_eta, axis=0)
    eta_spread = np.std(ensemble_eta, axis=0)

    diagnostics = {
        "control_max_eta": np.max(np.abs(control["eta"])),
        "nearby_ic_max_difference": np.max(np.abs(eta_diff_nearby)),
        "ensemble_mean_min": eta_mean.min(),
        "ensemble_mean_max": eta_mean.max(),
        "ensemble_spread_max": eta_spread.max(),
    }

    for amp in amps:
        diagnostics[f"amp_{amp}_max_eta"] = np.max(np.abs(ic_results[amp]["eta"]))
        diagnostics[f"amp_{amp}_energy_final"] = ic_results[amp]["energy"][-1]
        diagnostics[f"amp_{amp}_difference_from_control"] = np.max(
            np.abs(ic_results[amp]["eta"] - control["eta"])
        )

    return {
        "control": control,
        "ic_results": ic_results,
        "res_a": res_a,
        "res_b": res_b,
        "eta_diff_nearby": eta_diff_nearby,
        "eta_mean": eta_mean,
        "eta_spread": eta_spread,
        "diagnostics": diagnostics,
    }


def plot_g5_sensitivity(g5):
    control = g5["control"]
    ic_results = g5["ic_results"]
    eta_diff_nearby = g5["eta_diff_nearby"]
    eta_spread = g5["eta_spread"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 5))

    plots = [
        (control["eta"], "Control"),
        (ic_results[0.001]["eta"], "IC amplitude = 0.001"),
        (ic_results[0.01]["eta"], "IC amplitude = 0.01"),
        (ic_results[0.05]["eta"], "IC amplitude = 0.05"),
        (eta_diff_nearby, "Nearby IC difference"),
        (eta_spread, "Stochastic ensemble spread"),
    ]

    for ax, (field, title) in zip(axes.flat, plots):
        cf = ax.contourf(
            control["X_eta"] / 1000,
            control["Y_eta"] / 1000,
            field,
            levels=20
        )

        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"$\eta$ (m)")

    plt.suptitle(
        "Sensitivity to Initial Conditions and Stochastic Perturbations",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(
        "Sensitivity_to_initial_conditions.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    g5 = run_g5_experiments()
    print(g5["diagnostics"])
    plot_g5_sensitivity(g5)