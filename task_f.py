import numpy as np

from parameters import L, H, g, f0, beta, gamma, rho, tau0
from semi_lagrangian import step_semilag_cgrid_trapezoid
from plotting import plot_eta_snapshot


def run_task_f(Nx=100, Ny=100, days=5, dt=10.0, eta_limit=50.0):
    eta = np.zeros((Ny, Nx))
    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    p = {
        "L": L,
        "H": H,
        "g": g,
        "f0": f0,
        "beta": beta,
        "gamma": gamma,
        "rho": rho,
        "tau0": tau0,
        "dt": dt,
    }

    nt = int(days * 86400 / dt)

    stable = True
    stop_step = nt

    for n in range(nt):
        u, v, eta = step_semilag_cgrid_trapezoid(u, v, eta, p)

        finite = (
            np.all(np.isfinite(u))
            and np.all(np.isfinite(v))
            and np.all(np.isfinite(eta))
        )

        if (not finite) or np.nanmax(np.abs(eta)) > eta_limit:
            stable = False
            stop_step = n
            break

    return {
        "Nx": Nx,
        "Ny": Ny,
        "days": days,
        "dt": dt,
        "eta": eta,
        "u": u,
        "v": v,
        "stable": stable,
        "stop_step": stop_step,
        "time_reached_days": stop_step * dt / 86400,
    }


if __name__ == "__main__":
    result = run_task_f(Nx=100, Ny=100, days=5, dt=10.0)

    print("Task F semi-Lagrangian run")
    print("stable:", result["stable"])
    print("time reached:", result["time_reached_days"], "days")
    print("max |eta|:", np.nanmax(np.abs(result["eta"])))

    plot_eta_snapshot(
        result["eta"],
        L,
        days=result["days"],
        time=result["dt"],
        title=f"SL {result['days']} days, dt = {result['dt']} s"
    )