import numpy as np
from parameters import L, f0, beta, g, gamma, rho, H, tau0


# Analytical Stommel solution

def stommel_exact_solution(X, Y, eta0=0.0):
    eps = gamma / (L * beta)

    a = (-1.0 - np.sqrt(1.0 + (2.0 * np.pi * eps)**2)) / (2.0 * eps)
    b = (-1.0 + np.sqrt(1.0 + (2.0 * np.pi * eps)**2)) / (2.0 * eps)

    x = X / L

    f1 = np.pi * (
        1.0
        + ((np.exp(a) - 1.0) * np.exp(b * x)
        + (1.0 - np.exp(b)) * np.exp(a * x))
        / (np.exp(b) - np.exp(a))
    )

    f2 = (
        ((np.exp(a) - 1.0) * b * np.exp(b * x)
        + (1.0 - np.exp(b)) * a * np.exp(a * x))
        / (np.exp(b) - np.exp(a))
    )

    u_st = -tau0 / (np.pi * gamma * rho * H) * f1 * np.cos(np.pi * Y / L)

    v_st = tau0 / (np.pi * gamma * rho * H) * f2 * np.sin(np.pi * Y / L)

    eta_st = eta0 + (
        tau0 / (np.pi * gamma * rho * H)
        * (f0 * L / g)
        * (
            (gamma / (f0 * np.pi)) * f2 * np.cos(np.pi * Y / L)
            + (1.0 / np.pi) * f1
            * (
                np.sin(np.pi * Y / L) * (1.0 + beta * Y / f0)
                + (beta * L / (f0 * np.pi)) * np.cos(np.pi * Y / L)
            )
        )
    )

    return u_st, v_st, eta_st


def run_task_c(Nx=100, Ny=100, eta0=0.0):
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    u_st, v_st, eta_st = stommel_exact_solution(
        X,
        Y,
        eta0=eta0
    )

    eta_idx = np.argmin(np.abs(y - L / 2))
    eta_ref = eta_st[0, eta_idx]

    return {
        "Nx": Nx,
        "Ny": Ny,
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "u_st": u_st,
        "v_st": v_st,
        "eta_st": eta_st,
        "eta_ref": eta_ref,
    }