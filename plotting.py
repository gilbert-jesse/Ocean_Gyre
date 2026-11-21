import numpy as np
import matplotlib.pyplot as plt


def clean_filename(text):
    return (
        text.replace(" ", "_")
            .replace("×", "x")
            .replace("/", "_")
            .replace(":", "")
            .replace(",", "")
    )



def plot_gyre_model(result, title=""):
    eta = result["eta"]
    u = result["u"]
    v = result["v"]

    Ny = result["Ny"]
    j_south = 0
    i_west = 0
    j_mid = Ny // 2

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    ax[0, 0].plot(result["x_u"], u[j_south, :])
    ax[0, 0].set_xlabel("x (m)")
    ax[0, 0].set_ylabel("u (m s$^{-1}$)")
    ax[0, 0].set_title("(a) u versus x near southern edge")
    ax[0, 0].grid()

    ax[0, 1].plot(result["y_v"], v[:, i_west])
    ax[0, 1].set_xlabel("y (m)")
    ax[0, 1].set_ylabel("v (m s$^{-1}$)")
    ax[0, 1].set_title("(b) v versus y near western edge")
    ax[0, 1].grid()

    ax[1, 0].plot(result["x_eta"], eta[j_mid, :])
    ax[1, 0].set_xlabel("x (m)")
    ax[1, 0].set_ylabel("η (m)")
    ax[1, 0].set_title("(c) η versus x through middle")
    ax[1, 0].grid()

    cp = ax[1, 1].contourf(
        result["X_eta"],
        result["Y_eta"],
        eta,
        levels=20
    )
    fig.colorbar(cp, ax=ax[1, 1], label="η (m)")
    ax[1, 1].set_xlabel("x (m)")
    ax[1, 1].set_ylabel("y (m)")
    ax[1, 1].set_title("(d) Surface elevation η")

    fig.suptitle(f"Model {title}")
    fig.tight_layout()
    plt.savefig(f"gyre_model_{clean_filename(title)}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_energy(result, anomaly=False, title_suffix=""):
    E = result["energy"]

    if anomaly:
        E = E - E[0]
        ylabel = "Energy anomaly (J)"
        title = "(a) Perturbation energy anomaly versus time"
    else:
        ylabel = "Energy (J)"
        title = "(a) Total perturbation energy versus time"

    plt.figure(figsize=(8, 5))
    plt.plot(result["time_days"], E)
    plt.xlabel("time (days)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()

    suffix = clean_filename(title_suffix) if title_suffix else "energy"
    plt.savefig(f"energy_plot_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_error_fields(result, error_result, title=""):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    cp0 = ax[0].contourf(
        result["X_u"],
        result["Y_u"],
        error_result["u_prime"],
        levels=20
    )
    fig.colorbar(cp0, ax=ax[0], label="u error (m s$^{-1}$)")
    ax[0].set_title("(a) u error")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")

    cp1 = ax[1].contourf(
        result["X_v"],
        result["Y_v"],
        error_result["v_prime"],
        levels=20
    )
    fig.colorbar(cp1, ax=ax[1], label="v error (m s$^{-1}$)")
    ax[1].set_title("(b) v error")
    ax[1].set_xlabel("x (m)")
    ax[1].set_ylabel("y (m)")

    cp2 = ax[2].contourf(
        result["X_eta"],
        result["Y_eta"],
        error_result["eta_prime"],
        levels=20
    )
    fig.colorbar(cp2, ax=ax[2], label="η error (m)")
    ax[2].set_title("(c) η error")
    ax[2].set_xlabel("x (m)")
    ax[2].set_ylabel("y (m)")

    fig.suptitle(f"Numerical minus analytical {title}")
    fig.tight_layout()
    plt.savefig(f"error_fields_{clean_filename(title)}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_grid_refinement(coarse, fine, coarse_err=None, fine_err=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(
        coarse["time_days"],
        coarse["energy"] - coarse["energy"][0],
        label=f"{coarse['Nx']}x{coarse['Ny']}"
    )

    ax[0].plot(
        fine["time_days"],
        fine["energy"] - fine["energy"][0],
        label=f"{fine['Nx']}x{fine['Ny']}"
    )

    ax[0].set_xlabel("time (days)")
    ax[0].set_ylabel("Energy anomaly (J)")
    ax[0].set_title("(a) Energy evolution")
    ax[0].grid()
    ax[0].legend()

    if coarse_err is not None and fine_err is not None:
        ax[1].bar(
            [f"{coarse['Nx']}x{coarse['Ny']}", f"{fine['Nx']}x{fine['Ny']}"],
            [coarse_err["E_error"], fine_err["E_error"]]
        )
        ax[1].set_ylabel("Error energy (J)")
        ax[1].set_title("(b) Error energy")
        ax[1].grid(axis="y")
    else:
        ax[1].axis("off")

    fig.tight_layout()
    plt.savefig("fine_coarse_grid.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_eta_snapshot(eta, L, days, time, title=""):
    Ny, Nx = eta.shape

    dx = L / Nx
    dy = L / Ny

    x = (np.arange(Nx) + 0.5) * dx / 1000
    y = (np.arange(Ny) + 0.5) * dy / 1000

    plt.figure(figsize=(6, 5))

    cf = plt.contourf(x, y, eta, levels=30)

    plt.colorbar(cf, label=r"$\eta$ (m)")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title(title)

    plt.tight_layout()
    fname = clean_filename(f"SL_{days}_dt_{time}")
    plt.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_analytical_eta(task_c):
    plt.figure(figsize=(6, 5))

    cp = plt.contourf(
        task_c["X"],
        task_c["Y"],
        task_c["eta_st"],
        levels=20
    )

    plt.colorbar(cp, label="eta (m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Analytical steady-state elevation")

    plt.tight_layout()
    plt.savefig("taskC_analytical_eta.png", dpi=300, bbox_inches="tight")
    plt.show()

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
