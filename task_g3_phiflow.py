from phi.flow import *

import numpy as np
import matplotlib.pyplot as plt
import time

from parameters import L


def run_g3_phiflow(Nx=100, Ny=100):

    dx = L / Nx
    dy = L / Ny

    domain = dict(
        x=Nx,
        y=Ny,
        bounds=Box(x=L, y=L)
    )

    ################################
    # Staggered velocity field
    

    velocity = StaggeredGrid(
        lambda x: vec(
            x=0.02
            * math.sin(math.PI * x.vector["x"] / L)
            * math.sin(math.PI * x.vector["y"] / L),

            y=0.02
            * math.sin(math.PI * x.vector["x"] / L)
            * math.sin(math.PI * x.vector["y"] / L),
        ),
        extrapolation.BOUNDARY,
        **domain
    )

    #########################
    # Centered eta field

    x_eta = (np.arange(Nx) + 0.5) * dx
    y_eta = (np.arange(Ny) + 0.5) * dy

    X_np, Y_np = np.meshgrid(
        x_eta,
        y_eta,
        indexing="xy"
    )

    eta_np_init = 0.02 * np.exp(
        -(
            (X_np - 0.5 * L)**2 +
            (Y_np - 0.5 * L)**2
        ) / (2 * (0.15 * L)**2)
    )

    eta = CenteredGrid(
        tensor(eta_np_init, spatial("y,x")),
        extrapolation.BOUNDARY,
        **domain
    )

    #################################
    # PhiFlow operators


    div_u = field.divergence(velocity)

    grad_eta = field.spatial_gradient(eta)

    ##############################
    # Convert to NumPy

    u_np = velocity.vector['x'].values.numpy('y,x')
    v_np = velocity.vector['y'].values.numpy('y,x')

    eta_np = eta.values.numpy('y,x')

    div_phi_np = div_u.values.numpy('y,x')

    #############################
    # NumPy divergence operator
    

    def divergence_numpy(u, v, dx, dy):

        return (
            (u[:, 1:] - u[:, :-1]) / dx +
            (v[1:, :] - v[:-1, :]) / dy
        )

    ################################
    # Benchmark NumPy
    

    start = time.time()

    for _ in range(100):

        div_np = divergence_numpy(
            u_np,
            v_np,
            dx,
            dy
        )

    t_np = time.time() - start

    ##############################
    # Benchmark PhiFlow
    

    start = time.time()

    for _ in range(100):

        div_phi = field.divergence(velocity)

        div_phi.values.numpy('y,x')

    t_phi = time.time() - start

    return {
        "eta_np": eta_np,
        "div_phi_np": div_phi_np,
        "numpy_runtime": t_np,
        "phiflow_runtime": t_phi,
        "Nx": Nx,
        "Ny": Ny,
    }


def plot_g3_phiflow(g3):

    eta_np = g3["eta_np"]
    div_phi_np = g3["div_phi_np"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ##################
    # (a) Eta field


    cf1 = ax[0].contourf(
        eta_np,
        levels=20,
        cmap="viridis"
    )

    cbar1 = plt.colorbar(cf1, ax=ax[0])
    cbar1.set_label("eta (m)")

    ax[0].set_title("(a) PhiFlow centered eta field")
    ax[0].set_xlabel("x index")
    ax[0].set_ylabel("y index")

    ##################
    # (b) Divergence field

    cf2 = ax[1].contourf(
        div_phi_np,
        levels=20,
        cmap="RdBu_r"
    )

    cbar2 = plt.colorbar(cf2, ax=ax[1])
    cbar2.set_label("divergence")

    ax[1].set_title(
        "(b) PhiFlow divergence of staggered velocity field"
    )

    ax[1].set_xlabel("x index")
    ax[1].set_ylabel("y index")

    plt.tight_layout()

    plt.savefig(
        "PhiFlow_grid.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    g3 = run_g3_phiflow()

    print("NumPy divergence runtime:", g3["numpy_runtime"])
    print("PhiFlow divergence runtime:", g3["phiflow_runtime"])

    plot_g3_phiflow(g3)