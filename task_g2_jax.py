import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from parameters import L, H, g, gamma, rho, f0, beta, tau0
from diagnostics import run_gyre_model

jax.config.update("jax_enable_x64", True)


@jax.jit
def apply_bc_jax(u, v):
    u = u.at[:, 0].set(0.0)
    u = u.at[:, -1].set(0.0)
    v = v.at[0, :].set(0.0)
    v = v.at[-1, :].set(0.0)
    return u, v


@jax.jit
def divergence_jax(u, v, dx, dy):
    return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy


@jax.jit
def grad_eta_u_jax(eta, dx):
    Ny, Nx = eta.shape
    out = jnp.zeros((Ny, Nx + 1))
    out = out.at[:, 1:-1].set((eta[:, 1:] - eta[:, :-1]) / dx)
    return out


@jax.jit
def grad_eta_v_jax(eta, dy):
    Ny, Nx = eta.shape
    out = jnp.zeros((Ny + 1, Nx))
    out = out.at[1:-1, :].set((eta[1:, :] - eta[:-1, :]) / dy)
    return out


@jax.jit
def interp_v_to_u_jax(v):
    Ny_plus_1, Nx = v.shape
    Ny = Ny_plus_1 - 1
    out = jnp.zeros((Ny, Nx + 1))
    vals = 0.25 * (
        v[:-1, :-1] + v[:-1, 1:] +
        v[1:, :-1] + v[1:, 1:]
    )
    out = out.at[:, 1:-1].set(vals)
    return out


@jax.jit
def interp_u_to_v_jax(u):
    Ny, Nx_plus_1 = u.shape
    Nx = Nx_plus_1 - 1
    out = jnp.zeros((Ny + 1, Nx))
    vals = 0.25 * (
        u[:-1, :-1] + u[:-1, 1:] +
        u[1:, :-1] + u[1:, 1:]
    )
    out = out.at[1:-1, :].set(vals)
    return out


@jax.jit
def energy_jax(u, v, eta, dx, dy, rho, H, g):
    eta_anom = eta - jnp.mean(eta)
    E_u = 0.5 * rho * H * jnp.sum(u**2) * dx * dy
    E_v = 0.5 * rho * H * jnp.sum(v**2) * dx * dy
    E_eta = 0.5 * rho * g * jnp.sum(eta_anom**2) * dx * dy
    return E_u + E_v + E_eta

@jax.jit
def taskD_cycle_jax(state, const):
    u, v, eta = state

    dx, dy, dt, H, g, gamma, rho, f0, beta, tau0, L = const

    Ny, Nx = eta.shape

    y_eta = (jnp.arange(Ny) + 0.5) * dy
    y_u = y_eta
    y_v = jnp.arange(Ny + 1) * dy

    f_u = f0 + beta * y_u[:, None]
    f_v = f0 + beta * y_v[:, None]

    tau_x_u = -tau0 * jnp.cos(jnp.pi * y_u[:, None] / L)
    tau_y_v = jnp.zeros((Ny + 1, Nx))

    # first half step: u before v
    u, v = apply_bc_jax(u, v)

    eta_new = eta - H * dt * divergence_jax(u, v, dx, dy)

    u_new = (
        u
        + f_u * dt * interp_v_to_u_jax(v)
        - g * dt * grad_eta_u_jax(eta_new, dx)
        - gamma * dt * u
        + (tau_x_u / (rho * H)) * dt
    )

    u_new, v = apply_bc_jax(u_new, v)

    v_new = (
        v
        - f_v * dt * interp_u_to_v_jax(u_new)
        - g * dt * grad_eta_v_jax(eta_new, dy)
        - gamma * dt * v
        + (tau_y_v / (rho * H)) * dt
    )

    eta, u, v = eta_new, u_new, v_new
    u, v = apply_bc_jax(u, v)

    # second half step: v before u
    eta_new = eta - H * dt * divergence_jax(u, v, dx, dy)

    v_new = (
        v
        - f_v * dt * interp_u_to_v_jax(u)
        - g * dt * grad_eta_v_jax(eta_new, dy)
        - gamma * dt * v
        + (tau_y_v / (rho * H)) * dt
    )

    u, v_new = apply_bc_jax(u, v_new)

    u_new = (
        u
        + f_u * dt * interp_v_to_u_jax(v_new)
        - g * dt * grad_eta_u_jax(eta_new, dx)
        - gamma * dt * u
        + (tau_x_u / (rho * H)) * dt
    )

    eta, u, v = eta_new, u_new, v_new
    u, v = apply_bc_jax(u, v)

    return u, v, eta


@partial(jax.jit, static_argnames=("nt",))
def run_taskD_scan_jax(u0, v0, eta0, const, nt):
    dx, dy, dt, H, g, gamma, rho, f0, beta, tau0, L = const

    def body(carry, n):
        u, v, eta = taskD_cycle_jax(carry, const)
        E = energy_jax(u, v, eta, dx, dy, rho, H, g)
        t_days = (n + 1) * 2.0 * dt / 86400.0
        return (u, v, eta), (E, t_days)

    final_state, hist = lax.scan(body, (u0, v0, eta0), jnp.arange(nt))
    energy, time_days = hist

    return final_state, energy, time_days

def run_gyre_model_jax(Nx=100, Ny=100, days=1, cfl_factor=0.5, lower_dt=None):
    dx = L / Nx
    dy = L / Ny

    c = np.sqrt(g * H)
    dt_cfl = 1.0 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))

    if lower_dt is None:
        dt = cfl_factor * dt_cfl
    else:
        dt = lower_dt

    nt = int(days * 24 * 3600 / (2 * dt))

    x_eta = (np.arange(Nx) + 0.5) * dx
    y_eta = (np.arange(Ny) + 0.5) * dy

    x_u = np.arange(Nx + 1) * dx
    y_u = y_eta.copy()

    x_v = x_eta.copy()
    y_v = np.arange(Ny + 1) * dy

    X_eta, Y_eta = np.meshgrid(x_eta, y_eta, indexing="xy")
    X_u, Y_u = np.meshgrid(x_u, y_u, indexing="xy")
    X_v, Y_v = np.meshgrid(x_v, y_v, indexing="xy")

    eta0 = jnp.zeros((Ny, Nx))
    u0 = jnp.zeros((Ny, Nx + 1))
    v0 = jnp.zeros((Ny + 1, Nx))

    const = jnp.array([dx, dy, dt, H, g, gamma, rho, f0, beta, tau0, L])

    start = time.time()
    (u, v, eta), energy, time_days = run_taskD_scan_jax(u0, v0, eta0, const, nt)
    eta.block_until_ready()
    first_time = time.time() - start

    start = time.time()
    (u, v, eta), energy, time_days = run_taskD_scan_jax(u0, v0, eta0, const, nt)
    eta.block_until_ready()
    compiled_time = time.time() - start

    print("JAX first run including compilation:", first_time)
    print("JAX compiled runtime:", compiled_time)

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
        "eta": np.array(eta),
        "u": np.array(u),
        "v": np.array(v),
        "energy": np.array(energy),
        "time_days": np.array(time_days),
        "jax_first_time": first_time,
        "jax_compiled_time": compiled_time,
    }

def benchmark_taskD(Nx=100, Ny=100, days=30, dt=10.0):
    print("Benchmark setup")
    print("Nx, Ny:", Nx, Ny)
    print("days:", days)
    print("dt:", dt)

    start = time.time()

    res_np = run_gyre_model(
        Nx=Nx,
        Ny=Ny,
        days=days,
        lower_dt=dt
    )

    numpy_runtime = time.time() - start


    start = time.time()

    res_jax_first = run_gyre_model_jax(
        Nx=Nx,
        Ny=Ny,
        days=days,
        lower_dt=dt
    )

    jax.device_get(res_jax_first["eta"])
    jax_first_runtime = time.time() - start

    start = time.time()

    res_jax = run_gyre_model_jax(
        Nx=Nx,
        Ny=Ny,
        days=days,
        lower_dt=dt
    )

    jax.device_get(res_jax["eta"])
    jax_compiled_runtime = time.time() - start


    eta_diff = np.max(np.abs(res_np["eta"] - res_jax["eta"]))
    u_diff = np.max(np.abs(res_np["u"] - res_jax["u"]))
    v_diff = np.max(np.abs(res_np["v"] - res_jax["v"]))
    energy_diff = np.max(np.abs(res_np["energy"] - res_jax["energy"]))

    speedup = numpy_runtime / jax_compiled_runtime

    print("\nRuntime")
    print("NumPy runtime:", numpy_runtime)
    print("JAX first run, including compilation:", jax_first_runtime)
    print("JAX compiled runtime:", jax_compiled_runtime)
    print("Speedup, NumPy/JAX compiled:", speedup)

    print("\nAccuracy")
    print("max eta difference:", eta_diff)
    print("max u difference:", u_diff)
    print("max v difference:", v_diff)
    print("max energy difference:", energy_diff)

    return {
        "res_np": res_np,
        "res_jax": res_jax,
        "numpy_runtime": numpy_runtime,
        "jax_first_runtime": jax_first_runtime,
        "jax_compiled_runtime": jax_compiled_runtime,
        "speedup": speedup,
        "eta_diff": eta_diff,
        "u_diff": u_diff,
        "v_diff": v_diff,
        "energy_diff": energy_diff,
    }

if __name__ == "__main__":
    bench = benchmark_taskD(
        Nx=100,
        Ny=100,
        days=200,
        dt=30.0
    )