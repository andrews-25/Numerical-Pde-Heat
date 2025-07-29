import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import solve_banded

"""
Crank Nicolson ADI solver for the 2D heat equation on a unit square
with Dirichlet boundaries.
"""

#Parameters
L = 1.0          # domain side length  [m]
Nx = 100         # grid points in x
Ny = 100         # grid points in y
dx = L / (Nx - 1)
dy = L / (Ny - 1)

T  = 1.0         # total simulation time [s]
Nt = 500         # number of time steps
dt = T / Nt
k  = 0.1         # thermal diffusivity   [m^2/s]

x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# stability numbers (Crank–Nicolson half-step coefficients)
s_x = k * dt / (2 * dx**2)
s_y = k * dt / (2 * dy**2)
print(f"s_x: {s_x:.6f}, s_y: {s_y:.6f}")

#IC
u = np.sin(np.pi * X) * np.sin(np.pi * Y)
# dirichlet BCs (fixed zero temperature)
u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

solution = np.zeros((Nt + 1, Ny, Nx))
solution[0] = u

#build banded matrix for solve_banded
def build_banded_matrix(s, N):
    A = np.zeros((3, N))
    A[0, 1:] = -s           # super-diagonal
    A[1, :] = 1 + 2 * s     # main diagonal
    A[2, :-1] = -s          # sub-diagonal
    # boundary rows (Dirichlet to identity)
    A[1, 0] = A[1, -1] = 1.0
    A[0, 0] = A[2, -1] = 0.0
    return A

#compute banded matrices
Ax = build_banded_matrix(s_x, Nx - 2)
Ay = build_banded_matrix(s_y, Ny - 2)

#prep error metrics
phi0 = np.sin(np.pi * X) * np.sin(np.pi * Y) #shape at t=0

errors_mse  = np.empty(Nt+1)
errors_max  = np.empty(Nt+1)

def update_error_arrays(step, field):
    t = step * dt
    u_exact = np.exp(-2 * np.pi**2 * k * t) * phi0
    diff = field - u_exact
    errors_mse[step] = np.mean(diff**2)
    errors_max[step] = np.max(np.abs(diff))

# initial error
update_error_arrays(0, u)

# time-step loop
for n in range(1, Nt + 1):
    #ADI Stage 1: implicit in x, explicit in y
    u_half = u.copy()
    for j in range(1, Ny - 1):
        rhs = (
            s_y * u[j - 1, 1:-1] +
            (1 - 2 * s_y) * u[j, 1:-1] +
            s_y * u[j + 1, 1:-1]
        )
        rhs[0] = rhs[-1] = 0
        u_half[j, 1:-1] = solve_banded((1, 1), Ax, rhs)

    #ADI Stage 2: implicit in y, explicit in x
    for i in range(1, Nx - 1):
        rhs = (
            s_x * u_half[1:-1, i - 1] +
            (1 - 2 * s_x) * u_half[1:-1, i] +
            s_x * u_half[1:-1, i + 1]
        )
        rhs[0] = rhs[-1] = 0
        u[1:-1, i] = solve_banded((1, 1), Ay, rhs)

    #re-apply Dirichlet boundaries
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    solution[n] = u
    update_error_arrays(n, u)

#plot setup
vmin, vmax = solution.min(), solution.max()

fig, ax = plt.subplots(figsize=(6, 5))
plt.subplots_adjust(bottom=0.25) #room for slider

im = ax.imshow(solution[0], cmap="jet", vmin=vmin, vmax=vmax,
               origin="lower", extent=[0, L, 0, L], aspect="auto")
cbar = fig.colorbar(im, ax=ax, label="Temperature [°C]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

# Replace title with in-plot text for time
time_text = ax.text(
    0.5, 1.02, "", transform=ax.transAxes,
    ha="center", va="bottom", fontsize=12, color="black"
)

# Text box error
err_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    va="top", ha="left", fontsize=9, color="white",
    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5)
)

# slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Frame', 0, Nt, valinit=0, valstep=1)

# update slider function
def update_plot(frame):
    frame = int(frame)
    im.set_data(solution[frame])
    time_text.set_text(f"Temperature distribution, t = {frame*dt:.3f} s")
    err_text.set_text(
        f"MSE = {errors_mse[frame]:.4f}\n"
        f"Max Error = {errors_max[frame]:.4f}"
    )
    fig.canvas.draw_idle()

slider.on_changed(update_plot)

# initialize plot with frame 0
update_plot(0)

plt.show()

# print final summary metrics
print(f"Final-time MSE : {errors_mse[-1]:.3e}")
print(f"Final-time Max Error : {errors_max[-1]:.3e}")
print(f"Mean MSE  over time: {errors_mse.mean():.3e}")
print(f"Mean Max Error over time: {errors_max.mean():.3e}")
