import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# === Parameters ===
L = 1.0
Nx = 400
dx = L / (Nx - 1)
T = 1.0
Nt = 20000  
dt = T / Nt
k = 0.1

s = k * dt / dx**2
print(f"s: {s:.6f}")

x = np.linspace(0, L, Nx)
u = np.sin(np.pi * x)
u[0] = 0  # Dirichlet BC at x=0
u[-1] = 0  # Dirichlet BC at x=L

solution = np.zeros((Nt + 1, Nx))
solution[0, :] = u

# === Construct Crank-Nicolson Matrices (for interior points only) ===
A = np.zeros((3, Nx))
A[0, 1:-1] = -s / 2
A[1, 1:-1] = 1 + s
A[2, 1:-1] = -s / 2

# Dirichlet BCs: keep u[0] and u[-1] fixed at 0
A[1, 0] = 1.0
A[1, -1] = 1.0
A[0, 0] = 0.0
A[2, -1] = 0.0

# Matrix B
B = np.eye(Nx) * (1 - s)
B += np.diag([s / 2] * (Nx - 1), k=1)
B += np.diag([s / 2] * (Nx - 1), k=-1)  
B[0, :] = 0
B[0, 0] = 1
B[-1, :] = 0
B[-1, -1] = 1

# === Time stepping loop ===
for n in range(1, Nt + 1):
    b_ = B @ u
    b_[0] = 0.0     # enforce Dirichlet BC at x=0
    b_[-1] = 0.0    # enforce Dirichlet BC at x=L
    u = solve_banded((1, 1), A, b_)
    u[0] = 0.0      # enforce Dirichlet BC after solve
    u[-1] = 0.0
    solution[n, :] = u

# === Analytical solution ===
def exact_solution(x, t, k):
    return np.exp(-np.pi**2 * k * t) * np.sin(np.pi * x)

t_values = np.linspace(0, T, Nt + 1)
exact_sol = np.array([[exact_solution(x[j], t_values[i], k) for j in range(Nx)] for i in range(Nt + 1)])

error = np.abs(solution - exact_sol)
mae = np.mean(error)
mse = np.mean(error**2)
max_error = np.max(error)
print("_____Crank Nicolson Metrics_____")
print(f"Max Error: {max_error:.6f} °C")
print(f"Mean Squared Error (MSE): {mse:.6f} °C²")
print(f"Mean Absolute Error (MAE): {mae:.6f} °C")


# === Plotting ===
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

im1 = axs[0].imshow(solution, extent=[0, L, 0, T], origin='lower', aspect='auto', cmap='jet')
axs[0].set_title("Numerical Solution (Crank-Nicolson)")
axs[0].set_xlabel("Position x [m]")
axs[0].set_ylabel("Time t [s]")
fig.colorbar(im1, ax=axs[0], label="Temperature (°C)")

im2 = axs[1].imshow(exact_sol, extent=[0, L, 0, T], origin='lower', aspect='auto', cmap='jet')
axs[1].set_title("Analytical Solution")
axs[1].set_xlabel("Position x [m]")
axs[1].set_ylabel("Time t [s]")
fig.colorbar(im2, ax=axs[1], label="Temperature (°C)")

plt.tight_layout()
plt.show()

# === Plot Error Map ===
plt.figure(figsize=(8, 6))
plt.imshow(error, extent=[0, L, 0, T], origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Absolute Error (°C)")
plt.xlabel("Position (x) [m]")
plt.ylabel("Time (t) [s]")
plt.title("Error Between Numerical and Analytical Solutions")
plt.tight_layout()
plt.show()
