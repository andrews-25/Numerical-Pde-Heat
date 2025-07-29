import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the 1D neural network model
class Net1D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net1D, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        self.apply(self.xavier_init)

    def forward(self, x):
        return self.hidden(x)

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

# Compute the full loss
def compute_loss(net, x, t, bc_weight=1.0, ic_weight=1.0, pde_weight=10.0):
    xt = torch.cat([x, t], dim=1).to(device)
    xt.requires_grad = True
    u = net(xt)
    #PDE Loss
    grads = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 1:2]
    u_x = grads[:, 0:1]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    pde_loss = ((u_t - k * u_xx) ** 2).mean()
    # Boundary Condition loss
    t_bc = torch.rand(x.shape[0], 1).to(device)
    x0 = torch.zeros_like(t_bc)
    x1 = torch.ones_like(t_bc)
    u0 = net(torch.cat([x0, t_bc], dim=1))
    u1 = net(torch.cat([x1, t_bc], dim=1))
    bc_loss = (u0 ** 2).mean() + (u1 ** 2).mean()
    # Initial Condition loss
    x_ic = torch.rand(x.shape[0], 1).to(device)
    t0 = torch.zeros_like(x_ic)
    u_ic = net(torch.cat([x_ic, t0], dim=1))
    ic_loss = ((u_ic - torch.sin(np.pi * x_ic)) ** 2).mean()
    #total loss 
    total_loss = pde_weight * pde_loss + bc_weight * bc_loss + ic_weight * ic_loss
    return total_loss, pde_loss.item(), bc_loss.item(), ic_loss.item()
# Training parameters
k = 0.1
net = Net1D(2, 40, 1).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
epochs = 15000
N_train = 6000
loss_history = []
for epoch in range(1, epochs + 1):

    #x = torch.linspace(0, 1, N_train).to(device)
    #t = torch.linspace(0, 1, N_train).to(device)

    x = torch.rand(N_train, 1).to(device) #Random DIS points
    t = torch.rand(N_train, 1).to(device) #Random DIS points

    optimizer.zero_grad()
    loss, pde_l, bc_l, ic_l = compute_loss(net, x, t)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch}: Loss={loss.item():.6f} | PDE={pde_l:.6f}, BC={bc_l:.6f}, IC={ic_l:.6f}")
        loss_history.append(loss.item())

# After training, calculate the error between predicted and true solutions
# Compute the predicted u(x,t) from the PINN
u_pred_all = []
u_true_all = []
x_vals = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)  # Grid for x
time_stamps = [0.0, 0.25, 0.5, 0.75, 1.0]  # Time stamps to evaluate

for t_val in time_stamps:
    t_vals = torch.full_like(x_vals, t_val).to(device)  # Set time for each point in x
    xt = torch.cat([x_vals, t_vals], dim=1)  # Concatenate x and t
    u_pred = net(xt).detach().cpu().numpy().flatten()  # Get predicted values
    u_true = np.exp(-np.pi ** 2 * k * t_val) * np.sin(np.pi * x_vals.cpu().numpy().flatten())  # Analytical solution
    
    u_pred_all.append(u_pred)
    u_true_all.append(u_true)

# Flatten the lists into arrays for easier calculation
u_pred_all = np.concatenate(u_pred_all)
u_true_all = np.concatenate(u_true_all)

# Calculate MSE (Mean Squared Error)
mse = np.mean((u_pred_all - u_true_all) ** 2)

# Calculate MAE (Mean Absolute Error)
mae = np.mean(np.abs(u_pred_all - u_true_all))
# Calculate Maximum Absolute Error
max_error = np.max(np.abs(u_pred_all - u_true_all))

print("_____PINN Metrics_____")
print(f"Maximum Absolute Error (Max Error): {max_error:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Plot cross-sections at different times
plt.figure(figsize=(10, 6))
for t_val in time_stamps:
    t_vals = torch.full_like(x_vals, t_val).to(device)
    xt = torch.cat([x_vals, t_vals], dim=1)
    u_pred = net(xt).detach().cpu().numpy().flatten()
    u_true = np.exp(-np.pi ** 2 * k * t_val) * np.sin(np.pi * x_vals.cpu().numpy().flatten())
    plt.plot(x_vals.cpu().numpy(), u_pred, label=f"Predicted t={t_val}")
    plt.plot(x_vals.cpu().numpy(), u_true, '--', label=f"True t={t_val}")

plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Comparison of PINN vs Analytical Solution")
plt.grid(True)
plt.show()

# Generate a grid for (x, t)
x_heat = torch.linspace(0, 1, 100).reshape(-1, 1)
t_heat = torch.linspace(0, 1, 100).reshape(-1, 1)
X, T = torch.meshgrid(x_heat.squeeze(), t_heat.squeeze(), indexing='ij')

# Flatten and prepare input for the network
x_flat = X.reshape(-1, 1).to(device)
t_flat = T.reshape(-1, 1).to(device)
xt_heat = torch.cat([x_flat, t_flat], dim=1)

# Predict with PINN
u_pred_heat = net(xt_heat).detach().cpu().numpy().reshape(100, 100)

# Analytical solution
u_true_heat = np.exp(-np.pi**2 * k * T.numpy()) * np.sin(np.pi * X.numpy())

# Compute absolute error
error_heat = np.abs(u_pred_heat - u_true_heat)

plt.figure(figsize=(18, 5))  # Wider figure to fit three plots side by side

# 1. PINN prediction
plt.subplot(1, 3, 1)
plt.imshow(u_pred_heat.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='u(x,t)')
plt.title('PINN Prediction $u(x,t)$')
plt.xlabel('x')
plt.ylabel('t')

# 2. Analytical solution
plt.subplot(1, 3, 2)
plt.imshow(u_true_heat.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='u(x,t)')
plt.title('Analytical Solution $u(x,t)$')
plt.xlabel('x')
plt.ylabel('t')

# 3. Absolute error
plt.subplot(1, 3, 3)
plt.imshow(error_heat.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='|Error|')
plt.title('Absolute Error Heatmap')
plt.xlabel('x')
plt.ylabel('t')

plt.tight_layout()
plt.show()




