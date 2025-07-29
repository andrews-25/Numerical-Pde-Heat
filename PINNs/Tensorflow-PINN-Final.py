import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# === GPU SETUP ===
print("--- TensorFlow Environment Info ---")
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Available devices: {tf.config.list_physical_devices()}")

# Enable dynamic memory growth on GPU (if available)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0]}")
    else:
        print("No GPU found. Running on CPU.")
except RuntimeError as e:
    print(f"GPU configuration failed: {e}")
    gpus = []

# === Define the PINN Model ===
def create_network(input_size, hidden_size, output_size):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='tanh',
                              kernel_initializer='glorot_normal', bias_initializer='zeros',
                              input_shape=(input_size,)),
        tf.keras.layers.Dense(hidden_size, activation='tanh',
                              kernel_initializer='glorot_normal', bias_initializer='zeros'),
        tf.keras.layers.Dense(output_size,
                              kernel_initializer='glorot_normal', bias_initializer='zeros')
    ])

# === Compute Gradients ===
@tf.function
def compute_gradients(net, xt):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xt)
        with tf.GradientTape() as tape1:
            tape1.watch(xt)
            u = net(xt)
        grads = tape1.gradient(u, xt)
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
    u_xx = tape2.gradient(u_x, xt)[:, 0:1]
    del tape2
    return u, u_x, u_t, u_xx

# === Define the Loss Function ===
@tf.function
def compute_loss(net, x, t, k, bc_weight=1.0, ic_weight=1.0, pde_weight=10.0):
    xt = tf.concat([x, t], axis=1)
    u, u_x, u_t, u_xx = compute_gradients(net, xt)
    pde_loss = tf.reduce_mean((u_t - k * u_xx) ** 2)

    t_bc = tf.random.uniform(tf.shape(x), dtype=tf.float32)
    u0 = net(tf.concat([tf.zeros_like(t_bc), t_bc], axis=1))
    u1 = net(tf.concat([tf.ones_like(t_bc), t_bc], axis=1))
    bc_loss = tf.reduce_mean(u0 ** 2) + tf.reduce_mean(u1 ** 2)

    x_ic = tf.random.uniform(tf.shape(x), dtype=tf.float32)
    u_ic = net(tf.concat([x_ic, tf.zeros_like(x_ic)], axis=1))
    ic_loss = tf.reduce_mean((u_ic - tf.sin(np.pi * x_ic)) ** 2)

    total_loss = pde_weight * pde_loss + bc_weight * bc_loss + ic_weight * ic_loss
    return total_loss, pde_loss, bc_loss, ic_loss

# === Perform One Training Step ===
@tf.function
def train_step(net, optimizer, x, t, k):
    with tf.GradientTape() as tape:
        loss, pde_l, bc_l, ic_l = compute_loss(net, x, t, k)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss, pde_l, bc_l, ic_l

# === Sample Random Training Points ===
@tf.function
def generate_training_sample(N):
    x = tf.random.uniform((N, 1), dtype=tf.float32)
    t = tf.random.uniform((N, 1), dtype=tf.float32)
    return x, t

# === Training Loop Wrapper ===
@tf.function
def training_loop_step(net, optimizer, k, N):
    x_batch, t_batch = generate_training_sample(N)
    return train_step(net, optimizer, x_batch, t_batch, k)

# === Training Parameters ===
k = 0.1
N_train = 6000
net = create_network(2, 40, 1)
_ = net(tf.random.normal((1, 2)))  # build the model
optimizer = tf.keras.optimizers.Adam(1e-3)
epochs = 15000

print("Starting training...")
loss_history = []
for epoch in range(1, epochs + 1):
    loss, pde_l, bc_l, ic_l = training_loop_step(net, optimizer, k, N_train)
    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch}: Loss={loss.numpy():.6f} | PDE={pde_l.numpy():.6f}, BC={bc_l.numpy():.6f}, IC={ic_l.numpy():.6f}")
        loss_history.append(loss.numpy())

print("Training complete.")

# === Evaluate Model ===
x_vals = tf.linspace(0.0, 1.0, 100)
x_vals = tf.reshape(x_vals, (-1, 1))
time_stamps = [0.0, 0.25, 0.5, 0.75, 1.0]
u_pred_all = []
u_true_all = []

for t_val in time_stamps:
    t_vals = tf.fill(tf.shape(x_vals), t_val)
    xt = tf.concat([x_vals, t_vals], axis=1)
    u_pred = net(xt).numpy().flatten()
    u_true = np.exp(-np.pi ** 2 * k * t_val) * np.sin(np.pi * x_vals.numpy().flatten())
    u_pred_all.append(u_pred)
    u_true_all.append(u_true)

u_pred_all = np.concatenate(u_pred_all)
u_true_all = np.concatenate(u_true_all)

print("\n--- Model Metrics ---")
print(f"Max Error: {np.max(np.abs(u_pred_all - u_true_all)):.6f}")
print(f"MSE: {np.mean((u_pred_all - u_true_all)**2):.6f}")
print(f"MAE: {np.mean(np.abs(u_pred_all - u_true_all)):.6f}")

# === Plot Prediction vs Analytical ===
plt.figure(figsize=(10, 6))
for t_val in time_stamps:
    t_vals = tf.fill(tf.shape(x_vals), t_val)
    xt = tf.concat([x_vals, t_vals], axis=1)
    u_pred = net(xt).numpy().flatten()
    u_true = np.exp(-np.pi ** 2 * k * t_val) * np.sin(np.pi * x_vals.numpy().flatten())
    plt.plot(x_vals.numpy(), u_pred, label=f"Pred t={t_val}")
    plt.plot(x_vals.numpy(), u_true, '--', label=f"True t={t_val}")

plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("PINN Prediction vs Analytical Solution")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Heatmaps ===
print("Generating heatmaps...")
x_heat = tf.linspace(0.0, 1.0, 100)
t_heat = tf.linspace(0.0, 1.0, 100)
X, T = tf.meshgrid(x_heat, t_heat, indexing='ij')
x_flat = tf.reshape(X, (-1, 1))
t_flat = tf.reshape(T, (-1, 1))
ext = tf.concat([x_flat, t_flat], axis=1)
u_pred_heat = net(ext).numpy().reshape(100, 100)
u_true_heat = np.exp(-np.pi**2 * k * T.numpy()) * np.sin(np.pi * X.numpy())
error_heat = np.abs(u_pred_heat - u_true_heat)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(u_pred_heat.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='u(x,t)')
plt.title('PINN Prediction')

plt.subplot(1, 3, 2)
plt.imshow(u_true_heat.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='u(x,t)')
plt.title('Analytical Solution')

plt.subplot(1, 3, 3)
plt.imshow(error_heat.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis', interpolation ='bilinear')
plt.colorbar(label='|Error|')
plt.title('Absolute Error Heatmap')

plt.tight_layout()
plt.show()
print("Done.")
