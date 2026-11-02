import keras
import numpy as np

keras.utils.set_random_seed(0)

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 2000
LR = 1e-2
SAMPLES = 64
NOISE = 0.1

# model settings
LAYERS = [1, 24, 24, 24, 1]

# ------------------------------------ create data ------------------------------------
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (SAMPLES, 1))
Y = np.sin(2 * np.pi * X) + NOISE * rng.standard_normal((SAMPLES, 1))

# --------------------------- instantiate model & optimizer ---------------------------
model = keras.Sequential([keras.Input((LAYERS[0],))])
for width in LAYERS[1:-1]:
    model.add(keras.layers.Dense(width, activation="tanh"))
model.add(keras.layers.Dense(LAYERS[-1]))
model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")

# -------------------------------------- training -------------------------------------
history = model.fit(X, Y, batch_size=SAMPLES, epochs=EPOCHS, verbose=0)
print(f"cost {history.history['loss'][-1]:.2e}")
