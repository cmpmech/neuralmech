using Lux, Optimisers, Random, Zygote

rng = Xoshiro(0)

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 2000
LR = 1e-2
SAMPLES = 64
NOISE = 0.1f0

# define loss
cost_fun = MSELoss()

# model settings
LAYERS = [1, 24, 24, 24, 1]
L = length(LAYERS)

# ------------------------------------ create data ------------------------------------
X = 2 .* rand(rng, Float32, 1, SAMPLES) .- 1
Y = sin.(2f0 * pi .* X) .+ NOISE .* randn(rng, Float32, 1, SAMPLES)

# --------------------------- instantiate model & optimizer ---------------------------
hidden = [Dense(LAYERS[i] => LAYERS[i + 1], tanh) for i in 1:L - 2]
model = Chain(hidden..., Dense(LAYERS[L - 1] => LAYERS[L]))
params, state = Lux.setup(rng, model)
loss(params) = cost_fun(first(model(X, params, state)), Y)
opt_state = Optimisers.setup(Adam(LR), params)

# -------------------------------------- training -------------------------------------
for epoch in 1:EPOCHS
    global cost, grads = Zygote.withgradient(loss, params)
    Optimisers.update!(opt_state, params, grads[1])
end
println("cost $(round(cost, sigdigits=3))")
