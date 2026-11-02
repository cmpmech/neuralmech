#include <iostream>

#include <torch/torch.h>

int main() {
    torch::manual_seed(0);

    // ----------------------------------- settings -----------------------------------
    // hyperparameters
    const int EPOCHS = 2000;
    const double LR = 1e-2;
    const int SAMPLES = 64;
    const double NOISE = 0.1;

    // model settings
    const std::vector<int64_t> LAYERS = {1, 24, 24, 24, 1};

    // ---------------------------------- create data ---------------------------------
    auto X = 2 * torch::rand({SAMPLES, 1}) - 1;
    auto noise = NOISE * torch::randn({SAMPLES, 1});
    auto Y = torch::sin(2 * M_PI * X) + noise;

    // ------------------------- instantiate model & optimizer ------------------------
    torch::nn::Sequential model;
    for (size_t i = 0; i + 2 < LAYERS.size(); ++i) {
        model->push_back(torch::nn::Linear(LAYERS[i], LAYERS[i + 1]));
        model->push_back(torch::nn::Tanh());
    }
    int64_t last = LAYERS.size() - 1;
    model->push_back(torch::nn::Linear(LAYERS[last - 1], LAYERS[last]));
    torch::optim::Adam optimizer(model->parameters(), LR);

    // ----------------------------------- training -----------------------------------
    torch::Tensor cost;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        optimizer.zero_grad();
        cost = torch::mse_loss(model->forward(X), Y);
        cost.backward();
        optimizer.step();
    }
    std::cout << "cost " << cost.item<double>() << std::endl;
}
