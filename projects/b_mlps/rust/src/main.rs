use std::f64::consts::PI;

use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::Distribution;

// ------------------------------------- settings -------------------------------------
// hyperparameters
const EPOCHS: usize = 2000;
const LR: f64 = 1e-2;
const SAMPLES: usize = 64;
const NOISE: f64 = 0.1;

// model settings
const LAYERS: [usize; 5] = [1, 24, 24, 24, 1];

#[derive(Module, Debug)]
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn forward(&self, mut x: Tensor<2>) -> Tensor<2> {
        let (last, hidden) = self.layers.split_last().unwrap();
        for layer in hidden {
            x = layer.forward(x).tanh();
        }
        last.forward(x)
    }
}

fn main() {
    let device = Device::flex().autodiff();
    device.seed(0);

    // ---------------------------------- create data ---------------------------------
    let uniform = Distribution::Uniform(-1.0, 1.0);
    let normal = Distribution::Normal(0.0, NOISE);
    let x = Tensor::<2>::random([SAMPLES, 1], uniform, &device);
    let noise = Tensor::random([SAMPLES, 1], normal, &device);
    let y = (x.clone() * (2.0 * PI)).sin() + noise;

    // ------------------------- instantiate model & optimizer ------------------------
    let layers = LAYERS.windows(2)
        .map(|w| LinearConfig::new(w[0], w[1]).init(&device));
    let mut model = Mlp { layers: layers.collect() };
    let mut optimizer = AdamConfig::new().init();

    // ----------------------------------- training -----------------------------------
    for epoch in 1..=EPOCHS {
        let error = model.forward(x.clone()) - y.clone();
        let cost = error.powi_scalar(2).mean();
        if epoch == EPOCHS {
            println!("cost {:.2e}", cost.clone().into_scalar::<f32>());
        }
        let grads = GradientsParams::from_grads(cost.backward(), &model);
        model = optimizer.step(LR, model, grads);
    }
}
