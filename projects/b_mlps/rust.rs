struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f64 / (u32::MAX as f64)
    }
    fn randn(&mut self) -> f64 {
        let u = self.next().max(1e-10);
        let v = self.next();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

struct Layer {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
    last_input:  Vec<f64>,
    last_preact: Vec<f64>,
}

impl Layer {
    fn new(in_n: usize, out_n: usize, rng: &mut Lcg) -> Self {
        let scale = (2.0 / in_n as f64).sqrt();
        Layer {
            w: (0..out_n).map(|_| (0..in_n).map(|_| rng.randn() * scale).collect()).collect(),
            b: vec![0.0; out_n],
            last_input: vec![],
            last_preact: vec![],
        }
    }
    fn forward(&mut self, x: &[f64]) -> Vec<f64> {
        self.last_input = x.to_vec();
        let pre: Vec<f64> = self.w.iter().zip(self.b.iter())
            .map(|(row, &bias)| row.iter().zip(x).map(|(w, xi)| w * xi).sum::f64() + bias)
            .collect();
        self.last_preact = pre.clone();
        pre.iter().map(|&z| z.tanh()).collect()
    }
}

fn backward(layer: &mut Layer, d_out: &[f64], lr: f64) -> Vec<f64> {
    let d_pre: Vec<f64> = layer.last_preact.iter().zip(d_out)
        .map(|(&z, &g)| g * (1.0 - z.tanh().powi(2)))
        .collect();
    let mut d_in = vec![0.0; layer.last_input.len()];
    for (i, row) in layer.w.iter().enumerate() {
        for (j, &wij) in row.iter().enumerate() { d_in[j] += wij * d_pre[i]; }
    }
    for (i, row) in layer.w.iter_mut().enumerate() {
        for (j, wij) in row.iter_mut().enumerate() {
            *wij -= lr * d_pre[i] * layer.last_input[j];
        }
        layer.b[i] -= lr * d_pre[i];
    }
    d_in
}

fn fwd(l1: &mut Layer, l2: &mut Layer, l3: &mut Layer, x: f64) -> f64 {
    l3.forward(&l2.forward(&l1.forward(&[x])))[0]
}

fn main() {
    let mut rng = Lcg(42);
    let mut l1 = Layer::new(1,  32, &mut rng);
    let mut l2 = Layer::new(32, 32, &mut rng);
    let mut l3 = Layer::new(32, 1,  &mut rng);

    let pi = std::f64::consts::PI;
    let data: Vec<_> = (0..200)
        .map(|_| { let x = rng.next() * 2.0 * pi - pi; (x, x.sin()) })
        .collect();

    for epoch in 0..5000 {
        let mut loss = 0.0;
        for &(x, y) in &data {
            let pred = fwd(&mut l1, &mut l2, &mut l3, x);
            loss += (pred - y).powi(2);
            let g3 = backward(&mut l3, &[2.0 * (pred - y)], 1e-3);
            let g2 = backward(&mut l2, &g3, 1e-3);
            backward(&mut l1, &g2, 1e-3);
        }
        if epoch % 500 == 0 {
            println!("epoch {epoch:5} | MSE {:.6}", loss / data.len() as f64);
        }
    }
    for &x in &[0.0, pi/2.0, pi] {
        println!("sin({x:.4}) ≈ {:.4}  (true: {:.4})", fwd(&mut l1, &mut l2, &mut l3, x), x.sin());
    }
}
