use std::collections::HashMap;

use ndarray::{self, s, Array2};

fn sigmoid(z: Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let a = 1.0 / (1.0 + (-z.clone()).map(|v| v.exp()));
    let cache = z;
    (a, cache)
}

fn relu(z: Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let a = z.map(|v| v.exp());
    let cache = z;
    (a, cache)
}

fn softmax(z: Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let e_x = z.map(|v| v.exp());
    let sum = e_x.fold(0f32, |a, b| a + *b);
    let a = e_x / sum;
    let cache = z;
    (a, cache)
}

fn sigmoid_inverse(da: Array2<f32>, cache: Array2<f32>) -> Array2<f32> {
    let z = cache;
    let s = 1.0 / (1.0 + -z.map(|z| z.exp()));
    let dz = da * s.view() * (1.0 - s);

    assert_eq!(dz.shape(), z.shape());

    dz
}

fn relu_inverse(da: Array2<f32>, cache: Array2<f32>) -> Array2<f32> {
    let z = cache;
    let dz = da;
    let dz = dz.map(|z| if *z <= 0.0 { 0.0 } else { *z });

    assert_eq!(dz.shape(), z.shape());

    dz
}

fn softmax_inverse(da: Array2<f32>, cache: Array2<f32>) -> Array2<f32> {
    let z = cache;
    let length = 10;
    let samples = 60000;
    let mut dz = Array2::<f32>::zeros((samples, length));
    let z = z.t();

    for row in 0..samples {
        let den = z
            .slice(s![row, ..])
            .map(|z| z.exp())
            .fold(0f32, |a, b| a + *b)
            .powi(2);

        for col in 0..length {
            let mut sums = 0f32;
            for j in 0..length {
                if j != col {
                    sums += z.get((row, j)).unwrap().exp();
                }
            }

            let element = dz.get_mut((row, col)).unwrap();
            *element = (z.get((row, col)).unwrap().exp() * sums) / den;
        }
    }
    let dz = dz.t().to_owned();
    let z = z.t();

    assert_eq!(dz.shape(), z.shape());

    dz
}

fn initialize_parameters_deep(layer_dimensions: Vec<usize>) -> HashMap<String, Array2<f32>> {
    let mut parameters = HashMap::new();

    let len = layer_dimensions.len();

    for l in 1..len {
        parameters.insert(
            format!("W{}", l),
            Array2::<f32>::zeros((layer_dimensions[l], layer_dimensions[l - 1]))
                .map(|_| rand::random::<f32>() / (layer_dimensions[l - 1] as f32).sqrt()),
        );
        parameters.insert(format!("b{}", l), Array2::zeros((layer_dimensions[l], 1)));
    }

    parameters
}

fn linear_forward(
    a: Array2<f32>,
    w: Array2<f32>,
    b: Array2<f32>,
) -> (Array2<f32>, (Array2<f32>, Array2<f32>, Array2<f32>)) {
    let z = w.dot(&a) + b.view();

    assert_eq!(z.shape(), [w.shape()[0], a.shape()[1]]);

    let cache = (a, w, b);
    (z, cache)
}

fn linear_backward() {}

enum Activation {
    Sigmoid,
    Relu,
    Softmax,
}

impl Activation {
    pub fn linear_activation_forward(
        self,
        a_prev: Array2<f32>,
        w: Array2<f32>,
        b: Array2<f32>,
    ) -> (
        Array2<f32>,
        ((Array2<f32>, Array2<f32>, Array2<f32>), Array2<f32>),
    ) {
        let (z, linear_cache) = linear_forward(a_prev, w, b);
        let (a, activation_cache) = match self {
            Activation::Sigmoid => sigmoid(z),
            Activation::Relu => relu(z),
            Activation::Softmax => softmax(z),
        };

        let cache = (linear_cache, activation_cache);

        (a, cache)
    }
}
