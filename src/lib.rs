use rand::random;

mod autodiff;

use autodiff::PtrVar;

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<PtrVar>,
    pub bias: PtrVar,
}

#[derive(Debug)]
pub struct NonInputLayer {
    pub neurons: Vec<Neuron>,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    input_num: usize,
    pub non_input_layers: Vec<NonInputLayer>,
}

impl NeuralNetwork {
    /// 初始化
    ///
    /// 神经网络的形状
    /// ```txt
    ///               .  .
    /// [2, 2, 1] ->        .
    ///               .  .
    /// ```
    pub fn init(shape: &[usize]) -> Self {
        let input_num = shape[0];
        let mut non_input_layers: Vec<NonInputLayer> = Vec::new();
        for i in 1..shape.len() {
            let mut neurons: Vec<Neuron> = Vec::new();
            for _ in 0..shape[i] {
                let mut weights: Vec<PtrVar> = Vec::new();
                for _ in 0..shape[i - 1] {
                    weights.push(random::<f64>().into())
                }
                neurons.push(Neuron {
                    weights,
                    bias: random::<f64>().into(),
                })
            }
            non_input_layers.push(NonInputLayer { neurons });
        }
        Self {
            input_num,
            non_input_layers,
        }
    }

    fn act_fn(x: PtrVar) -> PtrVar {
        PtrVar::new(1.)
            / (PtrVar::new(1.) + PtrVar::new(std::f64::consts::E).pow(PtrVar::new(0.) - x))
    }

    pub fn forward(&self, input: &[f64]) -> Vec<PtrVar> {
        assert_eq!(input.len(), self.input_num, "input error!!!");
        let mut tmp_input: Vec<PtrVar> = input.iter().map(|x| PtrVar::new(*x)).collect();
        let mut tmp_output: Vec<PtrVar> = Vec::new();
        for nilayer in &self.non_input_layers {
            tmp_output.clear();
            for neuron in &nilayer.neurons {
                tmp_output.push(Self::act_fn(
                    neuron
                        .weights
                        .iter()
                        .zip(&tmp_input)
                        .fold(PtrVar::new(0.), |acc, x| acc + x.0.clone() * x.1.clone())
                        + neuron.bias.clone(),
                ))
            }
            tmp_input = tmp_output.clone();
        }

        tmp_output
    }

    pub fn calc_loss<const N: usize>(&self, data: &[[f64; N]]) -> PtrVar {
        data.iter().fold(PtrVar::new(0.), |loss, d| {
            loss + self
                .forward(&d[0..self.input_num])
                .iter()
                .zip(
                    &d[self.input_num..]
                        .iter()
                        .map(|x| PtrVar::new(*x))
                        .collect::<Vec<PtrVar>>(),
                )
                .fold(PtrVar::new(0.), |distance, pred_real| {
                    distance + (pred_real.0.clone() - pred_real.1.clone()).pow(PtrVar::new(2.))
                })
        })
    }
}
