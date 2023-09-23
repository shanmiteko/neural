use rand::random;

use crate::ad::PtrVar;

#[derive(Debug)]
struct Neuron {
    weights: Vec<PtrVar>,
    bias: PtrVar,
}

#[derive(Debug)]
struct NonInputLayer {
    neurons: Vec<Neuron>,
}

#[derive(Debug)]
pub struct NeuralNetwork<const I: usize, const O: usize> {
    non_input_layers: Vec<NonInputLayer>,
}

impl<const I: usize, const O: usize> NeuralNetwork<I, O> {
    /// 初始化
    ///
    /// 神经网络的形状
    /// `NeuralNetwork::<2, 1>::init(&[2])`
    /// ```txt
    ///               .  .
    /// [2, 2, 1] ->        .
    ///               .  .
    /// ```
    pub fn init(hidden: &[usize]) -> Self {
        let mut non_input_layers: Vec<NonInputLayer> = Vec::new();
        let shape = [&[I], hidden, &[O]].concat();
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
        Self { non_input_layers }
    }

    fn act_fn(x: PtrVar) -> PtrVar {
        PtrVar::new(1.)
            / (PtrVar::new(1.) + PtrVar::new(std::f64::consts::E).pow(PtrVar::new(0.) - x))
    }

    pub fn forward(&self, input: &[f64; I]) -> Vec<PtrVar> {
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

    fn calc_loss(&self, data: &[([f64; I], [f64; O])]) -> PtrVar {
        data.iter().fold(PtrVar::new(0.), |loss, d| {
            loss + self
                .forward(&d.0)
                .iter()
                .zip(&d.1.iter().map(|x| PtrVar::new(*x)).collect::<Vec<PtrVar>>())
                .fold(PtrVar::new(0.), |distance, pred_real| {
                    distance + (pred_real.0.clone() - pred_real.1.clone()).pow(PtrVar::new(2.))
                })
        })
    }

    pub fn train(&mut self, data: &[([f64; I], [f64; O])], times: usize, lr: f64) {
        for _ in 0..times {
            let loss = self.calc_loss(data);
            loss.calc_grad();
            self.non_input_layers.iter_mut().for_each(|nil| {
                nil.neurons.iter_mut().for_each(|n| {
                    n.weights.iter_mut().for_each(|w| {
                        w.value_mut(w.value() - w.grad() * lr);
                        w.grad_mut(0.);
                    });
                    n.bias.value_mut(n.bias.value() - n.bias.grad() * lr);
                    n.bias.grad_mut(0.);
                })
            });
        }
    }
}
