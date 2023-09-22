use rand::random;

mod autodiff;

use autodiff::PtrVar;

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
pub struct NeuralNetwork {
    input_num: usize,
    non_input_layers: Vec<NonInputLayer>,
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

#[cfg(test)]
mod tests {
    use crate::NeuralNetwork;

    #[test]
    fn nn_test() {
        const I: usize = 2;
        const O: usize = 1;
        let data: Vec<[f64; I + O]> = vec![[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        let lr = 1.0;

        let mut nn = NeuralNetwork::init(&[I, 5, O]);

        for _ in 0..1000 {
            let loss = nn.calc_loss::<{ I + O }>(&data);
            loss.calc_grad();
            nn.non_input_layers.iter_mut().for_each(|nil| {
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
        let r = nn.forward(&[1., 0.]); //1
        println!("{r:?}");
        let r = nn.forward(&[1., 1.]); //0
        println!("{r:?}");
        let r = nn.forward(&[0., 0.]); //0
        println!("{r:?}");
        let r = nn.forward(&[0., 1.]); //1
        println!("{r:?}");
    }
}
