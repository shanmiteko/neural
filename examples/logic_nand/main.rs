use neural::NeuralNetwork;

fn main() {
    const I: usize = 2;
    const O: usize = 1;
    let data: Vec<[f64; I + O]> = vec![[0., 0., 1.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
    let lr = 1.0;

    let mut nn = NeuralNetwork::init(&[I, 5, 5, O]);

    for _ in 0..10000 {
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
    let p1 = nn.forward(&[1., 0.]); //1
    let p2 = nn.forward(&[1., 1.]); //0
    let p3 = nn.forward(&[0., 0.]); //1
    let p4 = nn.forward(&[0., 1.]); //1
    let rate = 1.
        - ((p1.get(0).unwrap().value() - 1.).powi(2)
            + (p2.get(0).unwrap().value() - 0.).powi(2)
            + (p3.get(0).unwrap().value() - 1.).powi(2)
            + (p4.get(0).unwrap().value() - 1.).powi(2))
        .sqrt();
    assert!(rate > 0.);

    println!("准确率: {}%", rate * 100.)
}
