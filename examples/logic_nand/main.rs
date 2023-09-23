use neural::nn::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::<2, 1>::init(&[3, 3]);

    nn.train(
        &[
            ([0., 0.], [1.]),
            ([0., 1.], [1.]),
            ([1., 0.], [1.]),
            ([1., 1.], [0.]),
        ],
        1000,
        1.0,
    );

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

    println!("准确率: {}%", rate * 100.)
}
