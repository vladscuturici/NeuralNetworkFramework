#include "Dense.hpp"

static Matrix init_weights(int rows, int cols, WeightInit init, double a, double b) {
    switch (init) {
        case WeightInit::He:     return Matrix::he_init(rows, cols);
        case WeightInit::Glorot: return Matrix::glorot_init(rows, cols);
        case WeightInit::Random: return Matrix::random_values(rows, cols, a, b);
    }
    return Matrix::he_init(rows, cols);
}

Dense::Dense(int input_size, int output_size, WeightInit init, double weight_init_a, double weight_init_b)
    : weights(init_weights(input_size, output_size, init, weight_init_a, weight_init_b)),
    biases(1, output_size),
    last_input(1, 1),
    grad_weights(input_size, output_size),
    grad_biases(1, output_size),
    m_weights(1, 1), v_weights(1, 1),
    m_biases(1, 1), v_biases(1, 1)
{
}

Matrix Dense::forward(const Matrix& input) {
    last_input = input;
    //std::cout << "weights of dense:\n";
    //weights.print();
    return input.dot(weights).element_wise_add(biases);
}

Matrix Dense::backward(const Matrix& grad_output) {
    Matrix gradient_weights = last_input.transpose().dot(grad_output);
    Matrix gradient_biases(1, grad_output.get_cols());
    for (int i = 0; i < grad_output.get_rows(); i++) {
        for (int j = 0; j < grad_output.get_cols(); j++) {
            gradient_biases.set_value(0, j, gradient_biases.get_value(0, j) + grad_output.get_value(i, j));
        }
    }
    grad_weights = gradient_weights;
    grad_biases = gradient_biases;
    /*std::cout << "grad weights:\n";
    grad_weights.print();
    std::cout << "grad biases:\n";
    grad_biases.print();*/
    return grad_output.dot(weights.transpose());
}

void Dense::zero_gradients() {
    grad_weights = Matrix(weights.get_rows(), weights.get_cols());
    grad_biases = Matrix(1, biases.get_cols());
}

void Dense::update_weights(Optimizer& optimizer) {
    optimizer.update(weights, biases,
        grad_weights, grad_biases,
        m_weights, v_weights,
        m_biases, v_biases,
        opt_initialized, opt_t);
}
