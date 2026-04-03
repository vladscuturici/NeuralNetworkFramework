#include "Dropout.hpp"
#include <random>

static std::mt19937 dropout_rng(std::random_device{}());

Dropout::Dropout(double rate)
    : rate(rate), is_training(true), mask(1, 1)
{}

Matrix Dropout::forward(const Matrix& input) {

    if (!is_training) {
        mask = Matrix(input.get_rows(), input.get_cols());
        return input;
    }

    mask = Matrix(input.get_rows(), input.get_cols());
    Matrix output(input.get_rows(), input.get_cols());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double scale = 1.0 / (1.0 - rate);

    for (int i = 0; i < input.get_rows(); i++) {
        for (int j = 0; j < input.get_cols(); j++) {
            double keep = (dist(dropout_rng) >= rate) ? 1.0 : 0.0;
            mask.set_value(i, j, keep);
            output.set_value(i, j, input.get_value(i, j) * keep * scale);
        }
    }
    return output;
}

Matrix Dropout::backward(const Matrix& grad_output) {

    if (!is_training)
        return grad_output;

    Matrix result(grad_output.get_rows(), grad_output.get_cols());
    double scale = 1.0 / (1.0 - rate);

    for (int i = 0; i < grad_output.get_rows(); i++) {
        for (int j = 0; j < grad_output.get_cols(); j++) {
            result.set_value(i, j, grad_output.get_value(i, j) * mask.get_value(i, j) * scale);
        }
    }
    return result;
}

void Dropout::update_weights(Optimizer& optimizer) {
    //empty as Dropout doesn't update any weights
}

void Dropout::set_training(bool training) {
    is_training = training;
}