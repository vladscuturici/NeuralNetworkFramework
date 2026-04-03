#include "Adam.hpp"
#include <cmath>

Adam::Adam(double learning_rate, double beta1, double beta2, double epsilon)
    : lr(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon)
{}

void Adam::update(Matrix& weights, Matrix& biases,
    const Matrix& grad_weights, const Matrix& grad_biases,
    Matrix& m_w, Matrix& v_w, Matrix& m_b, Matrix& v_b,
    bool& initialized, int& t) {

    if (!initialized) {
        m_w = Matrix(weights.get_rows(), weights.get_cols());
        v_w = Matrix(weights.get_rows(), weights.get_cols());
        m_b = Matrix(biases.get_rows(), biases.get_cols());
        v_b = Matrix(biases.get_rows(), biases.get_cols());
        initialized = true;
    }

    t++;
    double lr_t = lr * std::sqrt(1.0 - std::pow(beta2, t))
        / (1.0 - std::pow(beta1, t));

    for (int i = 0; i < weights.get_rows(); i++) {
        for (int j = 0; j < weights.get_cols(); j++) {
            double g = grad_weights.get_value(i, j);
            double m = beta1 * m_w.get_value(i, j) + (1 - beta1) * g;
            double v = beta2 * v_w.get_value(i, j) + (1 - beta2) * g * g;
            m_w.set_value(i, j, m);
            v_w.set_value(i, j, v);
            weights.set_value(i, j, weights.get_value(i, j) - lr_t * m / (std::sqrt(v) + epsilon));
        }
    }

    for (int i = 0; i < biases.get_rows(); i++) {
        for (int j = 0; j < biases.get_cols(); j++) {
            double g = grad_biases.get_value(i, j);
            double m = beta1 * m_b.get_value(i, j) + (1 - beta1) * g;
            double v = beta2 * v_b.get_value(i, j) + (1 - beta2) * g * g;
            m_b.set_value(i, j, m);
            v_b.set_value(i, j, v);
            biases.set_value(i, j, biases.get_value(i, j) - lr_t * m / (std::sqrt(v) + epsilon));
        }
    }
}