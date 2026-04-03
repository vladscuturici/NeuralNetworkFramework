#include "SGD.hpp"

SGD::SGD(double learning_rate) : lr(learning_rate) {}

void SGD::update(Matrix& weights, Matrix& biases, const Matrix& grad_weights, const Matrix& grad_biases,
    Matrix& m_w, Matrix& v_w, Matrix& m_b, Matrix& v_b, bool& initialized, int& t) {
    weights = weights.element_wise_sub(grad_weights.element_wise_multiply_scalar(lr));
    biases = biases.element_wise_sub(grad_biases.element_wise_multiply_scalar(lr));
}