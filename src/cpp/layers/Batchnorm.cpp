#include "BatchNorm.hpp"
#include <cmath>

BatchNorm::BatchNorm(int size, double epsilon, double momentum)
    : size(size), epsilon(epsilon), momentum(momentum), is_training(true),
    gamma(1, size), beta(1, size),
    grad_gamma(1, size), grad_beta(1, size),
    last_input_norm(1, size), last_mean(1, size), last_var(1, size),
    running_mean(1, size), running_var(1, size),
    m_weights(1, 1), v_weights(1, 1),
    m_biases(1, 1), v_biases(1, 1)
{
    for (int j = 0; j < size; j++)
        gamma.set_value(0, j, 1.0);
}

Matrix BatchNorm::forward(const Matrix& input) {
    int rows = input.get_rows();
    int cols = input.get_cols();
    Matrix output(rows, cols);

    if (is_training) {

        for (int j = 0; j < cols; j++) {
            double mean = 0.0;
            for (int i = 0; i < rows; i++)
                mean += input.get_value(i, j);
            mean /= rows;
            last_mean.set_value(0, j, mean);

            double var = 0.0;
            for (int i = 0; i < rows; i++) {
                double diff = input.get_value(i, j) - mean;
                var += diff * diff;
            }
            var /= rows;
            last_var.set_value(0, j, var);

            double std_inv = 1.0 / std::sqrt(var + epsilon);
            for (int i = 0; i < rows; i++) {
                double norm = (input.get_value(i, j) - mean) * std_inv;
                last_input_norm.set_value(i, j, norm);
                output.set_value(i, j, gamma.get_value(0, j) * norm
                    + beta.get_value(0, j));
            }

            double rm = running_mean.get_value(0, j);
            double rv = running_var.get_value(0, j);
            running_mean.set_value(0, j, (1 - momentum) * rm + momentum * mean);
            running_var.set_value(0, j, (1 - momentum) * rv + momentum * var);
        }
    }
    else {

        for (int j = 0; j < cols; j++) {
            double mean = running_mean.get_value(0, j);
            double var = running_var.get_value(0, j);
            double std_inv = 1.0 / std::sqrt(var + epsilon);
            for (int i = 0; i < rows; i++) {
                double norm = (input.get_value(i, j) - mean) * std_inv;
                output.set_value(i, j, gamma.get_value(0, j) * norm
                    + beta.get_value(0, j));
            }
        }
    }
    return output;
}

Matrix BatchNorm::backward(const Matrix& grad_output) {
    int rows = grad_output.get_rows();
    int cols = grad_output.get_cols();
    Matrix grad_input(rows, cols);

    for (int j = 0; j < cols; j++) {
        double var = last_var.get_value(0, j);
        double std_inv = 1.0 / std::sqrt(var + epsilon);
        double dvar = 0.0;
        double dmean = 0.0;

        double dg = 0.0, db = 0.0;
        for (int i = 0; i < rows; i++) {
            dg += grad_output.get_value(i, j) * last_input_norm.get_value(i, j);
            db += grad_output.get_value(i, j);
        }
        grad_gamma.set_value(0, j, dg);
        grad_beta.set_value(0, j, db);

        for (int i = 0; i < rows; i++) {
            double dx_norm = grad_output.get_value(i, j) * gamma.get_value(0, j);
            dvar += dx_norm * last_input_norm.get_value(i, j);
            dmean += dx_norm;
        }
        dvar *= -0.5 * std::pow(var + epsilon, -1.5);
        dmean *= -std_inv;

        for (int i = 0; i < rows; i++) {
            double dx_norm = grad_output.get_value(i, j) * gamma.get_value(0, j);
            double dx = dx_norm * std_inv
                + dvar * 2.0 * last_input_norm.get_value(i, j) / rows
                + dmean / rows;
            grad_input.set_value(i, j, dx);
        }
    }
    return grad_input;
}

void BatchNorm::update_weights(Optimizer& optimizer) {

    optimizer.update(gamma, beta, grad_gamma, grad_beta,
        m_weights, v_weights, m_biases, v_biases,
        opt_initialized, opt_t);
}

void BatchNorm::set_training(bool training) {
    is_training = training;
}

void BatchNorm::zero_gradients() {
    grad_gamma = Matrix(1, size);
    grad_beta = Matrix(1, size);
}