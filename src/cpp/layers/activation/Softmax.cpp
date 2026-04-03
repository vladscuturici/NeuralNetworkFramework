#include "Softmax.hpp"
#include <cmath>

Softmax::Softmax()
	: last_input(1, 1), last_output(1, 1) {
}

Matrix Softmax::forward(const Matrix& input) {
    int rows = input.get_rows();
    int cols = input.get_cols();

    Matrix activation(rows, cols);

    for (int i = 0; i < rows; i++) {

        double max_val = input.get_value(i, 0);
        for (int j = 1; j < cols; j++) {
            double v = input.get_value(i, j);
            if (v > max_val)
                max_val = v;
        }

        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            double e = std::exp(input.get_value(i, j) - max_val);
            activation.set_value(i, j, e);
            sum += e;
        }

        for (int j = 0; j < cols; j++) {
            double norm = activation.get_value(i, j) / sum;
            activation.set_value(i, j, norm);
        }
    }

    last_input = input;
    last_output = activation;

    return activation;
}


Matrix Softmax::backward(const Matrix& grad_output) {
    int rows = last_output.get_rows();
    int cols = last_output.get_cols();

    Matrix grad_input(rows, cols);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < cols; j++) {

            double s_j = last_output.get_value(i, j);

            double sum = 0.0;
            for (int k = 0; k < cols; k++) {

                double s_k = last_output.get_value(i, k);
                double jac;

                if (j == k)
                    jac = s_j * (1 - s_j);
                else
                    jac = -s_j * s_k;

                sum += jac * grad_output.get_value(i, k);
            }

            grad_input.set_value(i, j, sum);
        }
    }

    return grad_input;
}

void Softmax::update_weights(Optimizer& optimizer) {
    //empty as Softmax doesn't update any weights
}


