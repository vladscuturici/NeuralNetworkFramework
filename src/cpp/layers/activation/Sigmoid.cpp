#include "Sigmoid.hpp"

Sigmoid::Sigmoid()
	: last_input(1, 1) {
}

Matrix Sigmoid::forward(const Matrix& input) {
	Matrix activation = Matrix(input.get_rows(), input.get_cols());
	for (int i = 0; i < input.get_rows(); i++) {
		for (int j = 0; j < input.get_cols(); j++) {
			double temp = input.get_value(i, j);
			temp = 1.0 / (1.0 + std::exp(-temp));
			activation.set_value(i, j, temp);
		}
	}

	last_input = input;

	return activation;
}

Matrix Sigmoid::backward(const Matrix& grad_output) {
	Matrix result(last_input.get_rows(), last_input.get_cols());

	for (int i = 0; i < last_input.get_rows(); i++) {
		for (int j = 0; j < last_input.get_cols(); j++) {
			double sig = 1.0 / (1.0 + std::exp(-last_input.get_value(i, j)));
			double grad = grad_output.get_value(i, j) * sig * (1.0 - sig);
			result.set_value(i, j, grad);
		}
	}

	return result;
}


void Sigmoid::update_weights(Optimizer& optimizer) {
	//empty as Sigmoid doesn't update any weights
}