#include "Tanh.hpp"
#include <cmath> 

Tanh::Tanh()
	: last_input(1, 1) {
}

double tanh(double x) {
	double result = (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));

	return result;
}

Matrix Tanh::forward(const Matrix& input) {
	Matrix activation = Matrix(input.get_rows(), input.get_cols());
	for (int i = 0; i < input.get_rows(); i++) {
		for (int j = 0; j < input.get_cols(); j++) {
			double temp = tanh(input.get_value(i, j));
			activation.set_value(i, j, temp);
		}
	}

	last_input = input;

	return activation;
}

Matrix Tanh::backward(const Matrix& grad_output) {
	Matrix result(last_input.get_rows(), last_input.get_cols());

	for (int i = 0; i < last_input.get_rows(); i++) {
		for (int j = 0; j < last_input.get_cols(); j++) {
			double grad = 1.0 - tanh(last_input.get_value(i, j)) * tanh(last_input.get_value(i, j));
			result.set_value(i, j, grad * grad_output.get_value(i, j));
		}
	}

	return result;
}


void Tanh::update_weights(Optimizer& optimizer) {
	//empty as Tanh doesn't update any weights
}