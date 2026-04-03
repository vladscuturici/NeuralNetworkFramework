#include "LeakyReLU.hpp"

LeakyReLU::LeakyReLU()
	: last_input(1, 1) {
}

Matrix LeakyReLU::forward(const Matrix& input) {
	Matrix activation = Matrix(input.get_rows(), input.get_cols());
	for (int i = 0; i < input.get_rows(); i++) {
		for (int j = 0; j < input.get_cols(); j++) {
			double temp = input.get_value(i, j);
			if (temp < 0)
				temp = 0.01*temp;
			activation.set_value(i, j, temp);
		}
	}

	last_input = input;

	return activation;
}

Matrix LeakyReLU::backward(const Matrix& grad_output) {
	Matrix result = Matrix(last_input.get_rows(), last_input.get_cols());
	for (int i = 0; i < last_input.get_rows(); i++) {
		for (int j = 0; j < last_input.get_cols(); j++) {
			if (last_input.get_value(i, j) > 0) {
				result.set_value(i, j, grad_output.get_value(i, j));
			}
			else {
				result.set_value(i, j, grad_output.get_value(i, j) * 0.01);
			}
		}
	}
	return result;
}

void LeakyReLU::update_weights(Optimizer& optimizer) {
	//empty as LeakyReLU doesn't update any weights
}