#include "ReLU.hpp"

ReLU::ReLU()
	: last_input(1, 1) {
}

Matrix ReLU::forward(const Matrix& input){
	Matrix activation = Matrix(input.get_rows(), input.get_cols());
	for (int i = 0; i < input.get_rows(); i++) {
		for (int j = 0; j < input.get_cols(); j++) {
			double temp = input.get_value(i, j);
			if (temp < 0)
				temp = 0;
			activation.set_value(i, j, temp);
		}
	}

	last_input = input;

	return activation;
}

Matrix ReLU::backward(const Matrix& grad_output) {
	Matrix result = Matrix(last_input.get_rows(), last_input.get_cols());
	for (int i = 0; i < last_input.get_rows(); i++) {
		for (int j = 0; j < last_input.get_cols(); j++) {
			if (last_input.get_value(i, j) > 0) {
				result.set_value(i, j, grad_output.get_value(i, j));
			}
			else {
				result.set_value(i, j, 0);
			}
		}
	}
	return result;
}

void ReLU::update_weights(Optimizer& optimizer) {
	//empty as ReLu doesn't update any weights
}