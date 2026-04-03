#include "MSE.hpp"

double MSE::loss(const Matrix& pred, const Matrix& ground_truth) {
	double sum = 0;
	for (int i = 0; i < pred.get_rows(); i++) {
		for (int j = 0; j < pred.get_cols(); j++) {
			sum += ((pred.get_value(i, j) - ground_truth.get_value(i, j)) * (pred.get_value(i, j) - ground_truth.get_value(i, j)));
		}
	}

	return sum / (pred.get_rows() * pred.get_cols());
}

Matrix MSE::backward(const Matrix& pred, const Matrix& ground_truth) {
	Matrix result = Matrix(pred.get_rows(), pred.get_cols());

	for (int i = 0; i < pred.get_rows(); i++) {
		for (int j = 0; j < pred.get_cols(); j++) {
			result.set_value(i, j, 2 * (pred.get_value(i, j) - ground_truth.get_value(i, j)) / (pred.get_rows() * pred.get_cols()));
		}
	}

	return result;
}