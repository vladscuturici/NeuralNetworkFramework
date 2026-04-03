#include "MAE.hpp"
#include<cmath>

double MAE::loss(const Matrix& pred, const Matrix& ground_truth) {
	double sum = 0;
	for (int i = 0; i < pred.get_rows(); i++) {
		for (int j = 0; j < pred.get_cols(); j++) {
			sum += std::abs(pred.get_value(i, j) - ground_truth.get_value(i, j));
		}
	}

	return sum / (pred.get_rows() * pred.get_cols());
}

Matrix MAE::backward(const Matrix& pred, const Matrix& ground_truth) {
    Matrix result(pred.get_rows(), pred.get_cols());
    double n = pred.get_rows() * pred.get_cols();

    for (int i = 0; i < pred.get_rows(); i++) {
        for (int j = 0; j < pred.get_cols(); j++) {
            double diff = pred.get_value(i, j) - ground_truth.get_value(i, j);
            if (diff > 0)
                result.set_value(i, j, 1.0 / n);
            else if (diff < 0)
                result.set_value(i, j, -1.0 / n);
            else
                result.set_value(i, j, 0); 
        }
    }

    return result;
}
