#include "BCE.hpp"
#include<cmath>

double BCE::loss(const Matrix& pred, const Matrix& ground_truth) {
    double sum = 0;
    const double epsilon = 1e-7;
    for (int i = 0; i < pred.get_rows(); i++) {
        for (int j = 0; j < pred.get_cols(); j++) {
            double p = pred.get_value(i, j);
            double y = ground_truth.get_value(i, j);

            p = std::max(epsilon, std::min(1.0 - epsilon, p));
            sum += -(y * std::log(p) + (1 - y) * std::log(1 - p));
        }
    }
    return sum / (pred.get_rows() * pred.get_cols());
}

Matrix BCE::backward(const Matrix& pred, const Matrix& ground_truth) {
    Matrix result(pred.get_rows(), pred.get_cols());
    const double epsilon = 1e-7;
    double n = pred.get_rows() * pred.get_cols();

    for (int i = 0; i < pred.get_rows(); i++) {
        for (int j = 0; j < pred.get_cols(); j++) {
            double p = pred.get_value(i, j);
            double y = ground_truth.get_value(i, j);

            p = std::max(epsilon, std::min(1.0 - epsilon, p));
            double grad = (p - y) / (p * (1.0 - p));

            result.set_value(i, j, grad / n);
        }
    }

    return result;
}
