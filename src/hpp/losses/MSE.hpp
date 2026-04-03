#pragma once
#include "Loss.hpp"
#include "Matrix.hpp"

class MSE : public Loss {
public:
	double loss(const Matrix& pred, const Matrix& ground_truth) override;
	Matrix backward(const Matrix& pred, const Matrix& ground_truth) override;
};