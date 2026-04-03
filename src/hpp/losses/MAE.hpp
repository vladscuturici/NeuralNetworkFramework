#pragma once
#include "Loss.hpp"
#include "Matrix.hpp"

class MAE : public Loss {
public:
	double loss(const Matrix& pred, const Matrix& ground_truth) override;
	Matrix backward(const Matrix& pred, const Matrix& ground_truth) override;
};