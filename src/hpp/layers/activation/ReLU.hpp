#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

class ReLU : public Layer {
public:
	Matrix last_input;
	ReLU();
	Matrix forward(const Matrix& input) override;
	Matrix backward(const Matrix& grad_output) override;
	void update_weights(Optimizer& optimizer) override;
};