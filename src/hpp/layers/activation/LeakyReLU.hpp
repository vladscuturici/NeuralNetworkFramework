#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

class LeakyReLU : public Layer {
public:
	Matrix last_input;
	LeakyReLU();
	Matrix forward(const Matrix& input) override;
	Matrix backward(const Matrix& grad_output) override;
	void update_weights(Optimizer& optimizer) override;
};