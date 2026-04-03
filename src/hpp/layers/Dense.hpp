#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
enum class WeightInit { Random, He, Glorot };

class Dense : public Layer {
public:
    Matrix weights;
    Matrix biases;
    Matrix last_input;
    Matrix grad_weights;
    Matrix grad_biases;

    Dense(int input_size, int output_size, WeightInit init = WeightInit::He, double weight_init_a = -1.0, double weight_init_b = -1.0);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_weights(Optimizer& optimizer) override;
    void zero_gradients() override;

private:
    Matrix m_weights, v_weights;
    Matrix m_biases, v_biases;
    int    opt_t = 0;
    bool   opt_initialized = false;
};

    