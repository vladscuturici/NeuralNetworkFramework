#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

class BatchNorm : public Layer {

public:

    BatchNorm(int size, double epsilon = 1e-8, double momentum = 0.1);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_weights(Optimizer& optimizer) override;
    void set_training(bool training) override;
    void zero_gradients() override;

private:

    int size;
    double epsilon;
    double momentum;
    bool is_training = true;

    Matrix gamma;
    Matrix beta;
    Matrix grad_gamma;
    Matrix grad_beta;

    Matrix last_input_norm;
    Matrix last_mean;
    Matrix last_var;

    Matrix running_mean;
    Matrix running_var;

    Matrix m_weights, v_weights;
    Matrix m_biases, v_biases;
    int    opt_t = 0;
    bool   opt_initialized = false;
};