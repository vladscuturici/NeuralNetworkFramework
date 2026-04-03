#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

class Dropout : public Layer {

public:

    Dropout(double rate);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_weights(Optimizer& optimizer) override;
    void set_training(bool training) override;

private:

    double rate;
    bool is_training = true;
    Matrix mask;
};