#pragma once

#include<iostream>
#include "Matrix.hpp" 
#include "Optimizer.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;

    virtual Matrix backward(const Matrix& grad_output) = 0;

    virtual void update_weights(Optimizer& optimizer) = 0;

    virtual void set_training(bool training) {}

    virtual void zero_gradients() {}
};

