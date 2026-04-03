#pragma once
#include "Matrix.hpp"

class Optimizer {

public:

    virtual ~Optimizer() = default;
    virtual void update(Matrix& weights, Matrix& biases, const Matrix& grad_weights, const Matrix& grad_biases,
        Matrix& m_w, Matrix& v_w, Matrix& m_b, Matrix& v_b, bool& initialized, int& t) = 0;

};