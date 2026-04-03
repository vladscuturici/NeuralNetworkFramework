#pragma once
#include "Optimizer.hpp"

class SGD : public Optimizer {

public:

    SGD(double learning_rate);
    void update(Matrix& weights, Matrix& biases, const Matrix& grad_weights, const Matrix& grad_biases, Matrix& m_w, Matrix& v_w,
        Matrix& m_b, Matrix& v_b, bool& initialized, int& t) override;

private:

    double lr;

};