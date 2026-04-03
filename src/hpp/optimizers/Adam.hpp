#pragma once
#include "Optimizer.hpp"
#include <unordered_map>

class Adam : public Optimizer {
public:
    Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
    void update(Matrix& weights, Matrix& biases,
        const Matrix& grad_weights, const Matrix& grad_biases,
        Matrix& m_w, Matrix& v_w,
        Matrix& m_b, Matrix& v_b,
        bool& initialized, int& t) override;
private:
    double lr, beta1, beta2, epsilon;
};