#pragma once

#include<iostream>
#include "Matrix.hpp"

class Loss {
public:
    virtual ~Loss() = default;

    virtual double loss(const Matrix& pred, const Matrix& ground_truth) = 0;

    virtual Matrix backward(const Matrix& pred, const Matrix& ground_truth) = 0;
};