#pragma once

#include <vector>
#include <iostream>
#include <random>

class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<double>> data;

    Matrix(int rows, int cols);

    static Matrix random_values(int rows, int cols, double range_a, double range_b);
    static Matrix he_init(int rows, int cols);
    static Matrix glorot_init(int rows, int cols);

    int get_rows() const;
    int get_cols() const;
    double get_value(int i, int j) const;
    std::vector<std::vector<double>> get_data() const;

    void set_value(int i, int j, double value);
    bool set_data(std::vector<std::vector<double>> new_data);
    void set_rows(int rows);
    void set_cols(int cols);

    void print() const;
    bool has_same_shape(const Matrix& mat) const;

    Matrix element_wise_add(const Matrix& mat) const;
    Matrix element_wise_sub(const Matrix& mat) const;
    Matrix element_wise_multiply(const Matrix& mat) const;
    Matrix element_wise_multiply_scalar(double x) const;

    Matrix dot(const Matrix& mat) const;
    Matrix transpose() const;

    static Matrix stack(const std::vector<Matrix>& matrices);
};