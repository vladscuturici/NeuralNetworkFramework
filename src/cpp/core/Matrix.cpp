#include "Matrix.hpp"

#include<vector>

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    for (int i = 0; i < rows; i++) {
        data.push_back(std::vector < double >(cols, 0.0));
    }
}

// We are using Mersenne Twister random, which produces a much better randomness than the rand() function in achieving weights that are not correlated
Matrix Matrix::random_values(int rows, int cols, double range_a, double range_b) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(range_a, range_b);

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = dist(rng);
    return result;
}

Matrix Matrix::he_init(int rows, int cols) {
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / rows));

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = dist(rng);
    return result;
}

Matrix Matrix::glorot_init(int rows, int cols) {
    static std::mt19937 rng(std::random_device{}());
    double limit = std::sqrt(6.0 / (rows + cols));
    std::uniform_real_distribution<double> dist(-limit, limit);

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = dist(rng);
    return result;
}

//getters and setters
bool Matrix::set_data(std::vector < std::vector < double >> new_data) {
    if (data.size() != new_data.size()) {
        std::cout << "Rows don't match\n";
        return false;
    }
    for (int i = 0; i < data.size(); i++)
        if (data[i].size() != new_data[i].size()) {
            std::cout << "Columns don't match\n";
            return false;
        }
    data = new_data;
    return true;
}

std::vector < std::vector < double >> Matrix::get_data(void) const {
    return data;
}

int Matrix::get_rows() const {
    return rows;
}

void Matrix::set_rows(int rows) {
    this->rows = rows;
}

int Matrix::get_cols() const {
    return cols;
}

void Matrix::set_cols(int cols) {
    this->cols = cols;
}

double Matrix::get_value(int i, int j) const {
    return data[i][j];
}

void Matrix::set_value(int i, int j, double value) {
    data[i][j] = value;
}

bool Matrix::has_same_shape(const Matrix& mat) const {
    if (data.size() != mat.data.size()) {
        return false;
    }
    for (int i = 0; i < data.size(); i++)
        if (data[i].size() != mat.data[i].size()) {
            return false;
        }
    return true;
}

//element wise operations
Matrix Matrix::element_wise_add(const Matrix& mat) const {
    Matrix result(rows, cols);

    if (!has_same_shape(mat)) {
        std::cout << "Shapes don't match, returning empty matrix\n";
        return result;
    }

    std::vector < std::vector < double >> new_data = data;

    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            new_data[i][j] += mat.data[i][j];
        }
    }
    result.set_data(new_data);
    return result;
}

Matrix Matrix::element_wise_sub(const Matrix& mat) const {
    Matrix result(rows, cols);

    if (!has_same_shape(mat)) {
        std::cout << "Shapes don't match, returning empty matrix\n";
        return result;
    }

    std::vector < std::vector < double >> new_data = data;

    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            new_data[i][j] -= mat.data[i][j];
        }
    }
    result.set_data(new_data);
    return result;
}

Matrix Matrix::element_wise_multiply(const Matrix& mat) const {
    Matrix result(rows, cols);

    if (!has_same_shape(mat)) {
        std::cout << "Shapes don't match, returning empty matrix\n";
        return result;
    }

    std::vector < std::vector < double >> new_data = data;

    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            new_data[i][j] *= mat.data[i][j];
        }
    }
    result.set_data(new_data);
    return result;
}

Matrix Matrix::element_wise_multiply_scalar(double x) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * x;
        }
    }
    return result;
}

//matrix operations
Matrix Matrix::dot(const Matrix& mat) const {
    Matrix result(rows, mat.cols);

    if (cols != mat.rows) {
        std::cout << "Shapes don't match, returning empty matrix\n";
        return result;
    }

    std::vector < std::vector < double >> new_data(rows, std::vector < double >(mat.cols, 0.0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            for (int k = 0; k < cols; k++) {
                new_data[i][j] += data[i][k] * mat.data[k][j];
            }
        }
    }

    result.set_data(new_data);
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.set_value(j, i, data[i][j]);
        }
    }

    return result;
}

void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}

Matrix Matrix::stack(const std::vector<Matrix>& matrices) {
    int rows = matrices.size();
    int cols = matrices[0].get_cols();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.set_value(i, j, matrices[i].get_value(0, j));
    return result;
}