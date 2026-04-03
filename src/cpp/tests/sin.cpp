//#define _USE_MATH_DEFINES
//
//#include<iostream>
//#include<vector>
//#include<cmath>
//#include "Matrix.hpp"
//#include "Layer.hpp"
//#include "Dense.hpp"
//#include "Network.hpp"
//#include "MSE.hpp"
//#include "ReLU.hpp"
//#include "Sigmoid.hpp"
//#include "Optimizer.hpp"
//#include "SGD.hpp"
//#include "Adam.hpp"
//#include "Tanh.hpp"
//
//int main() {
//    const int N = 100;
//
//    double x_vals[N];
//    double y_vals[N];
//
//    for (int i = 0; i < N; i++) {
//        double x = (2 * M_PI) * i / N;
//        x_vals[i] = x;
//        y_vals[i] = sin(x);
//    }
//
//    double min_x = 0;
//    double max_x = 2 * M_PI;
//
//    std::vector<Matrix> X;
//    std::vector<Matrix> y;
//
//    for (int i = 0; i < N; i++) {
//        Matrix input(1, 1);
//        double x_norm = (x_vals[i] - min_x) / (max_x - min_x);
//        input.set_value(0, 0, x_norm);
//
//        Matrix target(1, 1);
//        target.set_value(0, 0, y_vals[i]);
//
//        X.push_back(input);
//        y.push_back(target);
//    }
//
//    Network net = Network();
//    net.add_layer(std::make_unique<Dense>(1, 16, WeightInit::Glorot));
//    net.add_layer(std::make_unique<Tanh>());
//    net.add_layer(std::make_unique<Dense>(16, 16, WeightInit::Glorot));
//    net.add_layer(std::make_unique<Tanh>());
//    net.add_layer(std::make_unique<Dense>(16, 1, WeightInit::Glorot));
//
//    net.set_loss(std::make_unique<MSE>());
//
//    Adam optimizer(0.001);
//
//    net.set_training(true);
//    net.train(X, y, 10000, optimizer, 16);
//
//    net.set_training(false);
//
//    std::cout << "\n===== SIN(x) APPROXIMATION =====\n";
//
//    for (int i = 0; i < 10; i++) {
//        double x = (2 * M_PI) * i / 10;
//        double x_norm = (x - min_x) / (max_x - min_x);
//
//        Matrix input(1, 1);
//        input.set_value(0, 0, x_norm);
//
//        double pred = net.forward(input).get_value(0, 0);
//        double real = sin(x);
//
//        std::cout << "x=" << x
//            << " sin(x)=" << real
//            << " pred=" << pred
//            << " error=" << fabs(pred - real)
//            << "\n";
//    }
//
//    std::cout << "\n===== TEST CASES =====\n";
//
//    auto test = [&](double x) {
//        double x_norm = (x - min_x) / (max_x - min_x);
//        Matrix t(1, 1);
//        t.set_value(0, 0, x_norm);
//        double pred = net.forward(t).get_value(0, 0);
//        std::cout << "x=" << x
//            << " sin(x)=" << sin(x)
//            << " pred=" << pred << "\n";
//        };
//
//    test(0);
//    test(M_PI / 2);
//    test(M_PI);
//    test(3 * M_PI / 2);
//    test(2 * M_PI);
//
//    return 0;
//}