//#include<iostream>
//#include<vector>
//#include "Matrix.hpp"
//#include "Layer.hpp"
//#include "Dense.hpp"
//#include "Network.hpp"
//#include "MSE.hpp"
//#include "ReLU.hpp"
//#include "Sigmoid.hpp"
//#include "BCE.hpp"
//#include "Optimizer.hpp"
//#include "SGD.hpp"
//#include "Adam.hpp"
//
//int main() {
//    const int N = 4;
//
//    double x1[N] = { 0, 0, 1, 1 };
//    double x2[N] = { 0, 1, 0, 1 };
//    double labels[N] = { 0, 1, 1, 0 };
//
//    std::vector<Matrix> X;
//    std::vector<Matrix> y;
//
//    for (int i = 0; i < N; i++) {
//        Matrix input(1, 2);
//        input.set_value(0, 0, x1[i]);
//        input.set_value(0, 1, x2[i]);
//
//        Matrix target(1, 1);
//        target.set_value(0, 0, labels[i]);
//
//        X.push_back(input);
//        y.push_back(target);
//    }
//
//    Network net = Network();
//    net.add_layer(std::make_unique<Dense>(2, 4, WeightInit::Glorot));
//    net.add_layer(std::make_unique<Sigmoid>());
//    net.add_layer(std::make_unique<Dense>(4, 1, WeightInit::Glorot));
//    net.add_layer(std::make_unique<Sigmoid>());
//
//    net.set_loss(std::make_unique<BCE>());
//
//    Adam optimizer(0.01);
//
//    net.set_training(true);
//    net.train(X, y, 5000, optimizer, 4);
//
//    net.set_training(false);
//
//    std::cout << "\n===== XOR RESULTS =====\n";
//    int correct = 0;
//
//    for (int i = 0; i < N; i++) {
//        double pred = net.forward(X[i]).get_value(0, 0);
//        int predicted = pred >= 0.5 ? 1 : 0;
//
//        std::cout << x1[i] << " XOR " << x2[i]
//            << " = " << labels[i]
//            << " | pred=" << pred
//            << (predicted == labels[i] ? " OK" : " WRONG")
//            << "\n";
//
//        if (predicted == labels[i]) correct++;
//    }
//
//    std::cout << "\nAccuracy: " << correct << "/" << N
//        << " (" << (100.0 * correct / N) << "%)\n";
//
//    std::cout << "\n===== TEST CASES =====\n";
//
//    Matrix t1(1, 2);
//    t1.set_value(0, 0, 0);
//    t1.set_value(0, 1, 0);
//    std::cout << "0 XOR 0 -> " << net.forward(t1).get_value(0, 0) << "\n";
//
//    Matrix t2(1, 2);
//    t2.set_value(0, 0, 0);
//    t2.set_value(0, 1, 1);
//    std::cout << "0 XOR 1 -> " << net.forward(t2).get_value(0, 0) << "\n";
//
//    Matrix t3(1, 2);
//    t3.set_value(0, 0, 1);
//    t3.set_value(0, 1, 0);
//    std::cout << "1 XOR 0 -> " << net.forward(t3).get_value(0, 0) << "\n";
//
//    Matrix t4(1, 2);
//    t4.set_value(0, 0, 1);
//    t4.set_value(0, 1, 1);
//    std::cout << "1 XOR 1 -> " << net.forward(t4).get_value(0, 0) << "\n";
//
//    return 0;
//}