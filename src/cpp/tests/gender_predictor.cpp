#include<iostream>
#include<vector>
#include "Matrix.hpp"
#include "Layer.hpp"
#include "Dense.hpp"
#include "Network.hpp"
#include "MSE.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "BCE.hpp"
#include "Optimizer.hpp"
#include "SGD.hpp"
#include "Adam.hpp"
#include "Dropout.hpp"
#include "BatchNorm.hpp"

int main() {
    const int N = 30;
    int height[N] = {
        181, 176, 177, 188, 195, 183, 179, 192, 186, 174,
        180, 187, 191, 178, 185,
        149, 160, 154, 171, 176, 158, 163, 155, 168, 172,
        157, 161, 165, 153, 169
    };
    int weight[N] = {
        77, 90, 75, 90, 90, 82, 78, 95, 88, 71,
        80, 85, 92, 76, 84,
        44, 45, 54, 54, 52, 51, 57, 48, 61, 63,
        50, 55, 58, 47, 60
    };
    int gender[N] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int min_h = height[0], max_h = height[0];
    int min_w = weight[0], max_w = weight[0];
    for (int i = 1; i < N; i++) {
        if (height[i] < min_h) min_h = height[i];
        if (height[i] > max_h) max_h = height[i];
        if (weight[i] < min_w) min_w = weight[i];
        if (weight[i] > max_w) max_w = weight[i];
    }

    double height_norm[N];
    double weight_norm[N];
    for (int i = 0; i < N; i++) {
        height_norm[i] = (height[i] - min_h) / double(max_h - min_h);
        weight_norm[i] = (weight[i] - min_w) / double(max_w - min_w);
    }

    std::vector<Matrix> X;
    std::vector<Matrix> y;
    for (int i = 0; i < N; i++) {
        Matrix input(1, 2);
        input.set_value(0, 0, height_norm[i]);
        input.set_value(0, 1, weight_norm[i]);
        Matrix target(1, 1);
        target.set_value(0, 0, gender[i]);
        X.push_back(input);
        y.push_back(target);
    }

    Network test = Network();
    test.add_layer(std::make_unique<Dense>(2, 16, WeightInit::Glorot));
    test.add_layer(std::make_unique<ReLU>());
    test.add_layer(std::make_unique<Dropout>(0.2));
    test.add_layer(std::make_unique<Dense>(16, 1, WeightInit::Glorot));
    test.add_layer(std::make_unique<Sigmoid>());
    test.set_loss(std::make_unique<BCE>());

    Adam optimizer(0.001);
    test.set_training(true);
    test.train(X, y, 2000, optimizer, 15);

    test.set_training(false);

    std::cout << "\n===== TRAINING SET =====\n";
    int correct = 0;
    for (int i = 0; i < N; i++) {
        double pred = test.forward(X[i]).get_value(0, 0);
        int predicted = pred >= 0.5 ? 1 : 0;
        std::cout << "h=" << height[i] << " w=" << weight[i]
            << " | label=" << gender[i]
            << " pred=" << pred
            << (predicted == gender[i] ? " OK" : " WRONG")
            << "\n";
        if (predicted == gender[i]) correct++;
    }
    std::cout << "\nTraining accuracy: " << correct << "/" << N
        << " (" << (100.0 * correct / N) << "%)\n";

    std::cout << "\n===== TEST CASES =====\n";
    auto norm = [&](double v, double minv, double maxv) {
        return (v - minv) / double(maxv - minv);
        };

    Matrix t1(1, 2);
    t1.set_value(0, 0, norm(185, min_h, max_h));
    t1.set_value(0, 1, norm(88, min_w, max_w));
    std::cout << "185cm 88kg  (male)   -> " << test.forward(t1).get_value(0, 0) << "\n";

    Matrix t2(1, 2);
    t2.set_value(0, 0, norm(150, min_h, max_h));
    t2.set_value(0, 1, norm(48, min_w, max_w));
    std::cout << "150cm 48kg  (female) -> " << test.forward(t2).get_value(0, 0) << "\n";

    Matrix t3(1, 2);
    t3.set_value(0, 0, norm(170, min_h, max_h));
    t3.set_value(0, 1, norm(65, min_w, max_w));
    std::cout << "170cm 65kg  (unsure) -> " << test.forward(t3).get_value(0, 0) << "\n";

    Matrix t4(1, 2);
    t4.set_value(0, 0, norm(198, min_h, max_h));
    t4.set_value(0, 1, norm(82, min_w, max_w));
    std::cout << "198cm 82kg  (male)   -> " << test.forward(t4).get_value(0, 0) << "\n";

    Matrix t5(1, 2);
    t5.set_value(0, 0, norm(162, min_h, max_h));
    t5.set_value(0, 1, norm(57, min_w, max_w));
    std::cout << "162cm 57kg  (female) -> " << test.forward(t5).get_value(0, 0) << "\n";

    Matrix t6(1, 2);
    t6.set_value(0, 0, norm(175, min_h, max_h));
    t6.set_value(0, 1, norm(70, min_w, max_w));
    std::cout << "175cm 70kg  (unsure) -> " << test.forward(t6).get_value(0, 0) << "\n";

    return 0;
}