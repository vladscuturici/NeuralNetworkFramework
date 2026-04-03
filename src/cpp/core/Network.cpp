	#include "Network.hpp"

	#include<iostream>
	#include <vector>
	#include <algorithm>
	#include <numeric> 

	Network::Network() : loss_function(nullptr) {
		// Layers vector is automatically initialized empty
	}

	void Network::add_layer(std::unique_ptr<Layer> layer) {
		this->layers.push_back(std::move(layer)); 
	}

	Matrix Network::forward(const Matrix& input) {

		Matrix forward_pass_temp = input;

		for (const auto& layer : this->layers) {
			forward_pass_temp = layer->forward(forward_pass_temp);
		}

		return forward_pass_temp;
	}

	Matrix Network::backward(const Matrix& grad_output) {

		Matrix backward_pass_temp = grad_output;

		for (int i = (int) layers.size() - 1; i >= 0; --i) {
			backward_pass_temp = layers[i]->backward(backward_pass_temp);
		}

		return backward_pass_temp;
	}

	void Network::set_loss(std::unique_ptr<Loss> loss) {
		this->loss_function = std::move(loss);
	}

	void Network::update_weights(Optimizer& optimizer) {
    for (const auto& layer : layers)
        layer->update_weights(optimizer);
	}

	void Network::set_training(bool training) {
		for (const auto& layer : layers)
			layer->set_training(training);
	}

	void Network::zero_gradients() {
		for (const auto& layer : layers)
			layer->zero_gradients();
	}

	void Network::train(const std::vector<Matrix>& X, const std::vector<Matrix>& y, int epochs, Optimizer& optimizer, int batch_size) {

		for (int epoch = 0; epoch < epochs; epoch++) {
			double epoch_loss = 0.0;

			std::vector<int> indices(X.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

			int num_batches = 0;

			for (int start = 0; start < (int)X.size(); start += batch_size) {
				int end = std::min((int)X.size(), start + batch_size);
				int current_batch_size = end - start;

				zero_gradients();

				double batch_loss = 0.0;
				for (int i = start; i < end; i++) {
					int idx = indices[i];
					Matrix y_pred = this->forward(X[idx]);
					batch_loss += this->loss_function->loss(y_pred, y[idx]);
					Matrix grad = this->loss_function->backward(y_pred, y[idx]);
					this->backward(grad);
				}

				epoch_loss += batch_loss;
				num_batches++;

				this->update_weights(optimizer);
			}

			std::cout << "[Epoch " << epoch + 1 << "/" << epochs << "] Avg Loss: " << epoch_loss / X.size() << "\n";
		}
	}

