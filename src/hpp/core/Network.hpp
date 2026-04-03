	#pragma once

	#include <vector>
	#include <iostream>
	#include "Layer.hpp"
	#include "Loss.hpp"
	#include <memory>

	class Network {
	public:
		std::vector<std::unique_ptr<Layer>> layers;
		std::unique_ptr<Loss> loss_function;

		Network();
		//~Network();

		void add_layer(std::unique_ptr<Layer> layer);
		Matrix forward(const Matrix& input);
		Matrix backward(const Matrix& grad_output);
		void set_loss(std::unique_ptr<Loss> loss);
		void update_weights(Optimizer& optimizer);
		void set_training(bool training);
		void zero_gradients();
		void train(const std::vector<Matrix>& X, const std::vector<Matrix>& y, int epochs, Optimizer& optimizer, int batch_size = 8);
		//Matrix predict(const Matrix& input);
	};