
#include "training_cpu.h"

/**
 * @brief Feed forward the neural network with the input array
 * For a neural network using double as type
 * 
 * Detailed description of the feed forward algorithm:
 * - Copy the input array into the input layer of the neural network
 * - For each layer of the neural network (except the input layer):
 * 		- For each neuron of the current layer:
 * 			- Calculate the sum of all the weights of the previous layer linked to the current neuron
 * 			- Add the bias of the current neuron
 * 			- Apply the activation function to the weighted sum (often sigmoid)
 * 			- Save the result in the activations_values array of the current layer
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - activation_value = activation_function( sum( input * weight  ) + bias )
 * - where input is the activation_value of the previous layer
 * 
 * @details i = Index of the selected layer
 * @details j = Index of the selected neuron
 * @details k = Index of the previous layer selected neuron
 * 
 * @param network	Pointer to the neural network
 * @param input		Pointer to the input array (double), must be the same size as the input layer
 * 
 * @return void
 */
void NeuralNetworkDfeedForwardCPU(NeuralNetworkD *network, double *input) {

	// Set the input layer (copy the input array into the input layer of the neural network)
	memcpy(network->input_layer->activations_values, input, network->input_layer->nb_neurons * sizeof(double));

	// Feed forward: for each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the sum of all the weights of the previous layer linked to the current neuron
			double weighted_sum = 0;
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {
				double input_value = network->layers[i - 1].activations_values[k];
				double weight = network->layers[i].weights[j][k];	// weights[j][k] where j = current neuron, k = previous neuron
				weighted_sum += input_value * weight;
			}

			// Add the bias of the current neuron
			weighted_sum += network->layers[i].biases[j];

			// Apply the activation function to the weighted sum (often sigmoid)
			network->layers[i].activations_values[j] = network->activation_function(weighted_sum);
		}
	}
}

/**
 * @brief Backpropagate the neural network with the excepted output array
 * For a neural network using double as type
 * 
 * Detailed description of the backpropagation algorithm:
 * - For each neuron of the output layer:
 * 		- Calculate the error of the neuron (excepted_output - activation_value)
 * 		- Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
 * 		- Calculate the delta of the neuron (error * derivative)
 * 		- Save the delta of the neuron in the deltas array of the output layer
 * - For each layer of the neural network (except the output layer and the input layer) (order last to first):
 * 		- For each neuron of the current layer:
 * 			- Calculate the error of the neuron (sum( weight * delta of the next layer ))
 * 			- Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
 * 			- Calculate the delta of the neuron (error * derivative)
 * 			- Save the delta of the neuron in the deltas array of the current layer
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - delta = error * derivative
 * - where error = sum( weight * delta of the next layer )
 * - where derivative = activation_value * (1 - activation_value)
 * 
 * @details i = Index of the selected layer (order last to first)
 * @details j = Index of the selected neuron
 * @details k = Index of the next layer selected neuron
 * 
 * @param network			Pointer to the neural network
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * 
 * @return void
 */
void NeuralNetworkDbackpropagationCPU(NeuralNetworkD *network, double *excepted_output) {

	// For each neuron of the output layer,
	for (int neuron = 0; neuron < network->output_layer->nb_neurons; neuron++) {

		// Calculate the error of the neuron (excepted_output - activation_value)
		double error = excepted_output[neuron] - network->output_layer->activations_values[neuron];

		// Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
		double activation_value = network->output_layer->activations_values[neuron];
		double derivative = activation_value * (1 - activation_value);

		// Calculate the delta of the neuron (error * derivative)
		network->output_layer->deltas[neuron] = error * derivative;
	}

	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the error of the neuron (sum( weight * delta of the next layer ))
			double error = 0;
			for (int k = 0; k < network->layers[i + 1].nb_neurons; k++) {

				// Get the weight of the next layer neuron linked to the current neuron
				// (Not [j][k] because it's reversed compared to the feed forward algorithm (checking next layer instead of previous layer))
				double weight = network->layers[i + 1].weights[k][j];

				// Get the delta of the next layer neuron
				double delta = network->layers[i + 1].deltas[k];

				// Add the weight * delta to the error
				error += weight * delta;
			}

			// Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
			double activation_value = network->layers[i].activations_values[j];
			double derivative = activation_value * (1 - activation_value);

			// Calculate the delta of the neuron (error * derivative)
			network->layers[i].deltas[j] = error * derivative;
		}
	}
}

/**
 * @brief Update/Nudge the weights and biases of the neural network
 * For a neural network using double as type
 * 
 * Detailed description of the update weights and biases algorithm:
 * - For each layer of the neural network (except the input layer):
 * 		- For each neuron of the current layer:
 * 			- For each weight of the current neuron:
 * 				- Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
 * 			- Update the bias (bias + (learning_rate * delta of the current neuron))
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - bias = bias + (learning_rate * delta of the current neuron)
 * - for each weight of the current neuron:
 *   - weight = weight + (learning_rate * delta of the current neuron * activation_value of the previous layer)
 * 
 * @details i = Index of the selected layer (except the input layer)
 * @details j = Index of the selected neuron
 * @details k = Index of the previous layer selected neuron
 * 
 * @param network	Pointer to the neural network
 * 
 * @return void
 */
void NeuralNetworkDupdateWeightsAndBiasesCPU(NeuralNetworkD *network) {

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Variables for easier reading
			double learning_rate = network->learning_rate;
			double delta = network->layers[i].deltas[j];

			// For each weight of the current neuron,
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {

				// Variable for easier reading
				double activation_value = network->layers[i - 1].activations_values[k];

				// Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
				network->layers[i].weights[j][k] += learning_rate * delta * activation_value;
			}

			// Update the bias (bias + (learning_rate * delta of the current neuron))
			network->layers[i].biases[j] += learning_rate * delta;
		}
	}
}

/**
 * @brief Train the neural network using the backpropagation algorithm
 * For a neural network using double as type
 * 
 * @param network			Pointer to the neural network
 * @param input				Pointer to the input array (double), must be the same size as the input layer
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * 
 * @return void
 */
void NeuralNetworkDtrainCPU(NeuralNetworkD *network, double *input, double *excepted_output) {

	// Feed forward
	NeuralNetworkDfeedForwardCPU(network, input);

	// Backpropagation
	NeuralNetworkDbackpropagationCPU(network, excepted_output);

	// Update weights and biases
	NeuralNetworkDupdateWeightsAndBiasesCPU(network);
}
