
#include "training.h"

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
void NeuralNetworkDfeedForward(NeuralNetworkD *network, double *input) {

	// Set the input layer (copy the input array into the input layer of the neural network)
	memcpy(network->input_layer->activations_values, input, network->input_layer->nb_neurons * sizeof(double));

	// Feed forward: for each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the sum of all the weights of the previous layer linked to the current neuron
			double weighted_sum = 0;
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
				weighted_sum += network->layers[i - 1].activations_values[k] * network->layers[i].weights[j][k];	// weights[j][k] where j = current neuron, k = previous neuron

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
 * @param network			Pointer to the neural network
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * 
 * @return void
 */
void NeuralNetworkDbackpropagation(NeuralNetworkD *network, double *excepted_output) {
	
	// For each neuron of the output layer,
	for (int i = 0; i < network->output_layer->nb_neurons; i++) {

		// Calculate the error of the neuron (excepted_output - activation_value)
		double error = excepted_output[i] - network->output_layer->activations_values[i];

		// Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
		double derivative = network->output_layer->activations_values[i] * (1 - network->output_layer->activations_values[i]);

		// Calculate the delta of the neuron (error * derivative)
		network->output_layer->deltas[i] = error * derivative;
	}
	DEBUG_PRINT("NeuralNetworkDbackpropagation(): Output layer deltas calculated.\n");

	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {
		DEBUG_PRINT("NeuralNetworkDbackpropagation(): Layer %d deltas calculation in progress...\n", i);

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {
			DEBUG_PRINT("NeuralNetworkDbackpropagation(): Layer %d neuron %d delta calculation in progress...\n", i, j);

			// Calculate the error of the neuron (sum( weight * delta of the next layer ))
			double error = 0;
			for (int k = 0; k < network->layers[i + 1].nb_neurons; k++)
				error += network->layers[i].weights[j][k] * network->layers[i + 1].deltas[k];
			DEBUG_PRINT("NeuralNetworkDbackpropagation(): Layer %d neuron %d error calculated.\n", i, j);

			// Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
			double derivative = network->layers[i].activations_values[j] * (1 - network->layers[i].activations_values[j]);
			DEBUG_PRINT("NeuralNetworkDbackpropagation(): Layer %d neuron %d derivative calculated.\n", i, j);

			// Calculate the delta of the neuron (error * derivative)
			network->layers[i].deltas[j] = error * derivative;
			DEBUG_PRINT("NeuralNetworkDbackpropagation(): Layer %d neuron %d delta calculated.\n", i, j);
		}
		DEBUG_PRINT("NeuralNetworkDbackpropagation(): Layer %d deltas calculated.\n", i);
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
 * 				- Update the weight (weight + (learning_rate * delta of the next layer * activation_value of the previous layer))
 * 			- Update the bias (bias + (learning_rate * delta of the current neuron))
 * 
 * @param network	Pointer to the neural network
 * 
 * @return void
 */
void NeuralNetworkDupdateWeightsAndBiases(NeuralNetworkD *network) {

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// For each weight of the current neuron,
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)

				// Update the weight (weight + (learning_rate * delta of the next layer * activation_value of the previous layer))
				network->layers[i].weights[j][k] += network->learning_rate * network->layers[i].deltas[j] * network->layers[i - 1].activations_values[k];

			// Update the bias (bias + (learning_rate * delta of the current neuron))
			network->layers[i].biases[j] += network->learning_rate * network->layers[i].deltas[j];
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
void NeuralNetworkDtrain(NeuralNetworkD *network, double *input, double *excepted_output) {

	// Feed forward
	DEBUG_PRINT("NeuralNetworkDtrain(): Feed forward in progress...\n");
	NeuralNetworkDfeedForward(network, input);
	DEBUG_PRINT("NeuralNetworkDtrain(): Feed forward done!\n");

	// Backpropagation
	DEBUG_PRINT("NeuralNetworkDtrain(): Backpropagation in progress...\n");
	NeuralNetworkDbackpropagation(network, excepted_output);
	DEBUG_PRINT("NeuralNetworkDtrain(): Backpropagation done!\n");

	// Update weights and biases
	DEBUG_PRINT("NeuralNetworkDtrain(): Update weights and biases in progress...\n");
	NeuralNetworkDupdateWeightsAndBiases(network);
	DEBUG_PRINT("NeuralNetworkDtrain(): Update weights and biases done!\n");
}

