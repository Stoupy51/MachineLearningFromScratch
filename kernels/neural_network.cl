
/**
 * @brief Function to calculate the activation values of all the neurons
 * of the network in the feed forward algorithm.
 *
 * @param previous_layer_activation_values	The activation values of the previous layer
 * @param weights							The weights of the current layer
 * @param biases							The biases of the current layer
 * @param activation_values					The activation values of the current layer
 * @param current_layer_size				The size of the current layer
 * @param previous_layer_size				The size of the previous layer
 *
 * @return void
 */
kernel void feedForwardActivationValuesSigmoid(global double* previous_layer_activation_values, global double* weights, global double* biases, global double* activation_values, int current_layer_size, int previous_layer_size) {

	// Get the index of the current thread
	int index = get_global_id(0);

	// If the index is smaller than the current layer size
	if (index < current_layer_size) {

		// Prepare the index of the weights array
		int weight_index = index * previous_layer_size;

		// Calculate the sum of all the weights of the previous layer linked to the current neuron
		double weighted_sum = 0;
		for (int k = 0; k < previous_layer_size; k++)
			weighted_sum += previous_layer_activation_values[k] * weights[weight_index + k]; // weights[index][k] where index = current neuron, k = previous neuron

		// Apply the activation function to the weighted sum (often sigmoid)
		activation_values[index] = 1 / (1 + exp(-(weighted_sum + biases[index])));
		//printf("- id: %d, activation_value: %f\n", index, activation_values[index]);
	}
}

/**
 * @brief Function to calculate the output delta for the output layer
 * in the backpropagation algorithm.
 * 
 * @param excepted_output		The excepted output array of the network
 * @param activation_values		The activation values array of the output layer
 * @param output_deltas			The output deltas array of the output layer
 * @param output_layer_size		The size of the output layer
 */
kernel void backpropagationOutputLayerDeltas(global double* excepted_output, global double* activation_values, global double* output_deltas, int output_layer_size) {

	// Get the index of the current thread
	int index = get_global_id(0);

	// If the index is smaller than the output layer size
	if (index < output_layer_size) {
		output_deltas[index] = (excepted_output[index] - activation_values[index]) * activation_values[index] * (1 - activation_values[index]);
		//printf("- id: %d, output_delta: %f\n", index, output_deltas[index]);
	}
}

/**
 * @brief Function to calculate the deltas for the hidden layers
 * in the backpropagation algorithm.
 * 
 * @param next_layer_weights	The weights of the next layer
 * @param next_layer_deltas		The deltas of the next layer
 * @param activation_values		The activation values of the current layer
 * @param deltas				The deltas of the current layer
 * @param current_layer_size	The size of the current layer
 * @param next_layer_size		The size of the next layer
 */
kernel void backpropagationHiddenLayersDeltas(global double* next_layer_weights, global double* next_layer_deltas, global double* activation_values, global double* deltas, int current_layer_size, int next_layer_size) {

	// Get the index of the current thread
	int index = get_global_id(0);

	// If the index is smaller than the current layer size
	if (index < current_layer_size) {

		// Calculate the error of the neuron (sum( weight * delta of the next layer ))
		double error = 0;
		for (int k = 0; k < next_layer_size; k++)
			error += next_layer_weights[k * current_layer_size + index] * next_layer_deltas[k];
		
		// Calculate the delta of the neuron (error * derivative)
		deltas[index] = error * (activation_values[index] * (1 - activation_values[index]));
		//printf("- id: %d, delta: %f\n", index, deltas[index]);
	}
}

/**
 * @brief Function to update the weights and biases of the network
 * in the backpropagation algorithm.
 * 
 * @param previous_layer_activation_values	The activation values of the previous layer
 * @param deltas							The deltas of the current layer
 * @param weights							The weights of the current layer
 * @param biases							The biases of the current layer
 * @param current_layer_size				The size of the current layer
 * @param previous_layer_size				The size of the previous layer
 * @param learning_rate						The learning rate of the network
 */
kernel void updateWeightsAndBiases(global double* previous_layer_activation_values, global double* deltas, global double* weights, global double* biases, int current_layer_size, int previous_layer_size, double learning_rate) {

	// Get the index of the current thread
	int index = get_global_id(0);

	// If the index is smaller than the current layer size
	if (index < current_layer_size) {

		// Prepare the index of the weights array
		int weight_index = index * previous_layer_size;

		// For each weight of the current neuron,
		for (int k = 0; k < previous_layer_size; k++) {

			// Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
			weights[weight_index + k] += learning_rate * deltas[index] * previous_layer_activation_values[k];
		}

		// Update the bias (bias + (learning_rate * delta of the current neuron))
		biases[index] += learning_rate * deltas[index];
		//printf("- id: %d, bias: %f\n", index, biases[index]);
	}
}

