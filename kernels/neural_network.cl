

/**
 * @brief Function to calculate the output delta for the output layer
 * in the backpropagation algorithm.
 * 
 * @param excepted_output		The excepted output array of the network
 * @param activation_values		The activation values array of the output layer
 * @param output_deltas			The output deltas array of the output layer
 * @param output_layer_size		The size of the output layer
 * 
 * @return void
 */
kernel void backpropagationOutputLayerDeltas(global double* excepted_output, global double* activation_values, global double* output_deltas, int output_layer_size) {

	// Get the index of the current thread
	int index = get_global_id(0);

	// If the index is smaller than the output layer size
	if (index < output_layer_size) {
		output_deltas[index] = (excepted_output[index] - activation_values[index]) * activation_values[index] * (1 - activation_values[index]);
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
 * 
 * @return void
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
	}
}

