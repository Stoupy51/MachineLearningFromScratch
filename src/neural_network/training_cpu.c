
#include "training_cpu.h"
#include "../universal_utils.h"
#include "../universal_pthread.h"
#include "../st_benchmark.h"

///// Private functions /////

/**
 * @brief Copy a neural network only by duplicating memory for activations values
 * Used for multi-threaded feed forward
 * 
 * @param network			Neural network to copy
 * 
 * @return NeuralNetwork	Copy of the neural network
 */
NeuralNetwork getSimpleCopyForMultiThreadedFeedForward(NeuralNetwork network) {

	// Copy the neural network
	NeuralNetwork copy = network;

	// Allocate memory for the activations values of each layer
	copy.layers = mallocBlocking(network.nb_layers * sizeof(NeuronLayer), "getSimpleCopyForMultiThreadedFeedForward()");
	for (int i = 0; i < network.nb_layers; i++) {
		copy.layers[i] = network.layers[i];
		copy.layers[i].activations_values = mallocBlocking(network.layers[i].nb_neurons * sizeof(nn_type), "getSimpleCopyForMultiThreadedFeedForward()");
	}

	// Correct the pointers of the input layer and the output layer
	copy.input_layer = &copy.layers[0];
	copy.output_layer = &copy.layers[network.nb_layers - 1];

	// Return the copy
	return copy;
}

/**
 * @brief Free a neural network only by freeing memory for activations values
 * Used for multi-threaded feed forward
 * 
 * @param network			Neural network to free
 */
void freeSimpleCopyForMultiThreadedFeedForward(NeuralNetwork network) {
	
	// Free the activations values of each layer
	for (int i = 0; i < network.nb_layers; i++)
		free(network.layers[i].activations_values);

	// Free the neural network
	free(network.layers);
}

/**
 * @brief Mini-batch Gradient Descent algorithm of the neural network
 * using a batch of inputs and a batch of target outputs
 * 
 * @param network				Pointer to the neural network
 * @param inputs				Pointer to the inputs array
 * @param predictions		Pointer to the predicted outputs array
 * @param targets		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 */
void StochasticGradientDescentCPU(NeuralNetwork *network, nn_type **predictions, nn_type **targets, int batch_size) {

	// Initialize the gradients of the weights and the biases to 0
	initGradientsNeuralNetwork(network);
	for (int i = 1; i < network->nb_layers; i++) {
		memset(network->layers[i].biases_gradients, 0, network->layers[i].nb_neurons * sizeof(nn_type));
		memset(network->layers[i].weights_gradients_flat, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(nn_type));
	}

	// Calculate the derivatives of activation functions for output layer
	network->output_layer->activation_function_derivative(network->output_layer->activations_values, network->output_layer->nb_neurons);

	// For each sample of the batch,
	for (int sample = 0; sample < batch_size; sample++) {

		// For each neuron of the output layer,
		for (int j = 0; j < network->output_layer->nb_neurons; j++) {

			// Calculate the gradient of the cost function with respect to the activation value of the neuron
			nn_type gradient = network->loss_function_derivative(predictions[sample][j], targets[sample][j])
				* network->output_layer->activations_values[j];

			// For each input of the neuron, calculate the gradient of the cost function with respect to the weight of the input
			for (int k = 0; k < network->output_layer->nb_inputs_per_neuron; k++)
				network->output_layer->weights_gradients[j][k] += gradient * network->layers[network->nb_layers - 2].activations_values[k];

			// Calculate the gradient of the cost function with respect to the bias of the neuron
			network->output_layer->biases_gradients[j] += gradient;
		}
	}

	// For each layer of the neural network (except the input layer) (in reverse order),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// Calculate the derivatives of activation functions
		network->layers[i].activation_function_derivative(network->layers[i].activations_values, network->layers[i].nb_neurons);

		// For each neuron of the layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the gradient of the cost function with respect to the activation value of the neuron
			nn_type gradient = 0.0;
			for (int k = 0; k < network->layers[i + 1].nb_neurons; k++)
				gradient += network->layers[i + 1].weights[k][j]
					* network->layers[i + 1].biases_gradients[k];

			// Multiply the gradient by the derivative of the activation function of the neuron
			gradient *= network->layers[i].activations_values[j];

			// For each input of the neuron, calculate the gradient of the cost function with respect to the weight of the input
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
				network->layers[i].weights_gradients[j][k] += gradient * network->layers[i - 1].activations_values[k];

			// Calculate the gradient of the cost function with respect to the bias of the neuron
			network->layers[i].biases_gradients[j] += gradient;
		}
	}

	// Update the weights and the biases
	for (int i = 1; i < network->nb_layers; i++) {
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
				network->layers[i].weights[j][k] -= (network->learning_rate * network->layers[i].weights_gradients[j][k]);
			network->layers[i].biases[j] -= (network->learning_rate * network->layers[i].biases_gradients[j]);
		}
	}
}


///// Public functions /////



/**
 * @brief Feed forward algorithm of the neural network
 * 
 * @param network		Pointer to the neural network
 * @param input			Pointer to the input array
 */
void FeedForwardCPU(NeuralNetwork *network, nn_type *input) {

	// Copy the inputs to the input layer of the neural network
	size_t input_layer_size = network->input_layer->nb_neurons * sizeof(nn_type);
	memcpy(network->input_layer->activations_values, input, input_layer_size);

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the sum of the inputs multiplied by the weights
			nn_type input_sum = network->layers[i].biases[j];	// Add the bias to the sum
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
				input_sum += network->layers[i - 1].activations_values[k] * network->layers[i].weights[j][k];
			
			// Add the bias neuron if there is one
			if (network->layers[i].has_bias_neuron)
				input_sum += network->layers[i].weights[j][network->layers[i].nb_inputs_per_neuron];

			// Save the sum of the inputs multiplied by the weights
			network->layers[i].activations_values[j] = input_sum;
		}

		// Activate the layer with the activation function
		network->layers[i].activation_function(network->layers[i].activations_values, network->layers[i].nb_neurons);
	}
}

/**
 * @brief Feed forward algorithm of the neural network using a batch of inputs
 * 
 * @param network		Pointer to the neural network
 * @param inputs		Pointer to the inputs array
 * @param outputs		Pointer to the outputs array
 * @param batch_size	Number of samples in the batch
 */
void FeedForwardBatchCPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size) {

	// Local variables
	size_t output_layer_size = network->output_layer->nb_neurons * sizeof(nn_type);

	// For each sample of the batch,
	for (int i = 0; i < batch_size; i++) {

		// Feed forward the inputs
		FeedForwardCPU(network, inputs[i]);

		// Copy the outputs of the neural network to the outputs array
		memcpy(outputs[i], network->output_layer->activations_values, output_layer_size);
	}
}



/**
 * @brief Utility function to shuffle the training data
 * 
 * @param inputs			Pointer to the inputs array
 * @param targets	Pointer to the target outputs array
 * @param batch_size		Number of samples in the batch
 */
void shuffleTrainingData(nn_type **inputs, nn_type **targets, int batch_size) {

	// Prepare a new array of pointers to the inputs and the target outputs
	nn_type **new_inputs = mallocBlocking(batch_size * sizeof(nn_type *), "shuffleTrainingData()");
	nn_type **new_targets = mallocBlocking(batch_size * sizeof(nn_type *), "shuffleTrainingData()");
	int new_size = 0;

	// While there are samples in the batch,
	int nb_samples = batch_size;
	while (nb_samples > 0) {

		// Select a random sample
		int random_index = rand() % nb_samples;

		// Add the random sample to the new array
		new_inputs[new_size] = inputs[random_index];
		new_targets[new_size] = targets[random_index];
		new_size++;

		// Remove the random sample from the old array by replacing it with the last sample
		inputs[random_index] = inputs[nb_samples - 1];
		targets[random_index] = targets[nb_samples - 1];
		nb_samples--;
	}

	// Copy the new array to the old array
	memcpy(inputs, new_inputs, batch_size * sizeof(nn_type *));
	memcpy(targets, new_targets, batch_size * sizeof(nn_type *));

	// Free the new array
	free(new_inputs);
	free(new_targets);
}

/**
 * @brief Compute the cost of the neural network using a batch of inputs and a batch of target outputs
 * This is used to evaluate the performance of the neural network during the training phase
 * 
 * @param network				Pointer to the neural network
 * @param predictions		Pointer to the predicted outputs array
 * @param targets		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 * 
 * @return nn_type				Cost of the neural network
 */
nn_type ComputeCost(NeuralNetwork *network, nn_type **predictions, nn_type **targets, int batch_size) {
	
	// Add the cost of each output neuron of each sample of the batch
	nn_type cost = 0.0;
	for (int i = 0; i < batch_size; i++)
		for (int j = 0; j < network->output_layer->nb_neurons; j++)
			cost += network->loss_function(predictions[i][j], targets[i][j]) / network->output_layer->nb_neurons;

	// Return the cost of the neural network
	return cost / batch_size;
}

/**
 * @brief One epoch of the training of the neural network with the CPU (Single core)
 * 
 * @param network				Pointer to the neural network
 * @param inputs				Pointer to the inputs array
 * @param target				Pointer to the target outputs array
 * @param nb_inputs				Number of samples in the inputs array and in the target outputs array
 * @param nb_batches			Number of batches
 * @param batch_size			Number of samples in the batch
 * @param current_epoch			Current epoch
 * @param nb_epochs				Number of epochs to train the neural network
 * @param current_error			Pointer to the current error value
 * @param test_inputs			Pointer to the test inputs array
 * @param target_tests			Pointer to the target outputs array for the test inputs
 * @param nb_test_inputs		Number of samples in the test inputs array and in the target outputs array for the test inputs
 * @param verbose				Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: all)
 * @param optimizer				Pointer to the optimizer function (StochasticGradientDescentCPU(), AdamCPUSingleThread(), RMSPropCPUSingleThread(), ...)
 * 
 * @return void
 */
void epochCPU(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int nb_batches, int batch_size, int current_epoch, int nb_epochs, nn_type *current_error, nn_type **test_inputs, nn_type **target_tests, int nb_test_inputs, int verbose, optimizer_t optimizer) {

	// Shuffle the training data
	shuffleTrainingData(inputs, target, nb_inputs);

	// For each batch of the inputs,
	for (int current_batch = 0; current_batch < nb_batches; current_batch++) {

		// Calculate the index of the first and the last sample of the batch
		int first_sample = current_batch * batch_size;
		int last_sample = first_sample + batch_size - 1;
		if (last_sample >= nb_inputs)
			last_sample = nb_inputs - 1;
		
		// Calculate the number of samples in the batch
		int nb_samples = last_sample - first_sample + 1;

		// Verbose
		if (verbose > 1)
			DEBUG_PRINT("TrainCPU(1 thread): Epoch %d/%d,\tBatch %d/%d,\tSamples %d-%d/%d\n", current_epoch, nb_epochs, current_batch + 1, nb_batches, first_sample + 1, last_sample + 1, nb_inputs);
		
		// Prepare the predicted outputs array for the batch
		nn_type **predictions;
		nn_type *predicted_flat_outputs = try2DFlatMatrixAllocation((void***)&predictions, nb_samples, network->output_layer->nb_neurons, sizeof(nn_type), "epochCPU(predictions)");

		// Compute the forward pass to get predictions for the mini-batch
		FeedForwardBatchCPU(network, inputs + first_sample, predictions, nb_samples);

		// Optimize the neural network with the batch
		optimizer(network, predictions, target + first_sample, nb_samples);

		// Free the predicted outputs array for the batch
		free2DFlatMatrix((void**)predictions, predicted_flat_outputs, nb_samples);
	}

	///// Test the neural network to see the accuracy
	// Use the test inputs to calculate the current error
	if (nb_test_inputs > 0) {

		// Prepare predicted outputs array for the test inputs
		nn_type **predicted;
		nn_type *flat_predicted = try2DFlatMatrixAllocation((void***)&predicted, nb_test_inputs, network->output_layer->nb_neurons, sizeof(nn_type), "TrainCPU(1 thread)");

		// Feed forward the test inputs
		FeedForwardBatchCPU(network, test_inputs, predicted, nb_test_inputs);
		
		// Calculate the error of the test inputs using the loss function
		*current_error = ComputeCost(network, predicted, target_tests, nb_test_inputs);
		
		// Free the predicted outputs array for the test inputs
		free2DFlatMatrix((void**)predicted, flat_predicted, nb_test_inputs);
	}
}

/**
 * @brief Get the optimizer function from its name
 * 
 * @param optimizer		Optimizer function name
 * 
 * @return optimizer_t	Optimizer function
 */
optimizer_t get_optimizer(const char *optimizer) {
	if (strcmp(optimizer, "SGD") == 0 || strcmp(optimizer, "StochasticGradientDescent") == 0)
		return StochasticGradientDescentCPU;
	// else if (strcmp(optimizer, "Adam") == 0)
	// 	return AdamCPU;
	// else if (strcmp(optimizer, "RMSProp") == 0)
	// 	return RMSPropCPU;
	else {
		ERROR_PRINT("Unknown optimizer function: '%s'\n", optimizer);
		exit(EXIT_FAILURE);
	}
}

/**
 * @brief Train the neural network with the CPU (Single core)
 * by using a batch of inputs and a batch of target outputs,
 * a number of epochs and a target error value
 * 
 * @param network					Pointer to the neural network
 * 
 * @param inputs					Pointer to the inputs array
 * @param target					Pointer to the target outputs array
 * @param nb_inputs					Number of samples in the inputs array and in the target outputs array
 * @param test_inputs_percentage	Percentage of the inputs array to use as test inputs (from the end) (usually 10: 10%)
 * @param batch_size				Number of samples in the batch
 * 
 * @param nb_epochs					Number of epochs to train the neural network (optional, -1 to disable)
 * @param error_target				Target error value to stop the training (optional, 0 to disable)
 * At least one of the two must be specified. If both are specified, the training will stop when one of the two conditions is met.
 * 
 * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: all)
 * @param optimizer					Optimizer function to use ("SGD", "Adam", "RMSProp", ...)
 * 
 * @return int						Number of epochs done, -1 if there is an error
 */
int TrainCPU(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int test_inputs_percentage, int batch_size, int nb_epochs, nn_type error_target, int verbose, const char *optimizer) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = nb_epochs != -1 || error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "TrainCPU(1 thread): At least the number of epochs or the error target must be specified!\n");

	// Get the optimizer function
	optimizer_t optimizer_function = get_optimizer(optimizer);

	// Prepare the test inputs
	int nb_test_inputs = nb_inputs * test_inputs_percentage / 100;
	nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &inputs[nb_inputs];
	nn_type **target_tests = &target[nb_inputs];
	if (verbose > 0)
		INFO_PRINT("TrainCPU(1 thread): %d inputs, %d test inputs\n", nb_inputs, nb_test_inputs);

	// Local variables
	int current_epoch = 0;
	nn_type current_error = 100.0;
	int nb_batches = nb_inputs / batch_size + (nb_inputs % batch_size != 0);

	// Training
	while (current_epoch < nb_epochs && current_error > error_target) {

		// Reset the current error and increment the current epoch
		current_error = 0;
		current_epoch++;

		// Verbose, benchmark the training or do one epoch of the training
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == nb_epochs || current_epoch % 10 == 0)) || verbose > 1) {
			
			// Get current microseconds
			struct timeval start_time;
			st_gettimeofday(start_time, NULL);

			char benchmark_buffer[256];
			ST_BENCHMARK_SOLO_COUNT(benchmark_buffer,
				epochCPU(network, inputs, target, nb_inputs, nb_batches, batch_size, current_epoch, nb_epochs, &current_error, test_inputs, target_tests, nb_test_inputs, verbose, optimizer_function),
				"TrainCPU(1 thread): Epoch %d/%d, %s: " ST_COLOR_YELLOW "%.8"NN_FORMAT ST_COLOR_RED " ("ST_COLOR_YELLOW "%.2f" ST_COLOR_RED "us/input)", 1, 0
			);

			// Get current microseconds
			struct timeval end_time;
			st_gettimeofday(end_time, NULL);

			// Calculate the time spent per step (time of the epoch divided by the number of inputs)
			double time_per_step = (double)(end_time.tv_sec - start_time.tv_sec) * (1000000.0 / nb_inputs) + (double)(end_time.tv_usec - start_time.tv_usec) / nb_inputs;

			// Print the benchmark
			PRINTER(benchmark_buffer, current_epoch, nb_epochs, network->loss_function_name, current_error, time_per_step);
		}

		else
			// Do one epoch of the training
			epochCPU(network, inputs, target, nb_inputs, nb_batches, batch_size, current_epoch, nb_epochs, &current_error, test_inputs, target_tests, nb_test_inputs, verbose, optimizer_function);
	}

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("TrainCPU(1 thread): Training done!\n");
	return current_epoch;
}


