
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



///// Public functions /////



/**
 * @brief Feed forward algorithm of the neural network
 * 
 * @param network		Pointer to the neural network
 * @param input			Pointer to the input array
 */
void FeedForwardCPUSingleThread(NeuralNetwork *network, nn_type *input) {

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

			// Activate the neuron with the activation function
			network->layers[i].activations_values[j] = network->layers[i].activation_function(input_sum);
		}
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
void FeedForwardBatchCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size, int verbose) {

	// Local variables
	size_t output_layer_size = network->output_layer->nb_neurons * sizeof(nn_type);

	// For each sample of the batch,
	for (int i = 0; i < batch_size; i++) {

		// Feed forward the inputs
		if (verbose > 0)
			DEBUG_PRINT("FeedForwardBatchCPUSingleThread(): Sample %d/%d\n", i + 1, batch_size);
		FeedForwardCPUSingleThread(network, inputs[i]);
		if (verbose > 0)
			DEBUG_PRINT("FeedForwardBatchCPUSingleThread(): Sample %d/%d done\n", i + 1, batch_size);

		// Copy the outputs of the neural network to the outputs array
		memcpy(outputs[i], network->output_layer->activations_values, output_layer_size);
	}
}

/**
 * @brief Backpropagation algorithm of the neural network
 * and update the weights and the biases of the neural network
 * 
 * @param network				Pointer to the neural network
 * @param predicted_outputs		Pointer to the predicted outputs array
 * @param target_outputs		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 */
void BackpropagationCPUSingleThread(NeuralNetwork *network, nn_type **predicted_outputs, nn_type **target_outputs, int batch_size) {

	// Initialize the gradients of the weights and the biases to 0
	initGradientsNeuralNetwork(network);
	for (int i = 1; i < network->nb_layers; i++) {
		memset(network->layers[i].biases_gradients, 0, network->layers[i].nb_neurons * sizeof(nn_type));
		memset(network->layers[i].weights_gradients_flat, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(nn_type));
	}

	// For each sample of the batch,
	for (int sample = 0; sample < batch_size; sample++) {

		// For each neuron of the output layer,
		for (int j = 0; j < network->output_layer->nb_neurons; j++) {

			// Calculate the gradient of the cost function with respect to the activation value of the neuron
			nn_type gradient = network->loss_function_derivative(predicted_outputs[sample][j], target_outputs[sample][j])
				* network->output_layer->activation_function_derivative(network->output_layer->activations_values[j]);

			// For each input of the neuron, calculate the gradient of the cost function with respect to the weight of the input
			for (int k = 0; k < network->output_layer->nb_inputs_per_neuron; k++)
				network->output_layer->weights_gradients[j][k] += gradient * network->layers[network->nb_layers - 2].activations_values[k];

			// Calculate the gradient of the cost function with respect to the bias of the neuron
			network->output_layer->biases_gradients[j] += gradient;
		}
	}

	// For each layer of the neural network (except the input layer) (in reverse order), calculate the gradients and update the weights and the biases
	for (int i = network->nb_layers - 1; i > 0; i--) {

		// For each neuron of the layer,
		if (i > 1) {
			for (int j = 0; j < network->layers[i - 1].nb_neurons; j++) {

				// Calculate the gradient of the cost function with respect to the activation value of the neuron
				nn_type gradient = 0.0;
				for (int k = 0; k < network->layers[i].nb_neurons; k++)
					gradient += network->layers[i].weights[k][j]
						* network->layers[i].biases_gradients[k];
				gradient *= network->layers[i - 1].activation_function_derivative(network->layers[i - 1].activations_values[j]);

				// For each input of the neuron, calculate the gradient of the cost function with respect to the weight of the input
				for (int k = 0; k < network->layers[i - 1].nb_inputs_per_neuron; k++)
					network->layers[i - 1].weights_gradients[j][k] += gradient * network->layers[i - 2].activations_values[k];

				// Calculate the gradient of the cost function with respect to the bias of the neuron
				network->layers[i - 1].biases_gradients[j] += gradient;
			}
		}

		// Update the weights and the biases
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
				network->layers[i].weights[j][k] -= (network->learning_rate * network->layers[i].weights_gradients[j][k]) / batch_size;
			network->layers[i].biases[j] -= (network->learning_rate * network->layers[i].biases_gradients[j]) / batch_size;
		}
	}
}

/**
 * @brief Mini-batch Gradient Descent algorithm of the neural network
 * using a batch of inputs and a batch of target outputs
 * 
 * @param network				Pointer to the neural network
 * @param inputs				Pointer to the inputs array
 * @param target_outputs		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 */
void MiniBatchGradientDescentCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **target_outputs, int batch_size) {

	// Prepare the predicted outputs array for the batch
	nn_type **predicted_outputs;
	nn_type *predicted_flat_outputs = try2DFlatMatrixAllocation((void***)&predicted_outputs, batch_size, network->output_layer->nb_neurons, sizeof(nn_type), "MiniBatchGradientDescentCPUSingleThread()");

	// Compute the forward pass to get predictions for the mini-batch
	FeedForwardBatchCPUSingleThread(network, inputs, predicted_outputs, batch_size, 0);

	// Compute the gradients using backpropagation
	BackpropagationCPUSingleThread(network, predicted_outputs, target_outputs, batch_size);

	// Free the predicted outputs array for the batch
	free2DFlatMatrix((void**)predicted_outputs, predicted_flat_outputs, batch_size);
}

/**
 * @brief Utility function to shuffle the training data
 * 
 * @param inputs			Pointer to the inputs array
 * @param target_outputs	Pointer to the target outputs array
 * @param batch_size		Number of samples in the batch
 */
void shuffleTrainingData(nn_type **inputs, nn_type **target_outputs, int batch_size) {

	// Prepare a new array of pointers to the inputs and the target outputs
	nn_type **new_inputs = mallocBlocking(batch_size * sizeof(nn_type *), "shuffleTrainingData()");
	nn_type **new_target_outputs = mallocBlocking(batch_size * sizeof(nn_type *), "shuffleTrainingData()");
	int new_size = 0;

	// While there are samples in the batch,
	int nb_samples = batch_size;
	while (nb_samples > 0) {

		// Select a random sample
		int random_index = rand() % nb_samples;

		// Add the random sample to the new array
		new_inputs[new_size] = inputs[random_index];
		new_target_outputs[new_size] = target_outputs[random_index];
		new_size++;

		// Remove the random sample from the old array by replacing it with the last sample
		inputs[random_index] = inputs[nb_samples - 1];
		target_outputs[random_index] = target_outputs[nb_samples - 1];
		nb_samples--;
	}

	// Copy the new array to the old array
	memcpy(inputs, new_inputs, batch_size * sizeof(nn_type *));
	memcpy(target_outputs, new_target_outputs, batch_size * sizeof(nn_type *));

	// Free the new array
	free(new_inputs);
	free(new_target_outputs);
}

/**
 * @brief Compute the cost of the neural network using a batch of inputs and a batch of target outputs
 * This is used to evaluate the performance of the neural network during the training phase
 * 
 * @param network				Pointer to the neural network
 * @param predicted_outputs		Pointer to the predicted outputs array
 * @param target_outputs		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 * 
 * @return nn_type				Cost of the neural network
 */
nn_type ComputeCostCPUSingleThread(NeuralNetwork *network, nn_type **predicted_outputs, nn_type **target_outputs, int batch_size) {
	
	// Add the cost of each output neuron of each sample of the batch
	nn_type cost = 0.0;
	for (int i = 0; i < batch_size; i++)
		for (int j = 0; j < network->output_layer->nb_neurons; j++)
			cost += network->loss_function(predicted_outputs[i][j], target_outputs[i][j]) / network->output_layer->nb_neurons;

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
 * 
 * @return void
 */
void epochCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int nb_batches, int batch_size, int current_epoch, int nb_epochs, nn_type *current_error, nn_type **test_inputs, nn_type **target_tests, int nb_test_inputs, int verbose) {

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
		
		// Do the mini-batch gradient descent
		MiniBatchGradientDescentCPUSingleThread(network, inputs + first_sample, target + first_sample, nb_samples);
	}

	///// Test the neural network to see the accuracy
	// Use the test inputs to calculate the current error
	if (nb_test_inputs > 0) {

		// Prepare predicted outputs array for the test inputs
		nn_type **predicted;
		nn_type *flat_predicted = try2DFlatMatrixAllocation((void***)&predicted, nb_test_inputs, network->output_layer->nb_neurons, sizeof(nn_type), "TrainCPU(1 thread)");

		// Feed forward the test inputs
		FeedForwardBatchCPUSingleThread(network, test_inputs, predicted, nb_test_inputs, 0);
		
		// Calculate the error of the test inputs using the loss function
		*current_error = ComputeCostCPUSingleThread(network, predicted, target_tests, nb_test_inputs);
		
		// Free the predicted outputs array for the test inputs
		free2DFlatMatrix((void**)predicted, flat_predicted, nb_test_inputs);
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
 * 
 * @return int						Number of epochs done, -1 if there is an error
 */
int TrainCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int test_inputs_percentage, int batch_size, int nb_epochs, nn_type error_target, int verbose) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = nb_epochs != -1 || error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "TrainCPU(1 thread): At least the number of epochs or the error target must be specified!\n");

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
			char benchmark_buffer[256];
			ST_BENCHMARK_SOLO_COUNT(benchmark_buffer,
				epochCPUSingleThread(network, inputs, target, nb_inputs, nb_batches, batch_size, current_epoch, nb_epochs, &current_error, test_inputs, target_tests, nb_test_inputs, verbose),
				"TrainCPU(1 thread): Epoch %d/%d, Error: " ST_COLOR_YELLOW "%.12"NN_FORMAT, 1, 0
			);
			PRINTER(benchmark_buffer, current_epoch, nb_epochs, current_error);
		}

		else
			// Do one epoch of the training
			epochCPUSingleThread(network, inputs, target, nb_inputs, nb_batches, batch_size, current_epoch, nb_epochs, &current_error, test_inputs, target_tests, nb_test_inputs, verbose);
	}

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("TrainCPU(1 thread): Training done!\n");
	return current_epoch;
}






///// Multi-threaded version /////

// Feed forward multi-threaded algorithm
/**
 * @brief Structure representing the arguments of a multi-threaded feed forward algorithm
 * 
 * @param network			Neural network to use (duplicate of the original network, only the activations values are duplicated)
 * @param input				Pointer to the input array
 * @param output_to_fill	Pointer to the output array
 */
struct FeedForwardMultiThreadRoutineArgs {
	NeuralNetwork network;
	nn_type *input;
	nn_type *output_to_fill;
};

/**
 * @brief Routine of a thread of the multi-threaded feed forward algorithm
 * 
 * @param arg	Pointer to the arguments of the multi-threaded feed forward algorithm
 */
thread_return FeedForwardMultiThreadRoutine(thread_param arg) {

	// Get the arguments
	struct FeedForwardMultiThreadRoutineArgs *args = (struct FeedForwardMultiThreadRoutineArgs *)arg;
	NeuralNetwork network = args->network;
	nn_type *input = args->input;
	nn_type *output_to_fill = args->output_to_fill;

	// Feed forward the inputs
	FeedForwardCPUSingleThread(&network, input);

	// Copy the outputs of the neural network to the outputs array
	memcpy(output_to_fill, network.output_layer->activations_values, network.output_layer->nb_neurons * sizeof(nn_type));

	// Free the copy of the neural network & return
	freeSimpleCopyForMultiThreadedFeedForward(network);
	return 0;
}

/**
 * @brief Multi-threading FeedForward algorithm of the neural network using a batch of inputs
 * 
 * @param network		Pointer to the neural network
 * @param inputs		Pointer to the inputs array
 * @param outputs		Pointer to the outputs array
 * @param batch_size	Number of samples in the batch (also number of threads)
 */
int FeedForwardBatchCPUMultiThreads(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size) {

	// Prepare the arguments of the multi-threaded feed forward algorithm and the threads
	struct FeedForwardMultiThreadRoutineArgs *args = mallocBlocking(batch_size * sizeof(struct FeedForwardMultiThreadRoutineArgs), "FeedForwardBatchCPUMultiThreads()");
	pthread_t *threads = mallocBlocking(batch_size * sizeof(pthread_t), "FeedForwardBatchCPUMultiThreads()");

	// For each sample of the batch,
	for (int i = 0; i < batch_size; i++) {

		// Prepare the arguments of the multi-threaded feed forward algorithm
		args[i].network = getSimpleCopyForMultiThreadedFeedForward(*network);
		args[i].input = inputs[i];
		args[i].output_to_fill = outputs[i];

		// Create the thread
		int code = pthread_create(&threads[i], NULL, FeedForwardMultiThreadRoutine, &args[i]);
		ERROR_HANDLE_INT_RETURN_INT(code, "FeedForwardBatchCPUMultiThreads(): Error while creating thread #%d\n", i);
	}

	// Wait for the threads to finish
	for (int i = 0; i < batch_size; i++) {
		int code = pthread_join(threads[i], NULL);
		ERROR_HANDLE_INT_RETURN_INT(code, "FeedForwardBatchCPUMultiThreads(): Error while joining thread #%d\n", i);
	}
	
	// Free the memory
	free(threads);
	free(args);

	// Return
	return 0;
}


// Backpropagation multi-threaded algorithm
/**
 * @brief Structure representing the arguments of the first part of
 * multi-threaded backpropagation algorithm without updating
 * the weights and the biases of the neural network
 * 
 * @param network				Pointer to the neural network
 * @param predicted_outputs		Pointer to the predicted outputs array
 * @param target_outputs		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 * @param start_neuron			Index of the first neuron to process
 * @param end_neuron			Index of the last neuron to process
 */
struct BackpropagationPart1MultiThreadRoutineArgs {
	NeuralNetwork *network;
	nn_type **predicted_outputs;
	nn_type **target_outputs;
	int batch_size;
	int start_neuron;
	int end_neuron;
};

/**
 * @brief Routine of a thread of the first part of
 * multi-threaded backpropagation algorithm without updating
 * the weights and the biases of the neural network
 * 
 * @param arg	Pointer to BackpropagationPart1MultiThreadRoutineArgs
 */
thread_return BackpropagationPart1MultiThreadRoutine(thread_param arg) {
	
	// Get the arguments
	struct BackpropagationPart1MultiThreadRoutineArgs *args = (struct BackpropagationPart1MultiThreadRoutineArgs *)arg;
	NeuralNetwork *network = args->network;
	NeuronLayer *output_layer = network->output_layer;
	NeuronLayer *previous_layer = &network->layers[network->nb_layers - 2];
	nn_type **predicted_outputs = args->predicted_outputs;
	nn_type **target_outputs = args->target_outputs;
	int batch_size = args->batch_size;
	int start_neuron = args->start_neuron;
	int end_neuron = args->end_neuron;

	// For each sample of the batch,
	for (int sample = 0; sample < batch_size; sample++) {

		// For each neuron of the output layer,
		for (int j = start_neuron; j < end_neuron; j++) {

			// Calculate the gradient of the cost function with respect to the activation value of the neuron
			nn_type gradient = network->loss_function_derivative(predicted_outputs[sample][j], target_outputs[sample][j])
				* output_layer->activation_function_derivative(output_layer->activations_values[j]);

			// For each input of the neuron, calculate the gradient of the cost function with respect to the weight of the input
			for (int k = 0; k < output_layer->nb_inputs_per_neuron; k++)
				output_layer->weights_gradients[j][k] += gradient * previous_layer->activations_values[k];

			// Calculate the gradient of the cost function with respect to the bias of the neuron
			output_layer->biases_gradients[j] += gradient;
		}
	}

	// Return
	return 0;
}

/**
 * @brief Structure representing the arguments of the second part of
 * multi-threaded backpropagation algorithm with updating
 * the weights and the biases of the neural network
 * 
 * @param network				Pointer to the neural network
 * @param current_layer_index	Index of the current layer
 * @param first_neuron			Index of the first neuron to process in the current layer
 * @param last_neuron			Index of the last neuron to process in the current layer
 * @param batch_size			Number of samples in the batch
 */
struct BackpropagationPart2MultiThreadRoutineArgs {
	NeuralNetwork *network;
	int current_layer_index;
	int first_neuron;
	int last_neuron;
	int batch_size;
};

/**
 * @brief Routine of a thread of the second part of
 * multi-threaded backpropagation algorithm with updating
 * the weights and the biases of the neural network
 * 
 * @param arg	Pointer to BackpropagationPart2MultiThreadRoutineArgs
 */
thread_return BackpropagationPart2MultiThreadRoutine(thread_param arg) {

	// Get the arguments
	struct BackpropagationPart2MultiThreadRoutineArgs *args = (struct BackpropagationPart2MultiThreadRoutineArgs *)arg;
	NeuralNetwork *network = args->network;
	int current_layer_index = args->current_layer_index;
	int first_neuron = args->first_neuron;
	int last_neuron = args->last_neuron;
	int batch_size = args->batch_size;

	// For each neuron of the current layer,
	if (current_layer_index > 1) {
		for (int j = first_neuron; j < last_neuron; j++) {

			// Calculate the gradient of the cost function with respect to the activation value of the neuron
			nn_type gradient = 0.0;
			for (int k = 0; k < network->layers[current_layer_index].nb_neurons; k++)
				gradient += network->layers[current_layer_index].weights[k][j]
					* network->layers[current_layer_index].biases_gradients[k];
			gradient *= network->layers[current_layer_index - 1].activation_function_derivative(network->layers[current_layer_index - 1].activations_values[j]);

			// For each input of the neuron, calculate the gradient of the cost function with respect to the weight of the input
			for (int k = 0; k < network->layers[current_layer_index - 1].nb_inputs_per_neuron; k++)
				network->layers[current_layer_index - 1].weights_gradients[j][k] += gradient * network->layers[current_layer_index - 2].activations_values[k];

			// Calculate the gradient of the cost function with respect to the bias of the neuron
			network->layers[current_layer_index - 1].biases_gradients[j] += gradient;
		}
	}

	// Update the weights and the biases
	for (int j = first_neuron; j < last_neuron; j++) {
		for (int k = 0; k < network->layers[current_layer_index].nb_inputs_per_neuron; k++)
			network->layers[current_layer_index].weights[j][k] -= (network->learning_rate * network->layers[current_layer_index].weights_gradients[j][k]) / batch_size;
		network->layers[current_layer_index].biases[j] -= (network->learning_rate * network->layers[current_layer_index].biases_gradients[j]) / batch_size;
	}

	// Return
	return 0;
}

/**
 * @brief Multi-threading Backpropagation algorithm of the neural network
 * and update the weights and the biases of the neural network
 * 
 * @param network				Pointer to the neural network
 * @param predicted_outputs		Pointer to the predicted outputs array
 * @param target_outputs		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 */
int BackpropagationCPUMultiThreads(NeuralNetwork *network, nn_type **predicted_outputs, nn_type **target_outputs, int batch_size) {

	// Initialize the gradients of the weights and the biases to 0
	initGradientsNeuralNetwork(network);
	for (int i = 1; i < network->nb_layers; i++) {
		memset(network->layers[i].biases_gradients, 0, network->layers[i].nb_neurons * sizeof(nn_type));
		memset(network->layers[i].weights_gradients_flat, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(nn_type));
	}

	// Get number of threads and prepare the threads
	int nb_threads = getNumberOfThreads();
	pthread_t *threads = mallocBlocking(nb_threads * sizeof(pthread_t), "BackpropagationCPUMultiThreads()");
	struct BackpropagationPart1MultiThreadRoutineArgs *args_part1 = mallocBlocking(nb_threads * sizeof(struct BackpropagationPart1MultiThreadRoutineArgs), "BackpropagationCPUMultiThreads()");
	int part1_nb_neurons_per_thread = network->output_layer->nb_neurons / nb_threads;

	// For each thread of the first part of the multi-threaded backpropagation algorithm
	for (int i = 0; i < nb_threads; i++) {

		// Prepare the arguments
		args_part1[i].network = network;
		args_part1[i].predicted_outputs = predicted_outputs;
		args_part1[i].target_outputs = target_outputs;
		args_part1[i].batch_size = batch_size;
		args_part1[i].start_neuron = i * part1_nb_neurons_per_thread;
		args_part1[i].end_neuron = (i + 1) * part1_nb_neurons_per_thread;
		if (i == nb_threads - 1)
			args_part1[i].end_neuron = network->output_layer->nb_neurons;

		// Create the thread
		int code = pthread_create(&threads[i], NULL, BackpropagationPart1MultiThreadRoutine, &args_part1[i]);
		ERROR_HANDLE_INT_RETURN_INT(code, "BackpropagationCPUMultiThreads(): Error while creating thread #%d\n", i);
	}

	// Prepare the args of the second part of the multi-threaded backpropagation algorithm
	struct BackpropagationPart2MultiThreadRoutineArgs **args_part2;
	struct BackpropagationPart2MultiThreadRoutineArgs *args_part2_flat = try2DFlatMatrixAllocation((void***)&args_part2, network->nb_layers - 1, nb_threads, sizeof(struct BackpropagationPart2MultiThreadRoutineArgs), "BackpropagationCPUMultiThreads()");

	// For each layer of the neural network (except the input layer) (in reverse order), prepare the arguments of the second part of the multi-threaded backpropagation algorithm
	for (int i = network->nb_layers - 1; i > 0; i--) {

		// Calculate the number of neurons to process in the current layer
		int neurons_per_thread = network->layers[i].nb_inputs_per_neuron / nb_threads;

		// For each thread of the second part of the multi-threaded backpropagation algorithm
		for (int j = 0; j < nb_threads; j++) {

			// Prepare the arguments
			args_part2[i - 1][j].network = network;
			args_part2[i - 1][j].current_layer_index = i;
			args_part2[i - 1][j].first_neuron = j * neurons_per_thread;
			args_part2[i - 1][j].last_neuron = (j + 1) * neurons_per_thread;
			args_part2[i - 1][j].batch_size = batch_size;
			if (j == nb_threads - 1)
				args_part2[i - 1][j].last_neuron = network->layers[i].nb_inputs_per_neuron;
		}
	}

	// Wait for the threads of the first part of the multi-threaded backpropagation algorithm to finish
	for (int i = 0; i < nb_threads; i++) {
		int code = pthread_join(threads[i], NULL);
		ERROR_HANDLE_INT_RETURN_INT(code, "BackpropagationCPUMultiThreads(): Error while joining thread #%d\n", i);
	}

	// For each layer of the neural network (except the input layer) (in reverse order), create the threads of the second part of the multi-threaded backpropagation algorithm
	for (int i = network->nb_layers - 1; i > 0; i--) {

		// Create the threads of the second part of the multi-threaded backpropagation algorithm
		for (int j = 0; j < nb_threads; j++) {
			int code = pthread_create(&threads[j], NULL, BackpropagationPart2MultiThreadRoutine, &args_part2[i - 1][j]);
			ERROR_HANDLE_INT_RETURN_INT(code, "BackpropagationCPUMultiThreads(): Error while creating thread #%d\n", j);
		}

		// Wait for the threads of the second part of the multi-threaded backpropagation algorithm to finish
		for (int j = 0; j < nb_threads; j++) {
			int code = pthread_join(threads[j], NULL);
			ERROR_HANDLE_INT_RETURN_INT(code, "BackpropagationCPUMultiThreads(): Error while joining thread #%d\n", j);
		}
	}

	// Free the memory
	free(threads);
	free(args_part1);
	free2DFlatMatrix((void**)args_part2, args_part2_flat, network->nb_layers - 1);

	// Return
	return 0;
}


/**
 * @brief Mini-batch Gradient Descent algorithm of the neural network
 * using a batch of inputs and a batch of target outputs (multi-threaded version)
 * 
 * @param network				Pointer to the neural network
 * @param inputs				Pointer to the inputs array
 * @param target_outputs		Pointer to the target outputs array
 * @param batch_size			Number of samples in the batch
 */
void MiniBatchGradientDescentCPUMultiThreads(NeuralNetwork *network, nn_type **inputs, nn_type **target_outputs, int batch_size) {

	// Prepare the predicted outputs array for the batch
	nn_type **predicted_outputs;
	nn_type *predicted_flat_outputs = try2DFlatMatrixAllocation((void***)&predicted_outputs, batch_size, network->output_layer->nb_neurons, sizeof(nn_type), "MiniBatchGradientDescentCPUSingleThread()");

	// Compute the forward pass to get predictions for the mini-batch
	FeedForwardBatchCPUMultiThreads(network, inputs, predicted_outputs, batch_size);

	// Compute the gradients using backpropagation
	BackpropagationCPUMultiThreads(network, predicted_outputs, target_outputs, batch_size);

	// Free the predicted outputs array for the batch
	free2DFlatMatrix((void**)predicted_outputs, predicted_flat_outputs, batch_size);
}

/**
 * @brief One epoch of the training of the neural network with the CPU (Multi-threads)
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
 * 
 * @return void
 */
void epochCPUMultiThreads(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int nb_batches, int batch_size, int current_epoch, int nb_epochs, nn_type *current_error, nn_type **test_inputs, nn_type **target_tests, int nb_test_inputs, int verbose) {

	// Shuffle the training data
	shuffleTrainingData(inputs, target, nb_inputs);

	// Get prefix of the verbose
	char verbose_prefix[32];
	sprintf(verbose_prefix, "TrainCPU(%d threads)", getNumberOfThreads());

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
			DEBUG_PRINT("%s: Epoch %d/%d,\tBatch %d/%d,\tSamples %d-%d/%d\n", verbose_prefix, current_epoch, nb_epochs, current_batch + 1, nb_batches, first_sample + 1, last_sample + 1, nb_inputs);
		
		// Do the mini-batch gradient descent
		MiniBatchGradientDescentCPUMultiThreads(network, inputs + first_sample, target + first_sample, nb_samples);
	}

	///// Test the neural network to see the accuracy
	// Use the test inputs to calculate the current error
	if (nb_test_inputs > 0) {

		// Prepare predicted outputs array for the test inputs
		nn_type **predicted;
		nn_type *flat_predicted = try2DFlatMatrixAllocation((void***)&predicted, nb_test_inputs, network->output_layer->nb_neurons, sizeof(nn_type), verbose_prefix);

		// Feed forward the test inputs
		FeedForwardBatchCPUMultiThreads(network, test_inputs, predicted, nb_test_inputs);
		
		// Calculate the error of the test inputs using the loss function
		*current_error = ComputeCostCPUSingleThread(network, predicted, target_tests, nb_test_inputs);
		
		// Free the predicted outputs array for the test inputs
		free2DFlatMatrix((void**)predicted, flat_predicted, nb_test_inputs);
	}
}

/**
 * @brief Train the neural network with the CPU (Multi-threads)
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
 * 
 * @return int						Number of epochs done, -1 if there is an error
 */
int TrainCPUMultiThreads(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int test_inputs_percentage, int batch_size, int nb_epochs, nn_type error_target, int verbose) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = nb_epochs != -1 || error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "TrainCPU(%d threads): At least the number of epochs or the error target must be specified!\n", getNumberOfThreads());
	int nb_threads = getNumberOfThreads();

	// Prepare the test inputs
	int nb_test_inputs = nb_inputs * test_inputs_percentage / 100;
	nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &inputs[nb_inputs];
	nn_type **target_tests = &target[nb_inputs];
	if (verbose > 0)
		INFO_PRINT("TrainCPU(%d threads): %d inputs, %d test inputs\n", nb_threads, nb_inputs, nb_test_inputs);

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
			char benchmark_buffer[256];
			ST_BENCHMARK_SOLO_COUNT(benchmark_buffer,
				epochCPUMultiThreads(network, inputs, target, nb_inputs, nb_batches, batch_size, current_epoch, nb_epochs, &current_error, test_inputs, target_tests, nb_test_inputs, verbose),
				"TrainCPU(%d threads): Epoch %d/%d, Error: " ST_COLOR_YELLOW "%.12"NN_FORMAT, nb_threads, 0
			);
			PRINTER(benchmark_buffer, current_epoch, nb_epochs, current_error);
		}

		else
			// Do one epoch of the training
			epochCPUMultiThreads(network, inputs, target, nb_inputs, nb_batches, batch_size, current_epoch, nb_epochs, &current_error, test_inputs, target_tests, nb_test_inputs, verbose);
	}

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("TrainCPU(%d threads): Training done!\n", nb_threads);
	return current_epoch;
}

