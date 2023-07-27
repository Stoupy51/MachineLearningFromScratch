
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
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {
				nn_type input_value = network->layers[i - 1].activations_values[k];
				nn_type weight = network->layers[i].weights[j][k];
				input_sum += input_value * weight;
			}

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
void FeedForwardBatchCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size) {

	// Local variables
	size_t output_layer_size = network->output_layer->nb_neurons * sizeof(nn_type);

	// For each sample of the batch,
	for (int i = 0; i < batch_size; i++) {

		// Feed forward the inputs
		FeedForwardCPUSingleThread(network, inputs[i]);

		// Copy the outputs of the neural network to the outputs array
		memcpy(outputs[i], network->output_layer->activations_values, output_layer_size);
	}
}

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
thread_return_type FeedForwardMultiThreadRoutine(thread_param_type arg) {

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
void FeedForwardBatchCPUMultiThreads(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size) {

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
		pthread_create(&threads[i], NULL, FeedForwardMultiThreadRoutine, &args[i]);
	}

	// Wait for the threads to finish
	for (int i = 0; i < batch_size; i++)
		pthread_join(threads[i], NULL);
	
	// Free the memory
	free(threads);
	free(args);
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
	for (int i = 0; i < network->nb_layers; i++) {
		memset(network->layers[i].biases_gradients, 0, network->layers[i].nb_neurons * sizeof(nn_type));
		memset(network->layers[i].weights_gradients_flat, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(nn_type));
	}

	// For each sample of the batch,
	for (int sample = 0; sample < batch_size; sample++) {

		// For each neuron of the output layer,
		for (int i = 0; i < network->output_layer->nb_neurons; i++) {

			// Calculate the gradient of the cost function with respect to the activation value of the neuron
			nn_type loss_derivative = network->loss_function_derivative(predicted_outputs[sample][i], target_outputs[sample][i]);
			nn_type activation_derivative = network->output_layer->activation_function_derivative(network->output_layer->activations_values[i]);
			nn_type gradient = loss_derivative * activation_derivative;

			// For each input of the neuron,
			for (int j = 0; j < network->output_layer->nb_inputs_per_neuron; j++) {

				// Calculate the gradient of the cost function with respect to the weight of the input
				nn_type input_value = network->layers[network->nb_layers - 2].activations_values[j];
				network->output_layer->weights_gradients[i][j] += gradient * input_value;
			}

			// Calculate the gradient of the cost function with respect to the bias of the neuron
			network->output_layer->biases_gradients[i] += gradient;
		}

		// For each layer of the neural network (except the output layer and the input layer) (in reverse order),
		for (int i = network->nb_layers - 2; i > 0; i--) {

			// For each neuron of the layer,
			for (int j = 0; j < network->layers[i].nb_neurons; j++) {

				// Calculate the gradient of the cost function with respect to the activation value of the neuron
				nn_type gradient = 0.0;
				for (int k = 0; k < network->layers[i + 1].nb_neurons; k++) {
					nn_type weight = network->layers[i + 1].weights[k][j];
					nn_type activation_derivative = network->layers[i].activation_function_derivative(network->layers[i].activations_values[j]);
					gradient += weight * activation_derivative * network->layers[i + 1].biases_gradients[k];
				}

				// For each input of the neuron,
				for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {

					// Calculate the gradient of the cost function with respect to the weight of the input
					nn_type input_value = network->layers[i - 1].activations_values[k];
					network->layers[i].weights_gradients[j][k] += gradient * input_value;
				}

				// Calculate the gradient of the cost function with respect to the bias of the neuron
				network->layers[i].biases_gradients[j] += gradient;
			}
		}
	}

	// Update the weights and the biases of the neural network
	for (int i = 1; i < network->nb_layers; i++) {

		// Update the weights
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {
				nn_type weight_gradient = network->layers[i].weights_gradients[j][k];
				network->layers[i].weights[j][k] -= (network->learning_rate * weight_gradient) / batch_size;
			}
		}

		// Update the biases
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {
			nn_type bias_gradient = network->layers[i].biases_gradients[j];
			network->layers[i].biases[j] -= (network->learning_rate * bias_gradient) / batch_size;
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
	FeedForwardBatchCPUSingleThread(network, inputs, predicted_outputs, batch_size);

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
	
	// Local variables
	nn_type cost = 0.0;

	// Add the cost of each output neuron of each sample of the batch
	for (int i = 0; i < batch_size; i++)
		for (int j = 0; j < network->output_layer->nb_neurons; j++)
			cost += network->loss_function(predicted_outputs[i][j], target_outputs[i][j]);

	// Return the cost of the neural network
	return cost / batch_size;
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
 * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: normal verbose + benchmark, 4: all)
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
			if (verbose == 2 || verbose > 3)
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
			FeedForwardBatchCPUSingleThread(network, test_inputs, predicted, nb_test_inputs);
			
			// Calculate the error of the test inputs using the loss function
			for (int i = 0; i < nb_test_inputs; i++)
				for (int j = 0; j < network->output_layer->nb_neurons; j++)
					current_error += network->loss_function(predicted[i][j], target_tests[i][j]);
			
			// Free the predicted outputs array for the test inputs
			free2DFlatMatrix((void**)predicted, flat_predicted, nb_test_inputs);
		}

		// Verbose
		current_error /= nb_inputs;
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == nb_epochs || current_epoch % 10 == 0)) || verbose == 2)
			DEBUG_PRINT("TrainCPU(1 thread): Epoch %d/%d, Error: %.12"NN_FORMAT"\n", current_epoch, nb_epochs, current_error);
	}

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("TrainCPU(1 thread): Training done!\n");
	return current_epoch;
}







