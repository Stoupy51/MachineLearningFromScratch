
#include "training_cpu.h"
#include "../universal_utils.h"
#include "../st_benchmark.h"

/**
 * @brief Feed forward algorithm of the neural network
 * 
 * @details i = Index of the selected layer
 * @details j = Index of the selected neuron
 * @details k = Index of the selected input
 * 
 * @param network		Pointer to the neural network
 * @param input			Pointer to the input array (nn_type), must be the same size as the input layer
 * @param output		Pointer to the output array (nn_type), must be the same size as the output layer
 * 
 * @return void
 */
void NeuralNetworkFeedForwardCPUSingleThread(NeuralNetwork *network, nn_type *input, nn_type *output) {

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

	// Copy the outputs of the output layer to the outputs array
	size_t output_layer_size = network->output_layer->nb_neurons * sizeof(nn_type);
	memcpy(output, network->output_layer->activations_values, output_layer_size);
}

/**
 * @brief Start backpropagation algorithm of the neural network
 * takes a batch of expected outputs and adjusts the deltas of the output layer
 * 
 * @param network		Pointer to the neural network
 * @param predicted		Pointer to the predicted outputs array (nn_type), must be the same size as the output layer
 * @param expected		Pointer to the expected outputs array (nn_type), must be the same size as the output layer
 * @param batch_size	Number of samples in the batch
 * 
 * @return void
 */
void NeuralNetworkStartBackPropagationCPUSingleThread(NeuralNetwork *network, nn_type **predicted, nn_type **expected, int batch_size) {

	// For each neuron of the output layer,
	for (int i = 0; i < network->output_layer->nb_neurons; i++) {

		// Calculate the error of the neuron (expected - predicted)
		nn_type error = 0;
		for (int batch = 0; batch < batch_size; batch++)
			error += expected[batch][i] - predicted[batch][i];
		error /= batch_size;

		// Calculate the derivative of the activation function of the neuron
		nn_type derivative = 0;
		for (int batch = 0; batch < batch_size; batch++)
			derivative += network->output_layer->activation_function_derivative(predicted[batch][i]);

		// Calculate the delta of the neuron (error * derivative)
		network->output_layer->deltas[i] = error * derivative;
	}
}

/**
 * @brief Finish backpropagation algorithm of the neural network
 * by adjusting the deltas of the hidden layers
 * 
 * @param network		Pointer to the neural network
 * 
 * @return void
 */
void NeuralNetworkFinishBackPropagationCPUSingleThread(NeuralNetwork *network) {

	// For each hidden layer of the neural network, starting from the last hidden layer, add the deltas of the layer
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// For each neuron of the layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the error of the neuron (sum( weight * delta of the next layer ))
			nn_type error = 0;
			for (int k = 0; k < network->layers[i + 1].nb_neurons; k++) {

				// Get the weight of the next layer neuron linked to the current neuron
				// (Not [j][k] because it's reversed compared to the feed forward algorithm (checking next layer instead of previous layer))
				nn_type weight = network->layers[i + 1].weights[k][j];

				// Get the delta of the next layer neuron
				nn_type delta = network->layers[i + 1].deltas[k];

				// Add the weight * delta to the error
				error += weight * delta;
			}

			// Calculate the derivative of the activation function of the neuron
			nn_type input = network->layers[i].activations_values[j];
			nn_type derivative = network->layers[i].activation_function_derivative(input);

			// Calculate the delta of the neuron (error * derivative)
			network->layers[i].deltas[j] += error * derivative;
		}
	}
}

/**
 * @brief Update the weights of the neural network
 * 
 * @param network		Pointer to the neural network
 * 
 * @return void
 */
void NeuralNetworkUpdateWeightsCPUSingleThread(NeuralNetwork *network) {

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Variables for easier reading
			nn_type learning_rate = network->learning_rate;
			nn_type delta = network->layers[i].deltas[j];

			// For each weight of the current neuron,
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {

				// Variable for easier reading
				nn_type activation_value = network->layers[i - 1].activations_values[k];

				// Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
				network->layers[i].weights[j][k] += learning_rate * delta * activation_value;
			}
			// Update the bias (bias + (learning_rate * delta of the current neuron))
			network->layers[i].biases[j] += learning_rate * delta;
		}
	}
}


/**
 * @brief Train the neural network with the CPU (Single core)
 * by using a batch of inputs and a batch of expected outputs,
 * a number of epochs and a target error value
 * 
 * @param network					Pointer to the neural network
 * 
 * @param inputs					Pointer to the inputs array (nn_type), must be the same size as the input layer
 * @param expected					Pointer to the expected outputs array (nn_type), must be the same size as the output layer
 * @param nb_inputs					Number of samples in the inputs array and in the expected outputs array
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
int NeuralNetworkTrainCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **expected, int nb_inputs, int test_inputs_percentage, int batch_size, int nb_epochs, nn_type error_target, int verbose) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = nb_epochs != -1 || error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "NeuralNetworkTrainCPU(1 thread): At least the number of epochs or the error target must be specified!\n");

	// Prepare the test inputs
	int nb_test_inputs = nb_inputs * test_inputs_percentage / 100;
	nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &inputs[nb_inputs];
	nn_type **excepted_tests = &expected[nb_inputs];
	if (verbose > 0)
		INFO_PRINT("NeuralNetworkTrainCPU(1 thread): %d inputs, %d test inputs\n", nb_inputs, nb_test_inputs);

	// Local variables
	int current_epoch = 0;
	nn_type current_error = 100.0;
	int nb_batches = nb_inputs / batch_size + (nb_inputs % batch_size != 0);

	// Training
	while (current_epoch < nb_epochs && current_error > error_target) {

		// Reset the current error and increment the current epoch
		current_error = 0;
		current_epoch++;

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
				DEBUG_PRINT("NeuralNetworkTrainCPU(1 thread): Epoch %d/%d,\tBatch %d/%d,\tSamples %d-%d/%d\n", current_epoch, nb_epochs, current_batch + 1, nb_batches, first_sample + 1, last_sample + 1, nb_inputs);
			
			// Prepare predicted outputs array for the current batch
			nn_type **predicted = mallocBlocking(nb_samples * sizeof(nn_type *), "NeuralNetworkTrainCPU(1 thread)");
			for (int i = 0; i < nb_samples; i++)
				predicted[i] = mallocBlocking(network->output_layer->nb_neurons * sizeof(nn_type), "NeuralNetworkTrainCPU(1 thread)");

			// Feed forward the current batches
			for (int i = 0; i < nb_samples; i++)
				NeuralNetworkFeedForwardCPUSingleThread(network, inputs[first_sample + i], predicted[i]);
			
			// Reset deltas of all the layers (except the input layer)
			for (int i = 1; i < network->nb_layers; i++)
				memset(network->layers[i].deltas, 0, network->layers[i].nb_neurons * sizeof(nn_type));

			// Backpropagation
			NeuralNetworkStartBackPropagationCPUSingleThread(network, predicted, expected + first_sample, nb_samples);
			NeuralNetworkFinishBackPropagationCPUSingleThread(network);

			// Free the predicted outputs array for the current batch
			for (int i = 0; i < nb_samples; i++)
				free(predicted[i]);
			free(predicted);

			// Update the weights of the neural network
			NeuralNetworkUpdateWeightsCPUSingleThread(network);
		}

		// Use the test inputs to calculate the current error
		if (nb_test_inputs > 0) {

			// Prepare predicted outputs array for the test inputs
			nn_type **predicted = mallocBlocking(nb_test_inputs * sizeof(nn_type *), "NeuralNetworkTrainCPU(1 thread)");
			for (int i = 0; i < nb_test_inputs; i++)
				predicted[i] = mallocBlocking(network->output_layer->nb_neurons * sizeof(nn_type), "NeuralNetworkTrainCPU(1 thread)");

			// Feed forward the test inputs
			for (int i = 0; i < nb_test_inputs; i++)
				NeuralNetworkFeedForwardCPUSingleThread(network, test_inputs[i], predicted[i]);
			
			// Calculate the error of the test inputs using the loss function
			for (int i = 0; i < nb_test_inputs; i++)
				current_error += network->loss_function(predicted[i], excepted_tests[i], network->output_layer->nb_neurons);
			
			// Free the predicted outputs array for the test inputs
			for (int i = 0; i < nb_test_inputs; i++)
				free(predicted[i]);
			free(predicted);
		}

		// Verbose
		current_error /= nb_inputs;
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == nb_epochs || current_epoch % 10 == 0)) || verbose == 2)
			DEBUG_PRINT("NeuralNetworkTrainCPU(1 thread): Epoch %d/%d, Error: %.12"NN_FORMAT"\n", current_epoch, nb_epochs, current_error);
	}

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("NeuralNetworkTrainCPU(1 thread): Training done!\n");
	return current_epoch;
}


#include "../universal_pthread.h"

/**
 * @brief Structure used to pass the data to the FeedForwardThread function
 * 
 * @param current_layer					Pointer to the current layer
 * @param previous_activations_values	Pointer to the activations values of the previous layer
 * @param start_neuron					Index of the first neuron to process
 * @param end_neuron					Index of the last neuron to process
 */
struct FeedForwardThreadData {
	NeuronLayer *current_layer;
	nn_type *previous_activations_values;
	int start_neuron;
	int end_neuron;
};

/**
 * @brief Feed forward algorithm of the neural network
 * using multiple threads
 * 
 * @param thread_data	Pointer to the FeedForwardThreadData structure
*/
thread_return_type FeedForwardThread(thread_param_type thread_data) {

	// Get the thread data
	struct FeedForwardThreadData *data = (struct FeedForwardThreadData *)thread_data;
	NeuronLayer *current_layer = data->current_layer;
	nn_type *previous_activations_values = data->previous_activations_values;
	int start_neuron = data->start_neuron;
	int end_neuron = data->end_neuron;

	// For each neuron of the layer,
	for (int j = start_neuron; j < end_neuron; j++) {

		// Calculate the sum of the inputs multiplied by the weights
		nn_type input_sum = current_layer->biases[j];	// Add the bias to the sum
		for (int k = 0; k < current_layer->nb_inputs_per_neuron; k++)
			input_sum += previous_activations_values[k] * current_layer->weights[j][k];

		// Activate the neuron with the activation function
		current_layer->activations_values[j] = current_layer->activation_function(input_sum);
	}
	return 0;
}

/**
 * @brief Feed forward algorithm of the neural network
 * using multiple threads to process the neurons
 * 
 * @details i = Index of the selected layer
 * @details j = Index of the selected neuron
 * @details k = Index of the selected input
 * 
 * @param network		Pointer to the neural network
 * @param input			Pointer to the input array (nn_type), must be the same size as the input layer
 * @param output		Pointer to the output array (nn_type), must be the same size as the output layer
 * 
 * @return void
 */
void NeuralNetworkFeedForwardCPUMultiThreads(NeuralNetwork *network, nn_type *input, nn_type *output) {

	// Copy the inputs to the input layer of the neural network
	size_t input_layer_size = network->input_layer->nb_neurons * sizeof(nn_type);
	memcpy(network->input_layer->activations_values, input, input_layer_size);

	// Get number of threads
	int nb_threads = getNumberOfThreads();

	// Create the threads data and the threads array
	struct FeedForwardThreadData *threads_data = mallocBlocking(nb_threads * sizeof(struct FeedForwardThreadData), "NeuralNetworkFeedForwardCPUMultiCores");
	pthread_t *threads = mallocBlocking(nb_threads * sizeof(pthread_t), "NeuralNetworkFeedForwardCPUMultiCores");

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// Create the threads
		int nb_neurons_per_thread = network->layers[i].nb_neurons / nb_threads;
		for (int j = 0; j < nb_threads; j++) {
			
			// Prepare the thread data
			threads_data[j].current_layer = &network->layers[i];
			threads_data[j].previous_activations_values = network->layers[i - 1].activations_values;
			threads_data[j].start_neuron = j * nb_neurons_per_thread;
			if (j == nb_threads - 1)
				threads_data[j].end_neuron = network->layers[i].nb_neurons;	// The last thread must process the remaining neurons
			else
				threads_data[j].end_neuron = threads_data[j].start_neuron + nb_neurons_per_thread;
			
			// Create the thread
			pthread_create(&threads[j], NULL, FeedForwardThread, &threads_data[j]);
		}

		// Wait for the threads to finish
		for (int j = 0; j < nb_threads; j++)
			pthread_join(threads[j], NULL);
	}

	// Free the threads data and the threads array
	free(threads_data);
	free(threads);

	// Copy the outputs of the output layer to the outputs array
	size_t output_layer_size = network->output_layer->nb_neurons * sizeof(nn_type);
	memcpy(output, network->output_layer->activations_values, output_layer_size);
}

/**
 * @brief Structure used to pass the data to the StartBackPropagationThread function
 * 
 * @param output_layer					Pointer to the output layer
 * @param predicted						Pointer to the predicted outputs array
 * @param expected						Pointer to the expected outputs array
 * @param batch_size					Number of samples in the batch
 * @param start_neuron					Index of the first neuron to process
 * @param end_neuron					Index of the last neuron to process
 */
struct StartBackPropagationThreadData {
	NeuronLayer *output_layer;
	nn_type **predicted;
	nn_type **expected;
	int batch_size;
	int start_neuron;
	int end_neuron;
};

/**
 * @brief Start backpropagation algorithm of the neural network
 * using multiple threads
 * 
 * @param thread_data	Pointer to the StartBackPropagationThreadData structure
*/
thread_return_type StartBackPropagationThread(thread_param_type thread_data) {

	// Get the thread data
	struct StartBackPropagationThreadData *data = (struct StartBackPropagationThreadData *)thread_data;
	NeuronLayer *output_layer = data->output_layer;
	nn_type **predicted = data->predicted;
	nn_type **expected = data->expected;
	int batch_size = data->batch_size;
	int start_neuron = data->start_neuron;
	int end_neuron = data->end_neuron;

	// For each neuron of the output layer,
	for (int i = start_neuron; i < end_neuron; i++) {

		// Calculate the error of the neuron (expected - predicted)
		nn_type error = 0;
		for (int batch = 0; batch < batch_size; batch++)
			error += expected[batch][i] - predicted[batch][i];

		// Calculate the derivative of the activation function of the neuron
		nn_type derivative = 0;
		for (int batch = 0; batch < batch_size; batch++)
			derivative += output_layer->activation_function_derivative(predicted[batch][i]);

		// Calculate the delta of the neuron (error * derivative)
		output_layer->deltas[i] = (error / batch_size) * derivative;
	}
	return 0;
}

/**
 * @brief Start backpropagation algorithm of the neural network
 * using multiple threads to process the neurons
 * 
 * @param network		Pointer to the neural network
 * @param predicted		Pointer to the predicted outputs array (nn_type), must be the same size as the output layer
 * @param expected		Pointer to the expected outputs array (nn_type), must be the same size as the output layer
 * @param batch_size	Number of samples in the batch
 * 
 * @return void
 */
void NeuralNetworkStartBackPropagationCPUMultiThreads(NeuralNetwork *network, nn_type **predicted, nn_type **expected, int batch_size) {

	// Get number of threads
	int nb_threads = getNumberOfThreads();

	// Create the threads data and the threads array
	struct StartBackPropagationThreadData *threads_data = mallocBlocking(nb_threads * sizeof(struct StartBackPropagationThreadData), "NeuralNetworkStartBackPropagationCPUMultiCores");
	pthread_t *threads = mallocBlocking(nb_threads * sizeof(pthread_t), "NeuralNetworkStartBackPropagationCPUMultiCores");

	// Create the threads
	int nb_neurons_per_thread = network->output_layer->nb_neurons / nb_threads;
	for (int j = 0; j < nb_threads; j++) {
		
		// Prepare the thread data
		threads_data[j].output_layer = network->output_layer;
		threads_data[j].predicted = predicted;
		threads_data[j].expected = expected;
		threads_data[j].batch_size = batch_size;
		threads_data[j].start_neuron = j * nb_neurons_per_thread;
		if (j == nb_threads - 1)
			threads_data[j].end_neuron = network->output_layer->nb_neurons;	// The last thread must process the remaining neurons
		else
			threads_data[j].end_neuron = threads_data[j].start_neuron + nb_neurons_per_thread;
		
		// Create the thread
		pthread_create(&threads[j], NULL, StartBackPropagationThread, &threads_data[j]);
	}

	// Wait for the threads to finish
	for (int j = 0; j < nb_threads; j++)
		pthread_join(threads[j], NULL);
	
	// Free the threads data and the threads array
	free(threads_data);
	free(threads);
}

/**
 * @brief Structure used to pass the data to the FinishBackPropagationThread function
 * 
 * @param current_layer					Pointer to the current layer
 * @param next_layer					Pointer to the next layer
 * @param start_neuron					Index of the first neuron to process
 * @param end_neuron					Index of the last neuron to process
 */
struct FinishBackPropagationThreadData {
	NeuronLayer *current_layer;
	NeuronLayer *next_layer;
	int start_neuron;
	int end_neuron;
};

/**
 * @brief Finish backpropagation algorithm of the neural network
 * using multiple threads
 * 
 * @param thread_data	Pointer to the FinishBackPropagationThreadData structure
*/
thread_return_type FinishBackPropagationThread(thread_param_type thread_data) {

	// Get the thread data
	struct FinishBackPropagationThreadData *data = (struct FinishBackPropagationThreadData *)thread_data;
	NeuronLayer *current_layer = data->current_layer;
	NeuronLayer *next_layer = data->next_layer;
	int start_neuron = data->start_neuron;
	int end_neuron = data->end_neuron;

	// For each neuron of the layer,
	for (int j = start_neuron; j < end_neuron; j++) {

		// Calculate the error of the neuron (sum( weight * delta of the next layer ))
		nn_type error = 0;
		for (int k = 0; k < next_layer->nb_neurons; k++)
			error += next_layer->weights[k][j] * next_layer->deltas[k];

		// Calculate the delta of the neuron (error * derivative)
		current_layer->deltas[j] += error * current_layer->activation_function_derivative(current_layer->activations_values[j]);
	}
	return 0;
}

/**
 * @brief Finish backpropagation algorithm of the neural network
 * using multiple threads to process the neurons
 * 
 * @param network		Pointer to the neural network
 * 
 * @return void
 */
void NeuralNetworkFinishBackPropagationCPUMultiThreads(NeuralNetwork *network) {

	// Get number of threads
	int nb_threads = getNumberOfThreads();

	// Create the threads data and the threads array
	struct FinishBackPropagationThreadData *threads_data = mallocBlocking(nb_threads * sizeof(struct FinishBackPropagationThreadData), "NeuralNetworkFinishBackPropagationCPUMultiCores");
	pthread_t *threads = mallocBlocking(nb_threads * sizeof(pthread_t), "NeuralNetworkFinishBackPropagationCPUMultiCores");

	// For each hidden layer of the neural network, starting from the last hidden layer, add the deltas of the layer
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// Create the threads
		int nb_neurons_per_thread = network->layers[i].nb_neurons / nb_threads;
		for (int j = 0; j < nb_threads; j++) {
			
			// Prepare the thread data
			threads_data[j].current_layer = &network->layers[i];
			threads_data[j].next_layer = &network->layers[i + 1];
			threads_data[j].start_neuron = j * nb_neurons_per_thread;
			if (j == nb_threads - 1)
				threads_data[j].end_neuron = network->layers[i].nb_neurons;	// The last thread must process the remaining neurons
			else
				threads_data[j].end_neuron = threads_data[j].start_neuron + nb_neurons_per_thread;
			
			// Create the thread
			pthread_create(&threads[j], NULL, FinishBackPropagationThread, &threads_data[j]);
		}

		// Wait for the threads to finish
		for (int j = 0; j < nb_threads; j++)
			pthread_join(threads[j], NULL);
	}

	// Free the threads data and the threads array
	free(threads_data);
	free(threads);
}

/**
 * @brief Structure used to pass the data to the UpdateWeightsThread function
 * 
 * @param current_layer					Pointer to the current layer
 * @param previous_activations_values	Pointer to the activations values of the previous layer
 * @param learning_rate					Learning rate of the neural network
 * @param start_neuron					Index of the first neuron to process
 * @param end_neuron					Index of the last neuron to process
 */
struct UpdateWeightsThreadData {
	NeuronLayer *current_layer;
	nn_type *previous_activations_values;
	nn_type learning_rate;
	int start_neuron;
	int end_neuron;
};

/**
 * @brief Update the weights of the neural network
 * using multiple threads
 * 
 * @param thread_data	Pointer to the UpdateWeightsThreadData structure
*/
thread_return_type UpdateWeightsThread(thread_param_type thread_data) {

	// Get the thread data
	struct UpdateWeightsThreadData *data = (struct UpdateWeightsThreadData *)thread_data;
	NeuronLayer *current_layer = data->current_layer;
	nn_type *previous_activations_values = data->previous_activations_values;
	nn_type learning_rate = data->learning_rate;
	int start_neuron = data->start_neuron;
	int end_neuron = data->end_neuron;

	// For each neuron of the current layer,
	for (int j = start_neuron; j < end_neuron; j++) {

		// For each weight of the current neuron, update (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
		for (int k = 0; k < current_layer->nb_inputs_per_neuron; k++)
			current_layer->weights[j][k] += learning_rate * current_layer->deltas[j] * previous_activations_values[k];

		// Update the bias (bias + (learning_rate * delta of the current neuron))
		current_layer->biases[j] += learning_rate * current_layer->deltas[j];
	}
	return 0;
}

/**
 * @brief Update the weights of the neural network
 * using multiple threads to process the neurons
 * 
 * @param network		Pointer to the neural network
 * 
 * @return void
 */
void NeuralNetworkUpdateWeightsCPUMultiThreads(NeuralNetwork *network) {

	// Get number of threads
	int nb_threads = getNumberOfThreads();

	// Create the threads data and the threads array
	struct UpdateWeightsThreadData *threads_data = mallocBlocking(network->nb_layers * sizeof(struct UpdateWeightsThreadData), "NeuralNetworkUpdateWeightsCPUMultiCores");
	pthread_t *threads = mallocBlocking(network->nb_layers * sizeof(pthread_t), "NeuralNetworkUpdateWeightsCPUMultiCores");

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// Create the threads
		int nb_neurons_per_thread = network->layers[i].nb_neurons / nb_threads;
		for (int j = 0; j < nb_threads; j++) {
			
			// Prepare the thread data
			threads_data[j].current_layer = &network->layers[i];
			threads_data[j].previous_activations_values = network->layers[i - 1].activations_values;
			threads_data[j].learning_rate = network->learning_rate;
			threads_data[j].start_neuron = j * nb_neurons_per_thread;
			if (j == nb_threads - 1)
				threads_data[j].end_neuron = network->layers[i].nb_neurons;	// The last thread must process the remaining neurons
			else
				threads_data[j].end_neuron = threads_data[j].start_neuron + nb_neurons_per_thread;
			
			// Create the thread
			pthread_create(&threads[j], NULL, UpdateWeightsThread, &threads_data[j]);
		}

		// Wait for the threads to finish
		for (int j = 0; j < nb_threads; j++)
			pthread_join(threads[j], NULL);
	}

	// Free the threads data and the threads array
	free(threads_data);
	free(threads);
}


/**
 * @brief Train the neural network with the CPU (Multiple cores)
 * by using a batch of inputs and a batch of expected outputs,
 * a number of epochs and a target error value
 * 
 * @param network					Pointer to the neural network
 * 
 * @param inputs					Pointer to the inputs array (nn_type), must be the same size as the input layer
 * @param expected					Pointer to the expected outputs array (nn_type), must be the same size as the output layer
 * @param nb_inputs					Number of samples in the inputs array and in the expected outputs array
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
int NeuralNetworkTrainCPUMultiThreads(NeuralNetwork *network, nn_type **inputs, nn_type **expected, int nb_inputs, int test_inputs_percentage, int batch_size, int nb_epochs, nn_type error_target, int verbose) {

	// Get number of threads & prepare prefix
	int nb_threads = getNumberOfThreads();
	char prefix[100];
	sprintf(prefix, "NeuralNetworkTrainCPU(%d threads)", nb_threads);

	// Check if at least one of the two parameters is specified
	int boolean_parameters = nb_epochs != -1 || error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "%s: At least the number of epochs or the error target must be specified!\n", prefix);

	// Prepare the test inputs
	int nb_test_inputs = nb_inputs * test_inputs_percentage / 100;
	nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &inputs[nb_inputs];
	nn_type **excepted_tests = &expected[nb_inputs];
	if (verbose > 0)
		INFO_PRINT("%s: %d inputs, %d test inputs\n", prefix, nb_inputs, nb_test_inputs);

	// Local variables
	int current_epoch = 0;
	nn_type current_error = 100.0;
	int nb_batches = nb_inputs / batch_size + (nb_inputs % batch_size != 0);

	// Training
	while (current_epoch < nb_epochs && current_error > error_target) {

		// Reset the current error and increment the current epoch
		current_error = 0;
		current_epoch++;

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
				DEBUG_PRINT("%s: Epoch %d/%d,\tBatch %d/%d,\tSamples %d-%d/%d\n", prefix, current_epoch, nb_epochs, current_batch + 1, nb_batches, first_sample + 1, last_sample + 1, nb_inputs);

			// Prepare predicted outputs array for the current batch
			nn_type **predicted = mallocBlocking(nb_samples * sizeof(nn_type *), prefix);
			for (int i = 0; i < nb_samples; i++)
				predicted[i] = mallocBlocking(network->output_layer->nb_neurons * sizeof(nn_type), prefix);

			// Feed forward the current batches
			for (int i = 0; i < nb_samples; i++)
				NeuralNetworkFeedForwardCPUMultiThreads(network, inputs[first_sample + i], predicted[i]);

			// Reset deltas of all the layers (except the input layer)
			for (int i = 1; i < network->nb_layers; i++)
				memset(network->layers[i].deltas, 0, network->layers[i].nb_neurons * sizeof(nn_type));
			
			// Backpropagation
			NeuralNetworkStartBackPropagationCPUMultiThreads(network, predicted, expected + first_sample, nb_samples);
			NeuralNetworkFinishBackPropagationCPUMultiThreads(network);

			// Free the predicted outputs array for the current batch
			for (int i = 0; i < nb_samples; i++)
				free(predicted[i]);
			free(predicted);

			// Update the weights of the neural network
			NeuralNetworkUpdateWeightsCPUMultiThreads(network);
		}

		// Use the test inputs to calculate the current error
		if (nb_test_inputs > 0) {

			// Prepare predicted outputs array for the test inputs
			nn_type **predicted = mallocBlocking(nb_test_inputs * sizeof(nn_type *), prefix);
			for (int i = 0; i < nb_test_inputs; i++)
				predicted[i] = mallocBlocking(network->output_layer->nb_neurons * sizeof(nn_type), prefix);

			// Feed forward the test inputs
			for (int i = 0; i < nb_test_inputs; i++)
				NeuralNetworkFeedForwardCPUMultiThreads(network, test_inputs[i], predicted[i]);
			
			// Calculate the error of the test inputs using the loss function
			for (int i = 0; i < nb_test_inputs; i++)
				current_error += network->loss_function(predicted[i], excepted_tests[i], network->output_layer->nb_neurons);
			
			// Free the predicted outputs array for the test inputs
			for (int i = 0; i < nb_test_inputs; i++)
				free(predicted[i]);
			free(predicted);
		}

		// Verbose
		current_error /= nb_inputs;
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == nb_epochs || current_epoch % 10 == 0)) || verbose == 2)
			DEBUG_PRINT("%s: Epoch %d/%d, Error: %.12"NN_FORMAT"\n", prefix, current_epoch, nb_epochs, current_error);
	}

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("%s: Training done!\n", prefix);
	return current_epoch;
}


