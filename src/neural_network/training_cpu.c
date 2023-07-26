
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
 * 
 * @return void
 */
void NeuralNetworkFeedForwardCPUSingleThread(NeuralNetwork *network, nn_type *input) {

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
 * @brief Do all the steps of the neural network (Feed forward, backpropagation and update weights)
 * using a batch of inputs and a batch of target outputs.
 * 
 * @param network			Pointer to the neural network
 * @param inputs			Pointer to the inputs array (nn_type), must be the same size as the input layer
 * @param target_outputs	Pointer to the target outputs array (nn_type), must be the same size as the output layer
 * @param batch_size		Number of samples in the batch
 * 
 * @return void
 */
void NeuralNetworkAllInOneCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **target_outputs, int batch_size) {

	// TODO
}

/**
 * @brief Train the neural network with the CPU (Single core)
 * by using a batch of inputs and a batch of target outputs,
 * a number of epochs and a target error value
 * 
 * @param network					Pointer to the neural network
 * 
 * @param inputs					Pointer to the inputs array (nn_type), must be the same size as the input layer
 * @param target					Pointer to the target outputs array (nn_type), must be the same size as the output layer
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
int NeuralNetworkTrainCPUSingleThread(NeuralNetwork *network, nn_type **inputs, nn_type **target, int nb_inputs, int test_inputs_percentage, int batch_size, int nb_epochs, nn_type error_target, int verbose) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = nb_epochs != -1 || error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "NeuralNetworkTrainCPU(1 thread): At least the number of epochs or the error target must be specified!\n");

	// Prepare the test inputs
	int nb_test_inputs = nb_inputs * test_inputs_percentage / 100;
	nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &inputs[nb_inputs];
	nn_type **target_tests = &target[nb_inputs];
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
			
			// Do all the steps of the neural network (Feed forward, backpropagation and update weights)
			NeuralNetworkAllInOneCPUSingleThread(network, &inputs[first_sample], &target[first_sample], nb_samples);
		}

		// Use the test inputs to calculate the current error
		if (nb_test_inputs > 0) {

			// Prepare predicted outputs array for the test inputs
			nn_type **predicted = mallocBlocking(nb_test_inputs * sizeof(nn_type *), "NeuralNetworkTrainCPU(1 thread)");
			for (int i = 0; i < nb_test_inputs; i++)
				predicted[i] = mallocBlocking(network->output_layer->nb_neurons * sizeof(nn_type), "NeuralNetworkTrainCPU(1 thread)");

			// Feed forward the test inputs
			for (int i = 0; i < nb_test_inputs; i++) {
				NeuralNetworkFeedForwardCPUSingleThread(network, test_inputs[i]);
				memcpy(predicted[i], network->output_layer->activations_values, network->output_layer->nb_neurons * sizeof(nn_type));
			}
			
			// Calculate the error of the test inputs using the loss function
			for (int i = 0; i < nb_test_inputs; i++)
				current_error += network->loss_function(predicted[i], target_tests[i], network->output_layer->nb_neurons);
			
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
 * // TODO
 * @brief Objective create a multi-threaded version of the entire neural network training
 * Assuming the CPU has 12 threads
 * The multi-threaded algorithm is the following:
 * - Prepare 12 threads for the first layer
 * - Prepare 12 threads for the second layer waiting for sufficient data from the first layer
 * - ...
 * - Prepare 12 threads for the last layer waiting for sufficient data from the previous layer
 * 
 * - Prepare 12 threads for the backpropagation of the last layer
 * - Prepare 12 threads for the backpropagation of the previous layer waiting for sufficient data from the last layer
 * - ...
 * - Prepare 12 threads for the backpropagation of the first hidden layer waiting for sufficient data from the second hidden layer
 * 
 * - Prepare 12 threads for the update of the weights of the first hidden layer waiting for sufficient data from the backpropagation of the first hidden layer
**/

