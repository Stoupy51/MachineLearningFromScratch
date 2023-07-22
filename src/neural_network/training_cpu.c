
#include "training_cpu.h"
#include "../universal_utils.h"
#include "../st_benchmark.h"

/**
 * @brief Feed forward algorithm of the neural network
 * takes a batch of inputs and calculates the outputs of the neural network
 * 
 * @details batch = Index of the selected batch
 * @details i = Index of the selected layer
 * @details j = Index of the selected neuron
 * @details k = Index of the selected input
 * 
 * @param network		Pointer to the neural network
 * @param inputs		Pointer to the inputs array (nn_type), must be the same size as the input layer
 * @param outputs		Pointer to the outputs outputs array (nn_type), must be the same size as the output layer
 * @param batch_size	Number of samples in the batch
 * 
 * @return void
 */
void NeuralNetworkFeedForwardCPUSingleCore(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size) {

	// For each batch, calculate the outputs of the neural network
	for (int batch = 0; batch < batch_size; batch++) {

		// Copy the inputs to the input layer of the neural network
		size_t input_layer_size = network->input_layer->nb_neurons * sizeof(nn_type);
		memcpy(network->input_layer->activations_values, inputs[batch], input_layer_size);

		// For each layer of the neural network,
		for (int i = 0; i < network->nb_layers; i++) {

			// For each neuron of the layer,
			for (int j = 0; j < network->layers[i].nb_neurons; j++) {

				// Calculate the sum of the inputs multiplied by the weights
				nn_type input_sum = 0.0;
				for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
					input_sum += network->layers[i].weights[j][k] * network->layers[i].activations_values[k];

				// Add the bias to the sum
				input_sum += network->layers[i].biases[j];

				// Activate the neuron with the activation function
				network->layers[i].activations_values[j] = network->layers[i].activation_function(input_sum);
			}
		}

		// Copy the outputs of the output layer to the outputs array
		size_t output_layer_size = network->output_layer->nb_neurons * sizeof(nn_type);
		memcpy(outputs[batch], network->output_layer->activations_values, output_layer_size);
	}
}

/**
 * @brief Backpropagation algorithm of the neural network
 * takes a batch of expected outputs and adjusts the weights of the neural network
 * 
 * @details batch = Index of the selected batch
 * @details i = Index of the selected layer
 * @details j = Index of the selected neuron
 * @details k = Index of the selected input
 * 
 * @param network		Pointer to the neural network
 * @param predicted		Pointer to the predicted outputs array (nn_type), must be the same size as the output layer
 * @param expected		Pointer to the expected outputs array (nn_type), must be the same size as the output layer
 * @param batch_size	Number of samples in the batch
 * 
 * @return void
 */
void NeuralNetworkBackPropagationCPUSingleCore(NeuralNetwork *network, nn_type **predicted, nn_type **expected, int batch_size) {

	// For each batch, adjust the weights of the neural network
	for (int batch = 0; batch < batch_size; batch++) {

		// For each neuron of the output layer,
		for (int j = 0; j < network->output_layer->nb_neurons; j++) {

			// Calculate the error and the delta of the neuron
			nn_type error = expected[batch][j] - predicted[batch][j];
			nn_type delta = error * network->output_layer->activation_function_derivative(network->output_layer->activations_values[j]);
			network->output_layer->deltas[j] = delta;
		}

		// For each hidden layer of the neural network (from the last to the first),
		for (int i = network->nb_layers - 2; i >= 0; i--) {

			// For each neuron of the layer,
			for (int j = 0; j < network->layers[i].nb_neurons; j++) {

				// Calculate the error: sum of (the weights multiplied by the deltas) of the next layer
				nn_type error = 0.0;
				for (int k = 0; k < network->layers[i + 1].nb_neurons; k++)
					error += network->layers[i + 1].weights[k][j] * network->layers[i + 1].deltas[k];

				// Calculate the delta of the neuron
				nn_type delta = error * network->layers[i].activation_function_derivative(network->layers[i].activations_values[j]);
				network->layers[i].deltas[j] = delta;
			}
		}

		// For each layer of the neural network (except the input layer) (from the last to the first),
		for (int i = network->nb_layers - 1; i > 0; i--) {

			// For each neuron of the layer,
			for (int j = 0; j < network->layers[i].nb_neurons; j++) {

				// For each input of the neuron, adjust the weight of the neuron
				for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {
					nn_type adjustment = network->layers[i].deltas[j] * network->layers[i - 1].activations_values[k];
					network->layers[i].weights[j][k] += network->learning_rate * adjustment;
				}

				// Adjust the bias of the neuron
				network->layers[i].biases[j] += network->learning_rate * network->layers[i].deltas[j];
			}
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
 * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: verbose + benchmark)
 * 
 * @return int						Number of epochs done, -1 if there is an error
*/
int NeuralNetworkTrainCPUSingleCore(NeuralNetwork *network, nn_type **inputs, nn_type **expected, int nb_inputs, int test_inputs_percentage, int batch_size, int nb_epochs, nn_type error_target, int verbose) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = nb_epochs != -1 || error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "NeuralNetworkTrainCPUSingleCore(): At least the number of epochs or the error target must be specified!\n");

	// Prepare the test inputs
	int nb_test_inputs = nb_inputs * test_inputs_percentage / 100;
	nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &inputs[nb_inputs];
	nn_type **excepted_tests = &expected[nb_inputs];

	// Local variables
	int current_epoch = 0;
	nn_type current_error = 0.0;
	int nb_batches = nb_inputs / batch_size + (nb_inputs % batch_size != 0);

	// Training
	while (current_epoch < nb_epochs || current_error > error_target) {

		// Reset the current error and increment the current epoch
		current_error = 0.0;
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
			if (verbose > 0)
				DEBUG_PRINT("NeuralNetworkTrainCPUSingleCore(): Epoch %d/%d, Batch %d/%d, Samples %d-%d/%d\n", current_epoch + 1, nb_epochs, current_batch + 1, nb_batches, first_sample + 1, last_sample + 1, nb_inputs);
			
			// Prepare predicted outputs array for the current batch
			nn_type **predicted = mallocBlocking(nb_samples * sizeof(nn_type *), "NeuralNetworkTrainCPUSingleCore()");
			for (int i = 0; i < nb_samples; i++)
				predicted[i] = mallocBlocking(network->output_layer->nb_neurons * sizeof(nn_type), "NeuralNetworkTrainCPUSingleCore()");

			// Feed forward the current batch
			if (verbose > 0)
				PRINTER("- FeedForward in progress...");
			NeuralNetworkFeedForwardCPUSingleCore(network, inputs + first_sample, predicted, nb_samples);
			if (verbose > 0)
				PRINTER("\t\tDone\n");
			
			// Backpropagate the current batch
			if (verbose > 0)
				PRINTER("- BackPropagation in progress...");
			NeuralNetworkBackPropagationCPUSingleCore(network, predicted, expected + first_sample, nb_samples);
			if (verbose > 0)
				PRINTER("\tDone\n");
			
			// Free the predicted outputs array for the current batch
			for (int i = 0; i < nb_samples; i++)
				free(predicted[i]);
			free(predicted);
		}

		// Use the test inputs to calculate the current error
		if (nb_test_inputs > 0) {

			// Prepare predicted outputs array for the test inputs
			nn_type **predicted = mallocBlocking(nb_test_inputs * sizeof(nn_type *), "NeuralNetworkTrainCPUSingleCore()");
			for (int i = 0; i < nb_test_inputs; i++)
				predicted[i] = mallocBlocking(network->output_layer->nb_neurons * sizeof(nn_type), "NeuralNetworkTrainCPUSingleCore()");

			// Feed forward the test inputs
			if (verbose > 0)
				PRINTER("- FeedForward for test inputs in progress...");
			NeuralNetworkFeedForwardCPUSingleCore(network, test_inputs, predicted, nb_test_inputs);
			if (verbose > 0)
				PRINTER("\t\tDone\n");
			
			// Calculate the error of the test inputs using the loss function
			if (verbose > 0)
				PRINTER("- Error calculation in progress...");
			for (int i = 0; i < nb_test_inputs; i++)
				current_error += network->loss_function(predicted[i], excepted_tests[i], network->output_layer->nb_neurons);
			if (verbose > 0)
				PRINTER("\tDone\n");
			
			// Free the predicted outputs array for the test inputs
			for (int i = 0; i < nb_test_inputs; i++)
				free(predicted[i]);
			free(predicted);
		}

		// Verbose
		current_error /= nb_inputs;
		if (verbose > 0)
			DEBUG_PRINT("NeuralNetworkTrainCPUSingleCore(): Epoch %d/%d, Error: %f\n", current_epoch, nb_epochs, current_error);
	}

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("NeuralNetworkTrainCPUSingleCore(): Training done!\n");
	return current_epoch;
}

