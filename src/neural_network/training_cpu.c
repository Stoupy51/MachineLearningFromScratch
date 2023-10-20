
#include "training_cpu.h"
#include "loss_functions.h"
#include "../universal_utils.h"
#include "../st_benchmark.h"

#include <math.h>

#if NN_TYPE == 0
	#define nn_sqrt sqrtf
#elif NN_TYPE == 1
	#define nn_sqrt sqrt
#elif NN_TYPE == 2
	#define nn_sqrt sqrtl
#endif

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
 * @brief Feed forward algorithm of the neural network using a batch of inputs
 * 
 * @param network		Pointer to the neural network
 * @param inputs		Pointer to the inputs array
 * @param outputs		Pointer to the outputs array
 * @param batch_size	Number of samples in the batch
 */
void FeedForwardCPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size) {

	// Local variables
	size_t input_layer_size = network->input_layer->nb_neurons * sizeof(nn_type);
	size_t output_layer_size = network->output_layer->nb_neurons * sizeof(nn_type);

	// For each sample of the batch,
	for (int i = 0; i < batch_size; i++) {

		// Copy the inputs to the input layer of the neural network
		memcpy(network->input_layer->activations_values, inputs[i], input_layer_size);

		// For each layer of the neural network (except the input layer),
		for (int i = 1; i < network->nb_layers; i++) {

			// For each neuron of the layer,
			for (int j = 0; j < network->layers[i].nb_neurons; j++) {

				// Calculate the sum of the inputs multiplied by the weights
				nn_type input_sum = network->layers[i].biases[j];	// Add the bias to the sum
				for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
					input_sum += network->layers[i - 1].activations_values[k] * network->layers[i].weights[j][k];

				// Save the sum of the inputs multiplied by the weights
				network->layers[i].activations_values[j] = input_sum;
			}

			// Activate the layer with the activation function
			network->layers[i].activation_function(network->layers[i].activations_values, network->layers[i].nb_neurons);
		}

		// Copy the outputs of the neural network to the outputs array
		memcpy(outputs[i], network->output_layer->activations_values, output_layer_size);
	}
}

/**
 * @brief Feed forward algorithm of the neural network without inputs (assuming the inputs are already in the input layer)
 * 
 * @param network		Pointer to the neural network
 */
void FeedForwardCPUNoInput(NeuralNetwork *network) {

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the sum of the inputs multiplied by the weights
			nn_type input_sum = network->layers[i].biases[j];	// Add the bias to the sum
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
				input_sum += network->layers[i - 1].activations_values[k] * network->layers[i].weights[j][k];

			// Save the sum of the inputs multiplied by the weights
			network->layers[i].activations_values[j] = input_sum;
		}

		// Activate the layer with the activation function
		network->layers[i].activation_function(network->layers[i].activations_values, network->layers[i].nb_neurons);
	}
}






/**
 * @brief Train the neural network using the Stochastic Gradient Descent algorithm
 * 
 * @param network					Pointer to the neural network
 * @param training_data				Training data structure (inputs, target outputs, number of inputs, batch size, test inputs percentage)
 * @param training_parameters		Training parameters structure (number of epochs, error target, optimizer, loss function, learning rate)
 * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: all)
 * 
 * @return int						Number of epochs done, -1 if there is an error
 */
int TrainSGD(NeuralNetwork *network, TrainingData training_data, TrainingParameters training_parameters, int verbose) {

	// Prepare the test inputs (Taking the last inputs as test inputs depending on the percentage)
	int nb_test_inputs = training_data.nb_inputs * training_data.test_inputs_percentage / 100;
	training_data.nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &training_data.inputs[training_data.nb_inputs];
	nn_type **target_tests = &training_data.targets[training_data.nb_inputs];
	if (verbose > 0)
		INFO_PRINT("TrainSGD(): %d inputs, %d test inputs\n", training_data.nb_inputs, nb_test_inputs);

	// Local variables
	int current_epoch = 0;
	nn_type current_error = 100.0;
	int nb_batches = training_data.nb_inputs / training_data.batch_size + (training_data.nb_inputs % training_data.batch_size != 0);
	struct timeval epoch_start_time, epoch_end_time;
	memset(&epoch_start_time, 0, sizeof(struct timeval));

	// Prepare allocations for predictions (same size as the greatest batch size or the number of test inputs to avoid reallocations)
	nn_type **predictions;
	nn_type *predictions_flat_matrix = try2DFlatMatrixAllocation((void***)&predictions, nb_batches > nb_test_inputs ? nb_batches : nb_test_inputs, network->output_layer->nb_neurons, sizeof(nn_type), "TrainSGD(predictions)");

	// Initialize biases and weights gradients for hidden and output layers
	struct gradients_per_layer_t *gradients_per_layer = mallocBlocking(network->nb_layers * sizeof(struct gradients_per_layer_t), "TrainSGD(gradients_per_layer)");
	for (int i = 1; i < network->nb_layers; i++) {
		gradients_per_layer[i].biases_gradients = mallocBlocking(network->layers[i].nb_neurons * sizeof(nn_type), "TrainSGD(biases_gradients)");
		gradients_per_layer[i].weights_gradients_flat = try2DFlatMatrixAllocation((void***)&gradients_per_layer[i].weights_gradients, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, sizeof(nn_type), "TrainSGD(weights_gradients)");
	}

	// Get the loss function and its derivative
	nn_type (*loss_function)(nn_type, nn_type) = get_loss_function(training_parameters.loss_function_name);
	nn_type (*loss_function_derivative)(nn_type, nn_type) = get_loss_function_derivative(training_parameters.loss_function_name);

	// Verbose
	if (verbose > 0)
		INFO_PRINT("TrainSGD(): Starting training loop...\n");

	// Training loop until the number of epochs or the error target is reached
	while (current_epoch < training_parameters.nb_epochs && current_error > training_parameters.error_target) {

		// Reset the current error and increment the current epoch
		current_error = 0.0;
		current_epoch++;

		// Verbose, benchmark the training if needed
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == training_parameters.nb_epochs || current_epoch % 10 == 0)) || verbose > 1)
			st_gettimeofday(epoch_start_time, NULL);
		
		///// Epoch stuff
		// Shuffle the training data
		shuffleTrainingData(training_data.inputs, training_data.targets, training_data.nb_inputs);

		// For each batch of the training data,
		for (int batch = 0; batch < nb_batches; batch++) {

			// Calculate the index of the first and the last sample of the batch, and the number of samples in the batch
			int first_sample = batch * training_data.batch_size;
			int last_sample = first_sample + training_data.batch_size - 1;
			if (last_sample >= training_data.nb_inputs)
				last_sample = training_data.nb_inputs - 1;
			int nb_samples = last_sample - first_sample + 1;
			nn_type **targets = &training_data.targets[first_sample];

			// Feed forward algorithm from the first sample to the last sample
			FeedForwardCPU(network, &training_data.inputs[first_sample], predictions, nb_samples);

			///// Backpropagation stuff
			// Initialize the gradients of the weights and the biases to 0
			for (int i = 1; i < network->nb_layers; i++) {
				memset(gradients_per_layer[i].biases_gradients, 0, network->layers[i].nb_neurons * sizeof(nn_type));
				memset(gradients_per_layer[i].weights_gradients_flat, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(nn_type));
			}

			// Calculate the derivative of activation functions for the output layer
			network->output_layer->activation_function_derivative(network->output_layer->activations_values, network->output_layer->nb_neurons);

			// For each sample of the batch, for each neuron of the output layer,
			for (int sample = 0; sample < nb_samples; sample++) {
				for (int j = 0; j < network->output_layer->nb_neurons; j++) {

					// Calculate the gradient of the cost function with respect to the activation value of the neuron
					nn_type gradient = loss_function_derivative(predictions[sample][j], targets[sample][j])
						* network->output_layer->activations_values[j];

					// For each input of the neuron, calculate the weight gradient of the input
					for (int k = 0; k < network->output_layer->nb_inputs_per_neuron; k++)
						gradients_per_layer[network->nb_layers - 1].weights_gradients[j][k] += gradient * network->layers[network->nb_layers - 2].activations_values[k];

					// Calculate the gradient of the cost function with respect to the bias of the neuron
					gradients_per_layer[network->nb_layers - 1].biases_gradients[j] += gradient;
				}
			}

			// For each layer of the neural network (except the input layer) (in reverse order),
			for (int i = network->nb_layers - 2; i > 0; i--) {

				// Calculate the derivatives of activation functions
				network->layers[i].activation_function_derivative(network->layers[i].activations_values, network->layers[i].nb_neurons);

				// For each neuron of the layer,
				for (int j = 0; j < network->layers[i].nb_neurons; j++) {

					// Calculate the gradient by summing the gradients of the next layer multiplied by the weights
					nn_type gradient = 0.0;
					for (int k = 0; k < network->layers[i + 1].nb_neurons; k++)
						gradient += network->layers[i + 1].weights[k][j]
							* gradients_per_layer[i + 1].biases_gradients[k];

					// Multiply the gradient by the derivative of the activation function of the neuron
					gradient *= network->layers[i].activations_values[j];

					// For each input of the neuron, calculate the weight gradient of the input
					for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
						gradients_per_layer[i].weights_gradients[j][k] += gradient * network->layers[i - 1].activations_values[k];

					// Add the gradient of the bias of the neuron
					gradients_per_layer[i].biases_gradients[j] += gradient;
				}
			}

			// Update the weights and the biases
			for (int i = 1; i < network->nb_layers; i++) {
				for (int j = 0; j < network->layers[i].nb_neurons; j++) {
					for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
						network->layers[i].weights[j][k] -= (training_parameters.learning_rate * gradients_per_layer[i].weights_gradients[j][k]);
					network->layers[i].biases[j] -= (training_parameters.learning_rate * gradients_per_layer[i].biases_gradients[j]);
				}
			}
		}

		///// Use the test inputs to calculate the error
		// Feed forward algorithm from the first sample to the last sample
		FeedForwardCPU(network, test_inputs, predictions, nb_test_inputs);

		// Calculate the error
		for (int sample = 0; sample < nb_test_inputs; sample++)
			for (int j = 0; j < network->output_layer->nb_neurons; j++)
				current_error += loss_function(predictions[sample][j], target_tests[sample][j]) / network->output_layer->nb_neurons;
		current_error /= nb_test_inputs;

		// Verbose, benchmark the training if needed
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == training_parameters.nb_epochs || current_epoch % 10 == 0)) || verbose > 1) {
			
			// Calculate the time spent per step (time of the epoch divided by the number of inputs)
			st_gettimeofday(epoch_end_time, NULL);
			double time_per_step = (double)(epoch_end_time.tv_sec - epoch_start_time.tv_sec) * (1000000.0 / training_data.nb_inputs)
				+ (double)(epoch_end_time.tv_usec - epoch_start_time.tv_usec) / training_data.nb_inputs;

			PRINTER(
				YELLOW "[CPU TrainSGD] " RED "Epoch %3d /%d, %s: " YELLOW "%.8"NN_FORMAT RED " executed in " YELLOW "%.8f" RED "s (" YELLOW "%.2f" RED "us/input)\n" RESET,
				current_epoch,
				training_parameters.nb_epochs,
				training_parameters.loss_function_name,
				current_error,
				((double)(epoch_end_time.tv_sec - epoch_start_time.tv_sec) + (double)(epoch_end_time.tv_usec - epoch_start_time.tv_usec) / 1000000.0),
				time_per_step
			);
		}
	}

	// Free the predictions
	free2DFlatMatrix((void**)predictions, predictions_flat_matrix, nb_batches > nb_test_inputs ? nb_batches : nb_test_inputs);

	// Free the gradients
	for (int i = 1; i < network->nb_layers; i++) {
		free(gradients_per_layer[i].biases_gradients);
		free2DFlatMatrix((void**)gradients_per_layer[i].weights_gradients, gradients_per_layer[i].weights_gradients_flat, network->layers[i].nb_neurons);
	}
	free(gradients_per_layer);

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("TrainSGD(): Training done!\n");
	return current_epoch;
}


/**
 * @brief Train the neural network using the Adam algorithm (Adaptive Moment Estimation)
 * 
 * @param network					Pointer to the neural network
 * @param training_data				Training data structure (inputs, target outputs, number of inputs, batch size, test inputs percentage)
 * @param training_parameters		Training parameters structure (number of epochs, error target, optimizer, loss function, learning rate)
 * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: all)
 * 
 * @return int						Number of epochs done, -1 if there is an error
 */
int TrainAdam(NeuralNetwork *network, TrainingData training_data, TrainingParameters training_parameters, int verbose) {

	// Prepare the test inputs (Taking the last inputs as test inputs depending on the percentage)
	int nb_test_inputs = training_data.nb_inputs * training_data.test_inputs_percentage / 100;
	training_data.nb_inputs -= nb_test_inputs;
	nn_type **test_inputs = &training_data.inputs[training_data.nb_inputs];
	nn_type **target_tests = &training_data.targets[training_data.nb_inputs];
	if (verbose > 0)
		INFO_PRINT("TrainAdam(): %d inputs, %d test inputs\n", training_data.nb_inputs, nb_test_inputs);

	// Local variables
	int current_epoch = 0;
	nn_type current_error = 100.0;
	int nb_batches = training_data.nb_inputs / training_data.batch_size + (training_data.nb_inputs % training_data.batch_size != 0);
	struct timeval epoch_start_time, epoch_end_time;
	memset(&epoch_start_time, 0, sizeof(struct timeval));

	// Prepare allocations for predictions (same size as the greatest batch size or the number of test inputs to avoid reallocations)
	nn_type **predictions;
	nn_type *predictions_flat_matrix = try2DFlatMatrixAllocation((void***)&predictions, nb_batches > nb_test_inputs ? nb_batches : nb_test_inputs, network->output_layer->nb_neurons, sizeof(nn_type), "TrainSGD(predictions)");

	// Initialize some Adam optimizer parameters
	nn_type alpha = training_parameters.learning_rate;	// Learning rate
	nn_type beta1 = 0.9;								// 0.9 is recommended by the original paper
	nn_type beta2 = 0.999;								// 0.999 is recommended by the original paper
	nn_type epsilon = 1e-8;								// To avoid division by 0 in the bias-corrected moment estimates
	nn_type m = 0.0;									// First moment vector
	nn_type v = 0.0;									// Second moment vector
	nn_type m_hat;										// Bias-corrected first moment vector
	nn_type v_hat;										// Bias-corrected second moment vector
	nn_type minus_beta1 = 1.0 - beta1;					// 1 - beta1 (to avoid calculating it at each iteration)
	nn_type minus_beta2 = 1.0 - beta2;					// 1 - beta2 (to avoid calculating it at each iteration)

	// Initialize biases and weights gradients for hidden and output layers
	struct gradients_per_layer_t *gradients_per_layer = mallocBlocking(network->nb_layers * sizeof(struct gradients_per_layer_t), "TrainAdam(gradients_per_layer)");
	for (int i = 1; i < network->nb_layers; i++) {
		gradients_per_layer[i].biases_gradients = mallocBlocking(network->layers[i].nb_neurons * sizeof(nn_type), "TrainAdam(biases_gradients)");
		gradients_per_layer[i].weights_gradients_flat = try2DFlatMatrixAllocation((void***)&gradients_per_layer[i].weights_gradients, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, sizeof(nn_type), "TrainSGD(weights_gradients)");
	}

	// Get the loss function and its derivative
	nn_type (*loss_function)(nn_type, nn_type) = get_loss_function(training_parameters.loss_function_name);
	nn_type (*loss_function_derivative)(nn_type, nn_type) = get_loss_function_derivative(training_parameters.loss_function_name);

	// Verbose
	if (verbose > 0)
		INFO_PRINT("TrainAdam(): Starting training loop...\n");
	
	// Training loop until the number of epochs or the error target is reached
	while (current_epoch < training_parameters.nb_epochs && current_error > training_parameters.error_target) {

		// Reset the current error and increment the current epoch
		current_error = 0.0;
		current_epoch++;

		// Initialize more Adam optimizer parameters
		nn_type t = 1.0;						// Iteration counter
		nn_type beta1_t = beta1;				// beta1 to the power of t (to avoid calculating it at each iteration)
		nn_type beta2_t = beta2;				// beta2 to the power of t (to avoid calculating it at each iteration)
		m = 0.0;								// First moment vector
		v = 0.0;								// Second moment vector
		nn_type minus_beta1_t = 1.0 - beta1_t;	// 1 - (beta1 to the power of t (to avoid calculating it at each iteration))
		nn_type minus_beta2_t = 1.0 - beta2_t;	// 1 - (beta2 to the power of t (to avoid calculating it at each iteration))

		// Verbose, benchmark the training if needed
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == training_parameters.nb_epochs || current_epoch % 10 == 0)) || verbose > 1)
			st_gettimeofday(epoch_start_time, NULL);
		
		///// Epoch stuff
		// Shuffle the training data
		shuffleTrainingData(training_data.inputs, training_data.targets, training_data.nb_inputs);

		// For each batch of the training data,
		for (int batch = 0; batch < nb_batches; batch++) {

			// Calculate the index of the first and the last sample of the batch, and the number of samples in the batch
			int first_sample = batch * training_data.batch_size;
			int last_sample = first_sample + training_data.batch_size - 1;
			if (last_sample >= training_data.nb_inputs)
				last_sample = training_data.nb_inputs - 1;
			int nb_samples = last_sample - first_sample + 1;
			nn_type **targets = &training_data.targets[first_sample];

			// Feed forward algorithm from the first sample to the last sample
			FeedForwardCPU(network, &training_data.inputs[first_sample], predictions, nb_samples);

			///// Backpropagation stuff using the Adam optimizer (Adaptive Moment Estimation)
			// Initialize the gradients of the weights and the biases to 0
			for (int i = 1; i < network->nb_layers; i++) {
				memset(gradients_per_layer[i].biases_gradients, 0, network->layers[i].nb_neurons * sizeof(nn_type));
				memset(gradients_per_layer[i].weights_gradients_flat, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(nn_type));
			}

			// Calculate the derivative of activation functions for the output layer
			network->output_layer->activation_function_derivative(network->output_layer->activations_values, network->output_layer->nb_neurons);

			// For each sample of the batch, for each neuron of the output layer,
			for (int sample = 0; sample < nb_samples; sample++) {
				for (int j = 0; j < network->output_layer->nb_neurons; j++) {

					// Calculate the gradient of the cost function with respect to the activation value of the neuron
					nn_type gradient = loss_function_derivative(predictions[sample][j], targets[sample][j])
						* network->output_layer->activations_values[j];

					// For each input of the neuron, calculate the weight gradient of the input
					for (int k = 0; k < network->output_layer->nb_inputs_per_neuron; k++)
						gradients_per_layer[network->nb_layers - 1].weights_gradients[j][k] += gradient * network->layers[network->nb_layers - 2].activations_values[k];

					// Calculate the gradient of the cost function with respect to the bias of the neuron
					gradients_per_layer[network->nb_layers - 1].biases_gradients[j] += gradient;
				}
			}

			// For each layer of the neural network (except the input layer) (in reverse order),
			for (int i = network->nb_layers - 2; i > 0; i--) {

				// Calculate the derivatives of activation functions
				network->layers[i].activation_function_derivative(network->layers[i].activations_values, network->layers[i].nb_neurons);

				// For each neuron of the layer,
				for (int j = 0; j < network->layers[i].nb_neurons; j++) {

					// Calculate the gradient by summing the gradients of the next layer multiplied by the weights
					nn_type gradient = 0.0;
					for (int k = 0; k < network->layers[i + 1].nb_neurons; k++)
						gradient += network->layers[i + 1].weights[k][j]
							* gradients_per_layer[i + 1].biases_gradients[k];

					// Multiply the gradient by the derivative of the activation function of the neuron
					gradient *= network->layers[i].activations_values[j];

					// For each input of the neuron, calculate the weight gradient of the input
					for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++)
						gradients_per_layer[i].weights_gradients[j][k] += gradient * network->layers[i - 1].activations_values[k];

					// Add the gradient of the bias of the neuron
					gradients_per_layer[i].biases_gradients[j] += gradient;
				}
			}

			///// Update the weights and the biases along with the Adam optimizer parameters
			// For each layer of the neural network (except the input layer), for each weight
			for (int i = 1; i < network->nb_layers; i++) {
				for (int j = 0; j < network->layers[i].nb_neurons; j++) {
					for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {

						// Update the first moment vector (m) and the second moment vector (v)
						// m = beta1 * m + (1.0 - beta1) * gradients
						// v = beta2 * v + (1.0 - beta2) * gradients²
						m = beta1 * m + (minus_beta1) * gradients_per_layer[i].weights_gradients[j][k];
						v = beta2 * v + (minus_beta2) * gradients_per_layer[i].weights_gradients[j][k] * gradients_per_layer[i].weights_gradients[j][k];

						// Calculate the bias-corrected first moment vector (m_hat) and the bias-corrected second moment vector (v_hat)
						// m_hat = m / (1.0 - beta1_t)
						// v_hat = v / (1.0 - beta2_t)
						m_hat = m / (minus_beta1_t);
						v_hat = v / (minus_beta2_t);

						// Update the weight
						// weight -= (alpha * m_hat) / (nn_sqrt(v_hat) + epsilon)
						network->layers[i].weights[j][k] -= (alpha * m_hat) / (nn_sqrt(v_hat) + epsilon);
					}
				}
			}

			// Update the iteration counter (t), and the beta1 and beta2 powers (beta1_t and beta2_t)
			t++;
			beta1_t *= beta1;
			beta2_t *= beta2;
		}

		///// Use the test inputs to calculate the error
		// Feed forward algorithm from the first sample to the last sample
		FeedForwardCPU(network, test_inputs, predictions, nb_test_inputs);

		// Calculate the error
		for (int sample = 0; sample < nb_test_inputs; sample++)
			for (int j = 0; j < network->output_layer->nb_neurons; j++)
				current_error += loss_function(predictions[sample][j], target_tests[sample][j]) / network->output_layer->nb_neurons;
		current_error /= nb_test_inputs;

		// Verbose, benchmark the training if needed
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == training_parameters.nb_epochs || current_epoch % 10 == 0)) || verbose > 1) {
			
			// Calculate the time spent per step (time of the epoch divided by the number of inputs)
			st_gettimeofday(epoch_end_time, NULL);
			double time_per_step = (double)(epoch_end_time.tv_sec - epoch_start_time.tv_sec) * (1000000.0 / training_data.nb_inputs)
				+ (double)(epoch_end_time.tv_usec - epoch_start_time.tv_usec) / training_data.nb_inputs;

			PRINTER(
				YELLOW "[CPU TrainAdam] " RED "Epoch %3d/%d, %s: " YELLOW "%.8"NN_FORMAT RED " executed in " YELLOW "%.8f" RED "s (" YELLOW "%.2f" RED "us/input)\n" RESET,
				current_epoch,
				training_parameters.nb_epochs,
				training_parameters.loss_function_name,
				current_error,
				((double)(epoch_end_time.tv_sec - epoch_start_time.tv_sec) + (double)(epoch_end_time.tv_usec - epoch_start_time.tv_usec) / 1000000.0),
				time_per_step
			);
		}
	}

	// Free the predictions
	free2DFlatMatrix((void**)predictions, predictions_flat_matrix, nb_batches > nb_test_inputs ? nb_batches : nb_test_inputs);

	// Free the gradients
	for (int i = 1; i < network->nb_layers; i++) {
		free(gradients_per_layer[i].biases_gradients);
		free2DFlatMatrix((void**)gradients_per_layer[i].weights_gradients, gradients_per_layer[i].weights_gradients_flat, network->layers[i].nb_neurons);
	}
	free(gradients_per_layer);

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("TrainAdam(): Training done!\n");
	return current_epoch;
}

// /**
//  * @brief Train the neural network using the RMSProp algorithm (Root Mean Square Propagation)
//  * 
//  * @param network					Pointer to the neural network
//  * @param training_data				Training data structure (inputs, target outputs, number of inputs, batch size, test inputs percentage)
//  * @param training_parameters		Training parameters structure (number of epochs, error target, optimizer, loss function, learning rate)
//  * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: all)
//  * 
//  * @return int						Number of epochs done, -1 if there is an error
//  */
// int TrainRMSProp(NeuralNetwork *network, TrainingData training_data, TrainingParameters training_parameters, int verbose);


/**
 * @brief Train the neural network with the CPU (Single core)
 * by using a batch of inputs and a batch of target outputs,
 * a number of epochs and a target error value
 * 
 * @param network					Pointer to the neural network
 * @param training_data				Training data structure (inputs, target outputs, number of inputs, batch size, test inputs percentage)
 * @param training_parameters		Training parameters structure (number of epochs, error target, optimizer, loss function, learning rate)
 * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: all)
 * 
 * @return int						Number of epochs done, -1 if there is an error
 */
int TrainCPU(NeuralNetwork *network, TrainingData training_data, TrainingParameters training_parameters, int verbose) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = training_parameters.nb_epochs != -1 || training_parameters.error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "TrainCPU(): At least the number of epochs or the error target must be specified!\n");

	// Launch the training depending on the chosen optimizer
	if (strcmp(training_parameters.optimizer, "SGD") == 0 || strcmp(training_parameters.optimizer, "StochasticGradientDescent") == 0)
		return TrainSGD(network, training_data, training_parameters, verbose);

	else if (strcmp(training_parameters.optimizer, "Adam") == 0 || strcmp(training_parameters.optimizer, "ADAM") == 0)
		return TrainAdam(network, training_data, training_parameters, verbose);

	// else if (strcmp(training_parameters.optimizer, "RMSProp") == 0 || strcmp(training_parameters.optimizer, "RMS") == 0)
	// 	return TrainRMSProp(network, training_data, training_parameters, verbose);

	else {
		ERROR_PRINT("TrainCPU(): Unknown optimizer: '%s'\n", training_parameters.optimizer);
		return -1;
	}
}

