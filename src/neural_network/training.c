
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
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {
				double input_value = network->layers[i - 1].activations_values[k];
				double weight = network->layers[i].weights[j][k];	// weights[j][k] where j = current neuron, k = previous neuron
				weighted_sum += input_value * weight;
			}

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
 * Basically, for each layer, for each neuron, the formula is:
 * - delta = error * derivative
 * - where error = sum( weight * delta of the next layer )
 * - where derivative = activation_value * (1 - activation_value)
 * 
 * @details i = Index of the selected layer (order last to first)
 * @details j = Index of the selected neuron
 * @details k = Index of the next layer selected neuron
 * 
 * @param network			Pointer to the neural network
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * 
 * @return void
 */
void NeuralNetworkDbackpropagation(NeuralNetworkD *network, double *excepted_output) {

	// For each neuron of the output layer,
	for (int neuron = 0; neuron < network->output_layer->nb_neurons; neuron++) {

		// Calculate the error of the neuron (excepted_output - activation_value)
		double error = excepted_output[neuron] - network->output_layer->activations_values[neuron];

		// Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
		double activation_value = network->output_layer->activations_values[neuron];
		double derivative = activation_value * (1 - activation_value);

		// Calculate the delta of the neuron (error * derivative)
		network->output_layer->deltas[neuron] = error * derivative;
	}

	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// For each neuron of the current layer,
		for (int j = 0; j < network->layers[i].nb_neurons; j++) {

			// Calculate the error of the neuron (sum( weight * delta of the next layer ))
			double error = 0;
			for (int k = 0; k < network->layers[i + 1].nb_neurons; k++) {

				// Get the weight of the next layer neuron linked to the current neuron
				// (Not [j][k] because it's reversed compared to the feed forward algorithm (checking next layer instead of previous layer))
				double weight = network->layers[i + 1].weights[k][j];

				// Get the delta of the next layer neuron
				double delta = network->layers[i + 1].deltas[k];

				// Add the weight * delta to the error
				error += weight * delta;
			}

			// Calculate the derivative of the activation function of the neuron (activation_value * (1 - activation_value))
			double activation_value = network->layers[i].activations_values[j];
			double derivative = activation_value * (1 - activation_value);

			// Calculate the delta of the neuron (error * derivative)
			network->layers[i].deltas[j] = error * derivative;
		}
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
 * 				- Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
 * 			- Update the bias (bias + (learning_rate * delta of the current neuron))
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - bias = bias + (learning_rate * delta of the current neuron)
 * - for each weight of the current neuron:
 *   - weight = weight + (learning_rate * delta of the current neuron * activation_value of the previous layer)
 * 
 * @details i = Index of the selected layer (except the input layer)
 * @details j = Index of the selected neuron
 * @details k = Index of the previous layer selected neuron
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

			// Variables for easier reading
			double learning_rate = network->learning_rate;
			double delta = network->layers[i].deltas[j];

			// For each weight of the current neuron,
			for (int k = 0; k < network->layers[i].nb_inputs_per_neuron; k++) {

				// Variable for easier reading
				double activation_value = network->layers[i - 1].activations_values[k];

				// Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
				network->layers[i].weights[j][k] += learning_rate * delta * activation_value;
			}

			// Update the bias (bias + (learning_rate * delta of the current neuron))
			network->layers[i].biases[j] += learning_rate * delta;
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
	NeuralNetworkDfeedForward(network, input);

	// Backpropagation
	NeuralNetworkDbackpropagation(network, excepted_output);

	// Update weights and biases
	NeuralNetworkDupdateWeightsAndBiases(network);
}


///// GPU Part /////
#include "../gpu/gpu_utils.h"

// Variables for gpu efficiency
cl_int gpu_code = 0;
cl_program gpu_program = NULL;
cl_kernel gpu_kernels[] = {NULL, NULL, NULL, NULL};
struct opencl_context_t gpu_oc;
int gpu_initialized = 0, gpu_current_kernel = -1;
void stopNeuralNetworkGpuOpenCL() {
	if (!gpu_initialized) return;
	clReleaseProgram(gpu_program);
	clReleaseKernels((int)(sizeof(gpu_kernels) / sizeof(cl_kernel)), gpu_kernels);
	clReleaseCommandQueue(gpu_oc.command_queue);
	clReleaseContext(gpu_oc.context);
	gpu_initialized = 0;
}

/**
 * @brief This function setup the gpu efficiency variables.
 * This function is automatically called by the NeuralNetworkDtrainGpuOpenCL() function.
 * 
 * @return int	0 if success, -1 otherwise
 */
int setupNeuralNetworkGpuOpenCL() {
	if (gpu_initialized) return 0;

	// Initialize OpenCL
	gpu_oc = setupOpenCL(CL_DEVICE_TYPE_GPU);

	// Create the program
	char path[] = "kernels/neural_network.cl";
	char* kernel_source = readKernelProgram(path);
	ERROR_HANDLE_PTR_RETURN_INT(kernel_source, "setupNeuralNetworkGpuOpenCL(): Cannot read kernel program '%s'\n", path);
	gpu_program = clCreateProgramWithSource(gpu_oc.context, 1, (const char**)&kernel_source, NULL, &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create program, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Build the program
	gpu_code = clBuildProgram(gpu_program, 1, &gpu_oc.device_id, NULL, NULL, NULL);
	if (gpu_code != CL_SUCCESS) {
		ERROR_PRINT("setupNeuralNetworkGpuOpenCL(): Cannot build program, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		printProgramBuildLog(gpu_program, gpu_oc.device_id, ERROR_LEVEL, "setupNeuralNetworkGpuOpenCL(): ");
		return gpu_code;
	}

	// Free the kernel source code
	free(kernel_source);

	// Create the kernels
	gpu_kernels[0] = clCreateKernel(gpu_program, "feedForwardActivationValues", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_kernels[1] = clCreateKernel(gpu_program, "backpropagationOutputLayerDeltas", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_kernels[2] = clCreateKernel(gpu_program, "backpropagationHiddenLayersDeltas", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_kernels[3] = clCreateKernel(gpu_program, "updateWeightsAndBiases", &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "setupNeuralNetworkGpuOpenCL(): Cannot create kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Register the exit function
	atexit(stopNeuralNetworkGpuOpenCL);

	// Set the initialized flag & return
	gpu_initialized = 1;
	return 0;
}


/**
 * @brief Feed forward the neural network with the input array
 * For a neural network using double as type and using GPU
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
int NeuralNetworkDfeedForwardGPU(NeuralNetworkD *network, double *input) {
	setupNeuralNetworkGpuOpenCL();

	// Set the input layer (copy the input array into the input layer of the neural network)
	memcpy(network->input_layer->activations_values, input, network->input_layer->nb_neurons * sizeof(double));

	// Create the buffers & copy all the data to the GPU (except the input layer): activations_values, weights, biases
	cl_mem *activation_values_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	cl_mem *weights_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	cl_mem *biases_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	for (int i = 0; i < network->nb_layers; i++) {
		activation_values_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot create buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		weights_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot create buffer 'weights_buffers[%d]' (size: %d = %d * %d), reason: %d / %s\n", i, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, gpu_code, getOpenCLErrorString(gpu_code));
		biases_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot create buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].activations_values, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot write buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, weights_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot write buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, biases_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].biases, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot write buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		int current_layer_size = network->layers[i].nb_neurons;
		int previous_layer_size = network->layers[i - 1].nb_neurons;
		gpu_code = clSetKernelArg(gpu_kernels[0], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 0 of kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 1, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 1 of kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 2, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 2 of kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 3, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 3 of kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 4, sizeof(int), &current_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 4 of kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[0], 5, sizeof(int), &previous_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot set kernel argument 5 of kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size = current_layer_size;
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[0], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot enqueue kernel 'feedForwardActivationValues', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Copy the data from the GPU to the CPU (except the input layer): activations_values
	for (int i = 1; i < network->nb_layers; i++) {
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, activation_values_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].activations_values, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDfeedForwardGPU(): Cannot read buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Finish & free the buffers
	for (int i = 0; i < network->nb_layers; i++) {
		clReleaseMemObject(activation_values_buffers[i]);
		if (i == 0) continue;
		clReleaseMemObject(weights_buffers[i]);
		clReleaseMemObject(biases_buffers[i]);
	}
	free(activation_values_buffers);
	free(weights_buffers);
	free(biases_buffers);

	// Return
	return 0;
}

/**
 * @brief Backpropagate the neural network with the excepted output array
 * For a neural network using double as type and using GPU
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
 * Basically, for each layer, for each neuron, the formula is:
 * - delta = error * derivative
 * - where error = sum( weight * delta of the next layer )
 * - where derivative = activation_value * (1 - activation_value)
 * 
 * @details i = Index of the selected layer (order last to first)
 * @details j = Index of the selected neuron
 * @details k = Index of the next layer selected neuron
 * 
 * @param network			Pointer to the neural network
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * 
 * @return void
 */
int NeuralNetworkDbackpropagationGPU(NeuralNetworkD *network, double *excepted_output) {
	setupNeuralNetworkGpuOpenCL();

	// Create the buffers & copy all the data to the GPU
	cl_mem excepted_output_buffer = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->output_layer->nb_neurons * sizeof(double), NULL, &gpu_code);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot create buffer 'excepted_output_buffer', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	cl_mem *activation_values_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	cl_mem *deltas_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	cl_mem *weights_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	for (int i = 0; i < network->nb_layers; i++) {
		activation_values_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot create buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		deltas_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot create buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		weights_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot create buffer 'weights_buffers[%d]' (size: %d = %d * %d), reason: %d / %s\n", i, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, gpu_code, getOpenCLErrorString(gpu_code));
	}
	gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, excepted_output_buffer, CL_TRUE, 0, network->output_layer->nb_neurons * sizeof(double), excepted_output, 0, NULL, NULL);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot write buffer 'excepted_output_buffer', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].activations_values, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot write buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, deltas_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].deltas, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot write buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, weights_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot write buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// For each neuron of the output layer, calculate the delta of the neuron (excepted_output - activation_value) * derivative
	int output_layer_size = network->output_layer->nb_neurons;
	gpu_code = clSetKernelArg(gpu_kernels[1], 0, sizeof(cl_mem), &excepted_output_buffer);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 0 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 1, sizeof(cl_mem), &activation_values_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 1 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 2, sizeof(cl_mem), &deltas_buffers[network->nb_layers - 1]);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 2 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	gpu_code = clSetKernelArg(gpu_kernels[1], 3, sizeof(int), &output_layer_size);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 3 of kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	size_t global_work_size[] = {output_layer_size};
	gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[1], 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot enqueue kernel 'backpropagationOutputLayerDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	// For each layer of the neural network (except the output layer and the input layer) (order last to first),
	for (int i = network->nb_layers - 2; i > 0; i--) {

		// For each neuron of the current layer,
		int current_layer_size = network->layers[i].nb_neurons;
		int next_layer_size = network->layers[i + 1].nb_neurons;
		gpu_code = clSetKernelArg(gpu_kernels[2], 0, sizeof(cl_mem), &weights_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 0 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 1, sizeof(cl_mem), &deltas_buffers[i + 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 1 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 2, sizeof(cl_mem), &activation_values_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 2 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 3, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 3 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 4, sizeof(int), &current_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 4 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[2], 5, sizeof(int), &next_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot set kernel argument 5 of kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size[] = {current_layer_size};
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[2], 1, NULL, global_work_size, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot enqueue kernel 'backpropagationHiddenLayersDeltas', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Copy the deltas back to the CPU
	for (int i = 0; i < network->nb_layers; i++) {
		if (i == 0) continue;
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, deltas_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].deltas, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot read buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Finish & free the buffers
	gpu_code = clFinish(gpu_oc.command_queue);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDbackpropagationGPU(): Cannot finish, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	clReleaseMemObject(excepted_output_buffer);
	for (int i = 0; i < network->nb_layers; i++) {
		clReleaseMemObject(activation_values_buffers[i]);
		if (i == 0) continue;
		clReleaseMemObject(deltas_buffers[i]);
		clReleaseMemObject(weights_buffers[i]);
	}
	free(activation_values_buffers);
	free(deltas_buffers);
	free(weights_buffers);

	// Return
	return 0;
}

/**
 * @brief Update/Nudge the weights and biases of the neural network
 * For a neural network using double as type and using GPU
 * 
 * Detailed description of the update weights and biases algorithm:
 * - For each layer of the neural network (except the input layer):
 * 		- For each neuron of the current layer:
 * 			- For each weight of the current neuron:
 * 				- Update the weight (weight + (learning_rate * delta of the current neuron * activation_value of the previous layer))
 * 			- Update the bias (bias + (learning_rate * delta of the current neuron))
 * 
 * Basically, for each layer, for each neuron, the formula is:
 * - bias = bias + (learning_rate * delta of the current neuron)
 * - for each weight of the current neuron:
 *   - weight = weight + (learning_rate * delta of the current neuron * activation_value of the previous layer)
 * 
 * @details i = Index of the selected layer (except the input layer)
 * @details j = Index of the selected neuron
 * @details k = Index of the previous layer selected neuron
 * 
 * @param network	Pointer to the neural network
 * 
 * @return void
 */
int NeuralNetworkDupdateWeightsAndBiasesGPU(NeuralNetworkD *network) {
	setupNeuralNetworkGpuOpenCL();

	// Create the buffers & copy all the data to the GPU
	cl_mem *activation_values_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	cl_mem *deltas_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	cl_mem *weights_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	cl_mem *biases_buffers = (cl_mem*)malloc(network->nb_layers * sizeof(cl_mem));
	for (int i = 0; i < network->nb_layers; i++) {
		activation_values_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot create buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		deltas_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_ONLY, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot create buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		weights_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot create buffer 'weights_buffers[%d]' (size: %d = %d * %d), reason: %d / %s\n", i, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, gpu_code, getOpenCLErrorString(gpu_code));
		biases_buffers[i] = clCreateBuffer(gpu_oc.context, CL_MEM_READ_WRITE, network->layers[i].nb_neurons * sizeof(double), NULL, &gpu_code);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot create buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}
	for (int i = 0; i < network->nb_layers; i++) {
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, activation_values_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].activations_values, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot write buffer 'activation_values_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		if (i == 0) continue;
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, deltas_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].deltas, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot write buffer 'deltas_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, weights_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot write buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueWriteBuffer(gpu_oc.command_queue, biases_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].biases, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot write buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// For each layer of the neural network (except the input layer),
	for (int i = 1; i < network->nb_layers; i++) {

		// For each neuron of the current layer,
		int current_layer_size = network->layers[i].nb_neurons;
		int previous_layer_size = network->layers[i - 1].nb_neurons;
		double learning_rate = network->learning_rate;
		gpu_code = clSetKernelArg(gpu_kernels[3], 0, sizeof(cl_mem), &activation_values_buffers[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 0 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 1, sizeof(cl_mem), &deltas_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 1 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 2, sizeof(cl_mem), &weights_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 2 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 3, sizeof(cl_mem), &biases_buffers[i]);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 3 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 4, sizeof(int), &current_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 4 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 5, sizeof(int), &previous_layer_size);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 5 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clSetKernelArg(gpu_kernels[3], 6, sizeof(double), &learning_rate);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot set kernel argument 6 of kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
		size_t global_work_size = current_layer_size;
		gpu_code = clEnqueueNDRangeKernel(gpu_oc.command_queue, gpu_kernels[3], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot enqueue kernel 'updateWeightsAndBiases', reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Copy the data from the GPU to the CPU (except the input layer): weights and biases
	for (int i = 1; i < network->nb_layers; i++) {
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, weights_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double), network->layers[i].weights_flat, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot read buffer 'weights_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
		gpu_code = clEnqueueReadBuffer(gpu_oc.command_queue, biases_buffers[i], CL_TRUE, 0, network->layers[i].nb_neurons * sizeof(double), network->layers[i].biases, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot read buffer 'biases_buffers[%d]', reason: %d / %s\n", i, gpu_code, getOpenCLErrorString(gpu_code));
	}

	// Finish & free the buffers
	gpu_code = clFinish(gpu_oc.command_queue);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDupdateWeightsAndBiasesGPU(): Cannot finish, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));
	for (int i = 0; i < network->nb_layers; i++) {
		clReleaseMemObject(activation_values_buffers[i]);
		if (i == 0) continue;
		clReleaseMemObject(deltas_buffers[i]);
		clReleaseMemObject(weights_buffers[i]);
		clReleaseMemObject(biases_buffers[i]);
	}
	free(activation_values_buffers);
	free(deltas_buffers);
	free(weights_buffers);
	free(biases_buffers);

	// Return
	return 0;
}

/**
 * @brief Train the neural network using the backpropagation algorithm
 * For a neural network using double as type and using GPU
 * 
 * @param network			Pointer to the neural network
 * @param input				Pointer to the input array (double), must be the same size as the input layer
 * @param excepted_output	Pointer to the excepted output array (double), must be the same size as the output layer
 * 
 * @return void
 */
int NeuralNetworkDtrainGPU(NeuralNetworkD *network, double *input, double *excepted_output) {

	// Feed forward
	gpu_code = NeuralNetworkDfeedForwardGPU(network, input);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot feed forward, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Backpropagation
	gpu_code = NeuralNetworkDbackpropagationGPU(network, excepted_output);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot backpropagate, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Update weights and biases
	gpu_code = NeuralNetworkDupdateWeightsAndBiasesGPU(network);
	ERROR_HANDLE_INT_RETURN_INT(gpu_code, "NeuralNetworkDtrainGPU(): Cannot update weights and biases, reason: %d / %s\n", gpu_code, getOpenCLErrorString(gpu_code));

	// Return
	return 0;
}

