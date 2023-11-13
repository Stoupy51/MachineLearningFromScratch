
#include <math.h>

#include "training_gpu.h"
#include "training_utils.h"
#include "activation_functions.h"
#include "loss_functions.h"
#include "../universal_utils.h"
#include "../st_benchmark.h"

#include "gpu/program_strings.h"

#include "../gpu/gpu_utils.h"

/**
 * @brief Get the activation function string for the given name.
 * 
 * @param activation_function_name		Name of the activation function
 * 
 * @return The feed forward function string (must be freed)
 */
char* getActivationFunctionText(const char* activation_function_name) {

	// Copy the feed forward function
	char *copy = mallocBlocking(2048, "getActivationFunctionText(copy)");
	strcpy(copy, GC_FEED_FORWARD_FUNCTION);

	// Depending on the activation function, replace the activation function in the feed forward function
	if (strcmp(activation_function_name, "sigmoid") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION__", "1.0 / (1.0 + exp(-sum))");
	else if (strcmp(activation_function_name, "relu") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION__", "sum > 0.0 ? sum : 0.0");
	else if (strcmp(activation_function_name, "tanh") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION__", "tanh(sum)");
	else if (strcmp(activation_function_name, "identity") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION__", "sum");
	else if (strcmp(activation_function_name, "softmax") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION__", "exp(sum)");
	else {
		ERROR_PRINT("Activation function not found: '%s'\n", activation_function_name);
		exit(EXIT_FAILURE);
	}

	// Return the feed forward function
	return copy;
}

/**
 * @brief Get the activation function derivative string for the given name.
 * 
 * @param activation_function_name		Name of the activation function
 * 
 * @return The activation function derivative string (must be freed)
 */
char* getActivationFunctionDerivativeText(const char* activation_function_name) {

	// Copy the activation function derivative
	char *copy = mallocBlocking(2048, "getActivationFunctionDerivativeText(copy)");
	strcpy(copy, GC_ACTIVATION_DERIVATIVE);

	// Depending on the activation function, replace the activation function derivative in the string
	if (strcmp(activation_function_name, "sigmoid") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION_DERIVATIVE__", "activation_values[neuron] * (1.0 - activation_values[neuron])");
	else if (strcmp(activation_function_name, "relu") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION_DERIVATIVE__", "activation_values[neuron] > 0.0 ? 1.0 : 0.0");
	else if (strcmp(activation_function_name, "tanh") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION_DERIVATIVE__", "1.0 - activation_values[neuron] * activation_values[neuron]");
	else if (strcmp(activation_function_name, "identity") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION_DERIVATIVE__", "1.0");
	else if (strcmp(activation_function_name, "softmax") == 0)
		replaceString(copy, 1024, "__ACTIVATION_FUNCTION_DERIVATIVE__", "activation_values[neuron] * (1.0 - activation_values[neuron])");
	else {
		ERROR_PRINT("Activation function not found: '%s'\n", activation_function_name);
		exit(EXIT_FAILURE);
	}

	// Return the activation function derivative
	return copy;
}

/**
 * @brief Get the loss function string for the given name.
 * (The loss function is used to calculate the error)
 * 
 * @param loss_function_name		Name of the loss function
 * 
 * @return The loss function string (must be freed)
 */
char* getLossFunctionText(const char* loss_function_name) {
	
	// Copy the loss function
	char *copy = mallocBlocking(2048, "getLossFunctionText(copy)");
	strcpy(copy, GC_CALCULATE_ERROR);

	// Depending on the loss function, replace the loss function in the string
	if (strcmp(loss_function_name, "MAE") == 0 || strcmp(loss_function_name, "mean_absolute_error") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION__", NN_STRING" diff = target_tests[target_index + j] - predictions[target_index + j];error[sample] += (diff < 0 ? -diff : diff) / nb_neurons;");
	else if (strcmp(loss_function_name, "MSE") == 0 || strcmp(loss_function_name, "mean_squared_error") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION__", NN_STRING" diff = target_tests[target_index + j] - predictions[target_index + j];error[sample] += diff * diff / nb_neurons;");
	else if (strcmp(loss_function_name, "huber_loss") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION__", NN_STRING" diff = target_tests[target_index + j] - predictions[target_index + j];error[sample] += (diff < -1.0 ? 1.0 - 2.0 * diff : (diff < 1 ? diff * diff : 2.0 * diff - 1.0)) / nb_neurons;");
	else if (strcmp(loss_function_name, "binary_cross_entropy") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION__", NN_STRING" temp = predictions[target_index + j] < "NN_EPSILON_STR" ? "NN_EPSILON_STR" : (predictions[target_index + j] > 1 - "NN_EPSILON_STR" ? 1 - "NN_EPSILON_STR" : predictions[target_index + j]);error[sample] += -(target_tests[target_index + j] * log(temp) + (1 - target_tests[target_index + j]) * log(1 - temp)) / nb_neurons;");
	else if (strcmp(loss_function_name, "squared_hinge") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION__", NN_STRING" diff = 1 - target_tests[target_index + j] * predictions[target_index + j];error[sample] += (diff < 0 ? 0 : diff * diff) / nb_neurons;");
	else {
		ERROR_PRINT("Loss function not found: '%s'\n", loss_function_name);
		exit(EXIT_FAILURE);
	}

	// Return the loss function
	return copy;
}

/**
 * @brief Get the loss function derivative for the given loss function.
 * 
 * @param loss_function_name		Name of the loss function
 * 
 * @return The loss function derivative string (must be freed)
 */
char* getLossFunctionDerivativeText(const char* loss_function_name) {

	// Copy the loss function derivative
	char *copy = mallocBlocking(2048, "getLossFunctionDerivativeText(copy)");
	strcpy(copy, GC_OUTPUT_LOSS_AND_GRADIENT);

	// Depending on the loss function, replace the loss function derivative in the string
	if (strcmp(loss_function_name, "MAE") == 0 || strcmp(loss_function_name, "mean_absolute_error") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION_DERIVATIVE__", "(prediction[neuron] < target[neuron] ? -1.0 : 1.0)");
	else if (strcmp(loss_function_name, "MSE") == 0 || strcmp(loss_function_name, "mean_squared_error") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION_DERIVATIVE__", "2 * (prediction[neuron] - target[neuron])");
	else if (strcmp(loss_function_name, "huber_loss") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION_DERIVATIVE__", "(target[neuron] - prediction[neuron]);gradient = gradient < -1.0 ? -gradient : -1.0");
	else if (strcmp(loss_function_name, "binary_cross_entropy") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION_DERIVATIVE__", "prediction[neuron] == 0 ? "NN_EPSILON_STR" : (prediction[neuron] == 1 ? 1 - "NN_EPSILON_STR" : prediction[neuron]);gradient = (gradient - target[neuron]) / (gradient * (1 - gradient))");
	else if (strcmp(loss_function_name, "squared_hinge") == 0)
		replaceString(copy, 1024, "__LOSS_FUNCTION_DERIVATIVE__", "1 - target[neuron] * prediction[neuron];gradient = gradient < 0 ? 0 : -2 * gradient * target[neuron]");
	else {
		ERROR_PRINT("Loss function not found: '%s'\n", loss_function_name);
		exit(EXIT_FAILURE);
	}

	// Return the loss function derivative
	return copy;
}


int is_oc_setup = 0;
struct opencl_context_t oc;

/**
 * @brief Structure for each layer of the neural network during training (forwards and backwards).
 * 
 * @param activation_values				Activation values of the layer
 * @param weights						Weights of the layer
 * @param biases						Biases of the layer
 * @param biases_gradients				Biases gradients of the layer
 * @param weights_gradients_flat		Weights gradients of the layer (flattened)
 * 
 * @param program						Program of the layer
 * @param kernel						Kernel of the layer
 * @param optionnal_2nd_program			Optionnal second program of the layer (for softmax)
 * @param optionnal_2nd_kernel			Optionnal second kernel of the layer (for softmax)
 * 
 * @param act_derivative_program		Activation function derivative program
 * @param act_derivative_kernel			Activation function derivative kernel
 * @param hidden_layer_gradient_program	Hidden layer gradient program
 * @param hidden_layer_gradient_kernel	Hidden layer gradient kernel
 */
typedef struct LayerForGPU {
	cl_mem activation_values;
	cl_mem weights;
	cl_mem biases;
	cl_mem biases_gradients;
	cl_mem weights_gradients_flat;

	// Forward program and kernel
	cl_program program;
	cl_kernel kernel;
	cl_program optionnal_2nd_program;
	cl_kernel optionnal_2nd_kernel;

	// Activation function derivative program and kernel
	cl_program act_derivative_program;
	cl_kernel act_derivative_kernel;
	cl_program hidden_layer_gradient_program;
	cl_kernel hidden_layer_gradient_kernel;
} LayerForGPU;

/**
 * @brief Setup OpenCL for GPU.
 */
void setupOpenCLGPU() {
	if (is_oc_setup)
		return;
	oc = setupOpenCL(CL_DEVICE_TYPE_GPU);
	#if 0
		INFO_PRINT("setupOpenCLGPU(): OpenCL setup for GPU done\n");
		printDeviceInfo(oc.device_id);
	#endif
}

/**
 * @brief Feed forward the neural network on GPU with everything prepared.
 * 
 * @param network			Pointer to the neural network
 * @param inputs_buffers	Pointer to the inputs buffers array (cl_mem*)
 * @param outputs_buffers	Pointer to the outputs buffers array (cl_mem*)
 * @param batch_size		Number of inputs to process
 * @param events			Pointer to the events array (cl_event*)
 * @param events_count		Point to the number of events
 * @param layers			Pointer to the layers array (LayerForGPU*)
 * 
 * @return int				0 if everything went well, -1 if there is an error
 */
int FeedForwardGPUWithPreparedBuffers(NeuralNetwork *network, cl_mem *inputs_buffers, cl_mem *outputs_buffers, int batch_size, cl_event *events, int *events_count, LayerForGPU *layers) {

	// For each sample, launch the feed forward kernel
	for (int sample = 0; sample < batch_size; sample++) {

		// Set the arguments of the kernel (input and output buffers)
		// TODO fix : error for sample "185"
		cl_int error = clSetKernelArg(layers[1].kernel, 0, sizeof(cl_mem), &inputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 0 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		error = clSetKernelArg(layers[network->nb_layers - 1].kernel, 3, sizeof(cl_mem), &outputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 3 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));

		// If the final activation function is softmax, use the output buffer as input
		if (layers[network->nb_layers - 1].optionnal_2nd_kernel != NULL) {
			error = clSetKernelArg(layers[network->nb_layers - 1].optionnal_2nd_kernel, 0, sizeof(cl_mem), &outputs_buffers[sample]);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the optionnal_2nd_kernel argument 0 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		}

		// Launch the feed forward kernel for each layer
		for (int layer = 1; layer < network->nb_layers; layer++) {

			// Launch the feed forward kernel
			size_t global_dimensions[] = { network->layers[layer].nb_neurons, 0, 0 };
			if (*events_count == 0)
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[layer].kernel, 1, NULL, global_dimensions, NULL, 0, NULL, &events[0]);
			else
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[layer].kernel, 1, NULL, global_dimensions, NULL, 1, &events[*events_count - 1], &events[*events_count]);
			(*events_count)++;
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while launching the feed forward kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

			// If the final activation function is softmax, launch the softmax kernel
			if (layers[layer].optionnal_2nd_kernel != NULL) {
				global_dimensions[0] = 1;
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[layer].optionnal_2nd_kernel, 1, NULL, global_dimensions, NULL, 1, &events[*events_count - 1], &events[*events_count]);
				(*events_count)++;
				ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while launching the softmax kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			}
		}
	}

	// Return 0 if everything went well
	return 0;
}

/**
 * @brief Feed forward the neural network with the given inputs.
 * 
 * @param network		Pointer to the neural network
 * @param inputs		Pointer to the inputs array
 * @param outputs		Pointer to the outputs array
 * @param batch_size	Number of inputs to process
 */
int FeedForwardGPU(NeuralNetwork *network, nn_type **inputs, nn_type **outputs, int batch_size) {

	// Setup OpenCL
	setupOpenCLGPU();

	// Setup an array of cl_event to store all the preparation events
	cl_event *events = mallocBlocking((batch_size + network->nb_layers) * 2 * sizeof(cl_event), "FeedForwardGPU(events)");
	int events_count = 0;

	// Create the buffers
	cl_int error;
	cl_mem* inputs_buffers = mallocBlocking(batch_size * sizeof(cl_mem), "FeedForwardGPU(inputs_buffers)");
	cl_mem* outputs_buffers = mallocBlocking(batch_size * sizeof(cl_mem), "FeedForwardGPU(outputs_buffers)");
	for (int sample = 0; sample < batch_size; sample++) {
		inputs_buffers[sample] = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, network->input_layer->nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the input buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		outputs_buffers[sample] = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->output_layer->nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the output buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}

	// Send the inputs to the GPU
	for (int sample = 0; sample < batch_size; sample++) {
		error = clEnqueueWriteBuffer(oc.command_queue, inputs_buffers[sample], CL_FALSE, 0, network->input_layer->nb_neurons * sizeof(nn_type), inputs[sample], 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while writing the input buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}

	// For each layer of the neural network, create its structure
	LayerForGPU *layers = mallocBlocking(network->nb_layers * sizeof(LayerForGPU), "FeedForwardGPU(layers)");
	memset(layers, 0, network->nb_layers * sizeof(LayerForGPU));
	for (int layer = 1; layer < network->nb_layers; layer++) {

		// Prepare activations values buffer, weights and biases buffers and write them to the GPU
		layers[layer].activation_values = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the activation values buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		layers[layer].weights = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the weights buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		layers[layer].biases = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, network->layers[layer].nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the biases buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, layers[layer].weights, CL_FALSE, 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), network->layers[layer].weights_flat, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while writing the weights buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, layers[layer].biases, CL_FALSE, 0, network->layers[layer].nb_neurons * sizeof(nn_type), network->layers[layer].biases, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while writing the biases buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
	}

	// Details: the loop is cut in two parts because the buffers must be created before the kernels
	// Continue to prepare the layers
	for (int layer = 1; layer < network->nb_layers; layer++) {

		// Create the program and the kernel
		char *feed_forward_function_copy = getActivationFunctionText(network->layers[layer].activation_function_name);
		layers[layer].program = clCreateProgramWithSource(oc.context, 1, (const char**)&feed_forward_function_copy, NULL, &error);
		free(feed_forward_function_copy);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clBuildProgram(layers[layer].program, 1, &oc.device_id, NULL, NULL, NULL);
		if (error != CL_SUCCESS) {
			ERROR_PRINT("FeedForwardGPU(): Error while building the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			printProgramBuildLog(layers[layer].program, oc.device_id, ERROR_LEVEL, "FeedForwardGPU(): ");
			return error;
		}
		layers[layer].kernel = clCreateKernel(layers[layer].program, "feed_forward", &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

		// Set the arguments of the kernel
		if (layer > 1) {
			error = clSetKernelArg(layers[layer].kernel, 0, sizeof(cl_mem), &layers[layer - 1].activation_values);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 0 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
		error = clSetKernelArg(layers[layer].kernel, 1, sizeof(cl_mem), &layers[layer].weights);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 1 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clSetKernelArg(layers[layer].kernel, 2, sizeof(cl_mem), &layers[layer].biases);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 2 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		if (layer < network->nb_layers - 1) {
			error = clSetKernelArg(layers[layer].kernel, 3, sizeof(cl_mem), &layers[layer].activation_values);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 3 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
		error = clSetKernelArg(layers[layer].kernel, 4, sizeof(int), &network->layers[layer].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 4 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

		// If the activation function is softmax, create the program and the kernel for softmax
		if (strcmp(network->layers[layer].activation_function_name, "softmax") == 0) {
			DEBUG_PRINT("FeedForwardGPU(): Layer %d is softmax\n", layer);
			char *feed_forward_softmax = mallocBlocking(2048, "FeedForwardGPU(feed_forward_softmax)");
			strcpy(feed_forward_softmax, GC_FEED_FORWARD_SOFTMAX);
			layers[layer].optionnal_2nd_program = clCreateProgramWithSource(oc.context, 1, (const char**)&feed_forward_softmax, NULL, &error);
			free(feed_forward_softmax);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clBuildProgram(layers[layer].optionnal_2nd_program, 1, &oc.device_id, NULL, NULL, NULL);
			if (error != CL_SUCCESS) {
				ERROR_PRINT("FeedForwardGPU(): Error while building the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
				printProgramBuildLog(layers[layer].optionnal_2nd_program, oc.device_id, ERROR_LEVEL, "FeedForwardGPU(): ");
				return error;
			}
			layers[layer].optionnal_2nd_kernel = clCreateKernel(layers[layer].optionnal_2nd_program, "feed_forward_softmax", &error);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while creating the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

			// Set the arguments of the kernel
			if (layer < network->nb_layers - 1) {
				error = clSetKernelArg(layers[layer].optionnal_2nd_kernel, 0, sizeof(cl_mem), &layers[layer].activation_values);
				ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 0 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			}
			error = clSetKernelArg(layers[layer].optionnal_2nd_kernel, 1, sizeof(int), &network->layers[layer].nb_neurons);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 1 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
	}

	// Wait for the preparation events to finish and release them
	error = clFinish(oc.command_queue);
	ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while finishing the command queue with code %d / %s\n", error, getOpenCLErrorString(error));
	for (int i = 0; i < events_count; i++) {
		error = clReleaseEvent(events[i]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the preparation event %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
	}
	events_count = 0;

	// For each sample, launch the feed forward kernel
	error = FeedForwardGPUWithPreparedBuffers(network, inputs_buffers, outputs_buffers, batch_size, events, &events_count, layers);
	ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while launching the feed forward kernel with code %d / %s\n", error, getOpenCLErrorString(error));

	// Read the outputs from the GPU
	cl_event *last_forward = &events[events_count - 1];
	for (int sample = 0; sample < batch_size; sample++) {
		error = clEnqueueReadBuffer(oc.command_queue, outputs_buffers[sample], CL_FALSE, 0, network->output_layer->nb_neurons * sizeof(nn_type), outputs[sample], 1, last_forward, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while reading the output buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}

	// Finish command queue
	error = clFinish(oc.command_queue);
	ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while finishing the command queue with code %d / %s\n", error, getOpenCLErrorString(error));

	// Release the events
	for (int i = 0; i < events_count; i++) {
		error = clReleaseEvent(events[i]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the event %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
	}
	free(events);

	// Release the buffers
	for (int sample = 0; sample < batch_size; sample++) {
		error = clReleaseMemObject(inputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the input buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(outputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the output buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}
	free(inputs_buffers);
	free(outputs_buffers);

	// Release the layers
	for (int layer = 1; layer < network->nb_layers; layer++) {
		error = clReleaseMemObject(layers[layer].activation_values);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the activation values buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(layers[layer].weights);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the weights buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(layers[layer].biases);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the biases buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseProgram(layers[layer].program);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseKernel(layers[layer].kernel);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		if (layers[layer].optionnal_2nd_kernel != NULL) {
			error = clReleaseProgram(layers[layer].optionnal_2nd_program);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clReleaseKernel(layers[layer].optionnal_2nd_kernel);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
	}
	free(layers);

	// Return success
	return 0;
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
int TrainAdamGPU(NeuralNetwork *network, TrainingData training_data, TrainingParameters training_parameters, nn_type *error_per_epoch, int verbose) {

	// Prepare the test inputs (Taking the last inputs as test inputs depending on the percentage)
	int total_samples = training_data.nb_inputs;
	int nb_test_inputs = training_data.nb_inputs * training_data.test_inputs_percentage / 100;
	training_data.nb_inputs -= nb_test_inputs;
	if (verbose > 0)
		INFO_PRINT("TrainAdamGPU(): %d inputs, %d test inputs\n", training_data.nb_inputs, nb_test_inputs);

	// Set the batch size to the number of inputs if it is not set or if it's higher
	if (training_data.batch_size == -1 || training_data.batch_size > training_data.nb_inputs)
		training_data.batch_size = training_data.nb_inputs;

	// Local variables
	int current_epoch = 0;
	nn_type current_error = 100.0;
	int nb_batches =
		training_data.nb_inputs / training_data.batch_size				// integer division so some digits may be lost
		+ (training_data.nb_inputs % training_data.batch_size != 0);	// if there is a remainder, add 1 to the number of batches
	struct timeval epoch_start_time, epoch_end_time;
	memset(&epoch_start_time, 0, sizeof(struct timeval));

	// Setup OpenCL
	setupOpenCLGPU();

	// Setup an array of cl_event to store all the preparation events
	cl_event *events = mallocBlocking((2 * total_samples * network->nb_layers) * sizeof(cl_event), "TrainAdamGPU(events)");
	int events_count = 0;

	///// Copy everything to the GPU /////
	// Create the buffers for the inputs, the target outputs and the outputs
	cl_int error;
	cl_mem *inputs_buffers = mallocBlocking(total_samples * sizeof(cl_mem), "TrainAdamGPU(inputs_buffers)");
	cl_mem *target_outputs_buffers = mallocBlocking(total_samples * sizeof(cl_mem), "TrainAdamGPU(target_outputs_buffers)");
	cl_mem *outputs_buffers = mallocBlocking(total_samples * sizeof(cl_mem), "TrainAdamGPU(outputs_buffers)");
	for (int sample = 0; sample < total_samples; sample++) {
		inputs_buffers[sample] = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, network->input_layer->nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the input buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		target_outputs_buffers[sample] = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, network->output_layer->nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the target output buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		outputs_buffers[sample] = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->output_layer->nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the output buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}

	// Send the inputs and the target outputs to the GPU
	for (int sample = 0; sample < total_samples; sample++) {
		error = clEnqueueWriteBuffer(oc.command_queue, inputs_buffers[sample], CL_FALSE, 0, network->input_layer->nb_neurons * sizeof(nn_type), training_data.inputs[sample], 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the input buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, target_outputs_buffers[sample], CL_FALSE, 0, network->output_layer->nb_neurons * sizeof(nn_type), training_data.targets[sample], 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the target output buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}

	// Create a single cl_mem for error calculation
	cl_mem tests_target_buffer = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, network->output_layer->nb_neurons * nb_test_inputs * sizeof(nn_type), NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the tests target buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	cl_mem tests_outputs_buffer = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->output_layer->nb_neurons * nb_test_inputs * sizeof(nn_type), NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the tests outputs buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	for (int sample = 0; sample < nb_test_inputs; sample++) {
		error = clEnqueueWriteBuffer(oc.command_queue, tests_target_buffer, CL_FALSE, sample * network->output_layer->nb_neurons * sizeof(nn_type), network->output_layer->nb_neurons * sizeof(nn_type), training_data.targets[total_samples - nb_test_inputs + sample], 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the tests target buffer %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}

	// For each layer of the neural network, create its structure
	LayerForGPU *layers = mallocBlocking(network->nb_layers * sizeof(LayerForGPU), "TrainAdamGPU(layers)");
	memset(layers, 0, network->nb_layers * sizeof(LayerForGPU));
	for (int layer = 1; layer < network->nb_layers; layer++) {

		// Prepare activations values buffer, weights and biases buffers and write them to the GPU
		layers[layer].activation_values = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the activation values buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		layers[layer].weights = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the weights buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		layers[layer].biases = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the biases buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		layers[layer].biases_gradients = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the biases gradients buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		layers[layer].weights_gradients_flat = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the weights gradients buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, layers[layer].weights, CL_FALSE, 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), network->layers[layer].weights_flat, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the weights buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, layers[layer].biases, CL_FALSE, 0, network->layers[layer].nb_neurons * sizeof(nn_type), network->layers[layer].biases, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the biases buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
	}

	// Details: the loop is cut in 2 parts because the weights and biases buffers needs to be written before the kernel arguments are set
	// Continue the structure creation for each layer of the neural network
	for (int layer = 1; layer < network->nb_layers; layer++) {
		// Create the feed forward program and the kernel
		char *feed_forward_function_copy = getActivationFunctionText(network->layers[layer].activation_function_name);
		layers[layer].program = clCreateProgramWithSource(oc.context, 1, (const char**)&feed_forward_function_copy, NULL, &error);
		free(feed_forward_function_copy);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clBuildProgram(layers[layer].program, 1, &oc.device_id, NULL, NULL, NULL);
		if (error != CL_SUCCESS) {
			ERROR_PRINT("TrainAdamGPU(): Error while building the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			printProgramBuildLog(layers[layer].program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
			return error;
		}
		layers[layer].kernel = clCreateKernel(layers[layer].program, "feed_forward", &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

		// Set the arguments of the kernel
		if (layer > 0) {
			error = clSetKernelArg(layers[layer].kernel, 0, sizeof(cl_mem), &layers[layer - 1].activation_values);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
		error = clSetKernelArg(layers[layer].kernel, 1, sizeof(cl_mem), &layers[layer].weights);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clSetKernelArg(layers[layer].kernel, 2, sizeof(cl_mem), &layers[layer].biases);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 2 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		if (layer < network->nb_layers - 1) {
			error = clSetKernelArg(layers[layer].kernel, 3, sizeof(cl_mem), &layers[layer].activation_values);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
		error = clSetKernelArg(layers[layer].kernel, 4, sizeof(int), &network->layers[layer].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 4 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

		// If the activation function is softmax, create the program and the kernel for softmax
		if (strcmp(network->layers[layer].activation_function_name, "softmax") == 0) {
			char *feed_forward_softmax = mallocBlocking(2048, "TrainAdamGPU(feed_forward_softmax)");
			strcpy(feed_forward_softmax, GC_FEED_FORWARD_SOFTMAX);
			layers[layer].optionnal_2nd_program = clCreateProgramWithSource(oc.context, 1, (const char**)&feed_forward_softmax, NULL, &error);
			free(feed_forward_softmax);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clBuildProgram(layers[layer].optionnal_2nd_program, 1, &oc.device_id, NULL, NULL, NULL);
			if (error != CL_SUCCESS) {
				ERROR_PRINT("TrainAdamGPU(): Error while building the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
				printProgramBuildLog(layers[layer].optionnal_2nd_program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
				return error;
			}
			layers[layer].optionnal_2nd_kernel = clCreateKernel(layers[layer].optionnal_2nd_program, "feed_forward_softmax", &error);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

			// Set the arguments of the kernel
			if (layer < network->nb_layers - 1) {
				error = clSetKernelArg(layers[layer].optionnal_2nd_kernel, 0, sizeof(cl_mem), &layers[layer].activation_values);
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			}
			error = clSetKernelArg(layers[layer].optionnal_2nd_kernel, 1, sizeof(int), &network->layers[layer].nb_neurons);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}

		// Create the activation function derivative program and the kernel
		char *activation_derivative_copy = getActivationFunctionDerivativeText(network->layers[layer].activation_function_name);
		layers[layer].act_derivative_program = clCreateProgramWithSource(oc.context, 1, (const char**)&activation_derivative_copy, NULL, &error);
		free(activation_derivative_copy);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clBuildProgram(layers[layer].act_derivative_program, 1, &oc.device_id, NULL, NULL, NULL);
		if (error != CL_SUCCESS) {
			ERROR_PRINT("TrainAdamGPU(): Error while building the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			printProgramBuildLog(layers[layer].act_derivative_program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
			return error;
		}
		layers[layer].act_derivative_kernel = clCreateKernel(layers[layer].act_derivative_program, "activation_derivative", &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

		// Set the arguments of the kernel
		error = clSetKernelArg(layers[layer].act_derivative_kernel, 0, sizeof(cl_mem), &layers[layer].activation_values);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clSetKernelArg(layers[layer].act_derivative_kernel, 1, sizeof(int), &network->layers[layer].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

		// Create the hidden layer gradient program and the kernel
		if (layer < network->nb_layers - 1) {
			char *gc_hidden_layer_gradient = mallocBlocking(20480, "TrainAdamGPU(gc_hidden_layer_gradient)");
			strcpy(gc_hidden_layer_gradient, GC_HIDDEN_LAYER_GRADIENT);
			layers[layer].hidden_layer_gradient_program = clCreateProgramWithSource(oc.context, 1, (const char**)&gc_hidden_layer_gradient, NULL, &error);
			free(gc_hidden_layer_gradient);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clBuildProgram(layers[layer].hidden_layer_gradient_program, 1, &oc.device_id, NULL, NULL, NULL);
			if (error != CL_SUCCESS) {
				ERROR_PRINT("TrainAdamGPU(): Error while building the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
				printProgramBuildLog(layers[layer].hidden_layer_gradient_program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
				return error;
			}
			layers[layer].hidden_layer_gradient_kernel = clCreateKernel(layers[layer].hidden_layer_gradient_program, "hidden_layer_gradient", &error);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

			// Set the arguments of the kernel
			error = clSetKernelArg(layers[layer].hidden_layer_gradient_kernel, 0, sizeof(cl_mem), &layers[layer + 1].weights);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clSetKernelArg(layers[layer].hidden_layer_gradient_kernel, 1, sizeof(int), &network->layers[layer + 1].nb_neurons);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clSetKernelArg(layers[layer].hidden_layer_gradient_kernel, 2, sizeof(cl_mem), &layers[layer].activation_values);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 2 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clSetKernelArg(layers[layer].hidden_layer_gradient_kernel, 3, sizeof(cl_mem), &layers[layer - 1].activation_values);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clSetKernelArg(layers[layer].hidden_layer_gradient_kernel, 4, sizeof(cl_mem), &layers[layer].biases_gradients);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 4 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clSetKernelArg(layers[layer].hidden_layer_gradient_kernel, 5, sizeof(cl_mem), &layers[layer].weights_gradients_flat);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 5 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clSetKernelArg(layers[layer].hidden_layer_gradient_kernel, 6, sizeof(int), &network->layers[layer].nb_inputs_per_neuron);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 6 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
	}

	// Wait
	clFinish(oc.command_queue);
	for (int i = 0; i < events_count; i++)
		clReleaseEvent(events[i]);
	events_count = 0;

	// Create the program and the kernel for the loss function derivative
	char *loss_function = getLossFunctionDerivativeText(training_parameters.loss_function_name);
	cl_program loss_program = clCreateProgramWithSource(oc.context, 1, (const char**)&loss_function, NULL, &error);
	free(loss_function);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clBuildProgram(loss_program, 1, &oc.device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) {
		ERROR_PRINT("TrainAdamGPU(): Error while building the program for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
		printProgramBuildLog(loss_program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
		return error;
	}
	cl_kernel loss_kernel = clCreateKernel(loss_program, "output_loss_and_gradient", &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(loss_kernel, 2, sizeof(cl_mem), &layers[network->nb_layers - 1].activation_values);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 2 for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	if (network->nb_layers > 2) {
		error = clSetKernelArg(loss_kernel, 3, sizeof(cl_mem), &layers[network->nb_layers - 2].activation_values);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	}
	error = clSetKernelArg(loss_kernel, 4, sizeof(cl_mem), &layers[network->nb_layers - 1].biases_gradients);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 4 for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(loss_kernel, 5, sizeof(cl_mem), &layers[network->nb_layers - 1].weights_gradients_flat);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 5 for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(loss_kernel, 6, sizeof(int), &network->output_layer->nb_inputs_per_neuron);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 6 for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));

	///// Prepare the Adam algorithm /////
	// Create all the buffers for the Adam optimizer variables
	cl_mem beta1_t = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, sizeof(nn_type), NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the beta1_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	cl_mem beta2_t = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, sizeof(nn_type), NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the beta2_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	cl_mem minus_beta1_t = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, sizeof(nn_type), NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the minus_beta1_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	cl_mem minus_beta2_t = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, sizeof(nn_type), NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the minus_beta2_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));

	// Create the buffers for the first moment vector (m) and the second moment vector (v), and their bias-corrected versions (m_hat and v_hat)
	cl_mem *m = mallocBlocking((network->nb_layers - 1) * sizeof(cl_mem), "TrainAdamGPU(m)");
	cl_mem *v = mallocBlocking((network->nb_layers - 1) * sizeof(cl_mem), "TrainAdamGPU(v)");
	cl_mem *m_hat = mallocBlocking((network->nb_layers - 1) * sizeof(cl_mem), "TrainAdamGPU(m_hat)");
	cl_mem *v_hat = mallocBlocking((network->nb_layers - 1) * sizeof(cl_mem), "TrainAdamGPU(v_hat)");
	for (int layer = 1; layer < network->nb_layers; layer++) {
		m[layer - 1] = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the m buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		v[layer - 1] = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the v buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		m_hat[layer - 1] = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the m_hat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		v_hat[layer - 1] = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), NULL, &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the v_hat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
	}

	///// Create the program and the kernel for the weights and biases update
	// Get the text of the program and replace the placeholders with the values (constants: alpha, beta1, beta2, minus_beta1, minus_beta2, epsilon)
	char *gc_adam_update_weights_and_biases = mallocBlocking(2048, "TrainAdamGPU(gc_adam_update_weights_and_biases)");
	strcpy(gc_adam_update_weights_and_biases, GC_UPDATE_WEIGHTS_AND_BIASES_ADAM);
	char buffer[32] = "";
	sprintf(buffer, "%"NN_FORMAT, training_parameters.learning_rate);
	replaceString(gc_adam_update_weights_and_biases, 2048, "__LEARNING_RATE__", buffer);

	// Create the program and the kernel
	cl_program update_program = clCreateProgramWithSource(oc.context, 1, (const char**)&gc_adam_update_weights_and_biases, NULL, &error);
	free(gc_adam_update_weights_and_biases);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for the weights and biases update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clBuildProgram(update_program, 1, &oc.device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) {
		ERROR_PRINT("TrainAdamGPU(): Error while building the program for the weights and biases update with code %d / %s\n", error, getOpenCLErrorString(error));
		printProgramBuildLog(update_program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
		return error;
	}
	cl_kernel *update_kernels = mallocBlocking((network->nb_layers - 1) * sizeof(cl_kernel), "TrainAdamGPU(update_kernels)");
	for (int i = 1; i < network->nb_layers; i++) {
		update_kernels[i - 1] = clCreateKernel(update_program, "update_weights_and_biases", &error);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));

		// Set the arguments of the kernel
		error = clSetKernelArg(update_kernels[i - 1], 0, sizeof(cl_mem), &layers[i].weights);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 1, sizeof(cl_mem), &layers[i].biases);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 2, sizeof(cl_mem), &layers[i].weights_gradients_flat);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 2 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 3, sizeof(cl_mem), &layers[i].biases_gradients);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 4, sizeof(cl_mem), &m[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 4 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 5, sizeof(cl_mem), &v[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 5 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 6, sizeof(cl_mem), &m_hat[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 6 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 7, sizeof(cl_mem), &v_hat[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 7 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 8, sizeof(cl_mem), &minus_beta1_t);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 8 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 9, sizeof(cl_mem), &minus_beta2_t);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 9 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
		error = clSetKernelArg(update_kernels[i - 1], 10, sizeof(int), &network->layers[i].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 10 for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
	}

	// Create the program and the kernel for the Adam optimizer parameters update
	char *gc_adam_update_parameters = mallocBlocking(2048, "TrainAdamGPU(gc_adam_update_parameters)");
	strcpy(gc_adam_update_parameters, GC_UPDATE_PARAMETERS_ADAM);
	cl_program update_parameters_program = clCreateProgramWithSource(oc.context, 1, (const char**)&gc_adam_update_parameters, NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clBuildProgram(update_parameters_program, 1, &oc.device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) {
		ERROR_PRINT("TrainAdamGPU(): Error while building the program for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
		printProgramBuildLog(update_parameters_program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
		return error;
	}
	cl_kernel update_parameters_kernel = clCreateKernel(update_parameters_program, "update_parameters_adam", &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(update_parameters_kernel, 0, sizeof(cl_mem), &beta1_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(update_parameters_kernel, 1, sizeof(cl_mem), &beta2_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(update_parameters_kernel, 2, sizeof(cl_mem), &minus_beta1_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 2 for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(update_parameters_kernel, 3, sizeof(cl_mem), &minus_beta2_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));

	// Create the program and the kernel for the error calculation
	char *gc_error = getLossFunctionText(training_parameters.loss_function_name);
	cl_program error_program = clCreateProgramWithSource(oc.context, 1, (const char**)&gc_error, NULL, &error);
	free(gc_error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the program for the error calculation with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clBuildProgram(error_program, 1, &oc.device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS) {
		ERROR_PRINT("TrainAdamGPU(): Error while building the program for the error calculation with code %d / %s\n", error, getOpenCLErrorString(error));
		printProgramBuildLog(error_program, oc.device_id, ERROR_LEVEL, "TrainAdamGPU(): ");
		return error;
	}
	cl_kernel error_kernel = clCreateKernel(error_program, "calculate_error", &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the kernel for the error calculation with code %d / %s\n", error, getOpenCLErrorString(error));
	cl_mem errors_buffer = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, sizeof(nn_type) * nb_test_inputs, NULL, &error);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while creating the errors buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(error_kernel, 2, sizeof(cl_mem), &errors_buffer);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 2 for the error calculation with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clSetKernelArg(error_kernel, 3, sizeof(int), &network->output_layer->nb_neurons);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for the error calculation with code %d / %s\n", error, getOpenCLErrorString(error));
	nn_type *errors_to_sum = mallocBlocking(sizeof(nn_type) * nb_test_inputs, "TrainAdamGPU(errors_to_sum)");
	memset(errors_to_sum, 0, sizeof(nn_type) * nb_test_inputs);

	// Wait for everything to be finished and release the events
	clFinish(oc.command_queue);
	for (int i = 0; i < events_count; i++)
		clReleaseEvent(events[i]);
	events_count = 0;

	///// Launch the training /////
	// Verbose
	if (verbose > 0)
		INFO_PRINT("TrainAdamGPU(): Starting training loop...\n");

	// Training loop until the number of epochs or the error target is reached
	nn_type beta1 = 0.9;
	nn_type beta2 = 0.999;
	nn_type minus_beta1 = 1.0 - beta1;
	nn_type minus_beta2 = 1.0 - beta2;
	nn_type zero = 0.0;
	while (current_epoch < training_parameters.nb_epochs && current_error > training_parameters.error_target) {

		// Reset the current error and increment the current epoch
		current_error = 0.0;
		current_epoch++;

		// Initialize adam parameters
		error = clEnqueueWriteBuffer(oc.command_queue, beta1_t, CL_FALSE, 0, sizeof(nn_type), &beta1, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the beta1_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, beta2_t, CL_FALSE, 0, sizeof(nn_type), &beta2, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the beta2_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, minus_beta1_t, CL_FALSE, 0, sizeof(nn_type), &minus_beta1, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the minus_beta1_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
		error = clEnqueueWriteBuffer(oc.command_queue, minus_beta2_t, CL_FALSE, 0, sizeof(nn_type), &minus_beta2, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while writing the minus_beta2_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
		error = clEnqueueFillBuffer(oc.command_queue, errors_buffer, &zero, sizeof(nn_type), 0, sizeof(nn_type) * nb_test_inputs, 0, NULL, &events[events_count++]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while filling the errors buffer with code %d / %s\n", error, getOpenCLErrorString(error));
		for (int layer = 1; layer < network->nb_layers; layer++) {
			error = clEnqueueFillBuffer(oc.command_queue, m[layer - 1], &zero, sizeof(nn_type), 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), 0, NULL, &events[events_count++]);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while filling the m buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clEnqueueFillBuffer(oc.command_queue, v[layer - 1], &zero, sizeof(nn_type), 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), 0, NULL, &events[events_count++]);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while filling the v buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clEnqueueFillBuffer(oc.command_queue, m_hat[layer - 1], &zero, sizeof(nn_type), 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), 0, NULL, &events[events_count++]);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while filling the m_hat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clEnqueueFillBuffer(oc.command_queue, v_hat[layer - 1], &zero, sizeof(nn_type), 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), 0, NULL, &events[events_count++]);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while filling the v_hat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}

		// Verbose, benchmark the training if needed
		if ((verbose > 0 && (current_epoch < 6 || current_epoch == training_parameters.nb_epochs || current_epoch % 10 == 0)) || verbose > 1)
			st_gettimeofday(epoch_start_time, NULL);
		
		///// Epoch stuff
		// Shuffle the training data
		shuffleTrainingData(inputs_buffers, target_outputs_buffers, training_data.nb_inputs, cl_mem);

		// Wait for everything to be finished and release the events
		clFinish(oc.command_queue);
		for (int i = 0; i < events_count; i++)
			clReleaseEvent(events[i]);
		events_count = 0;

		// For each batch of the training data,
		for (int batch = 0; batch < nb_batches; batch++) {

			// Calculate the index of the first and the last sample of the batch, and the number of samples in the batch
			int first_sample = batch * training_data.batch_size;
			int last_sample = first_sample + training_data.batch_size - 1;
			if (last_sample >= training_data.nb_inputs)
				last_sample = training_data.nb_inputs - 1;
			int nb_samples = last_sample - first_sample + 1;

			///// Feed forward
			error = FeedForwardGPUWithPreparedBuffers(network, &inputs_buffers[first_sample], &outputs_buffers[first_sample], nb_samples, events, &events_count, layers);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while feeding forward with code %d / %s\n", error, getOpenCLErrorString(error));
			int last_feed_forward_event = events_count - 1;

			///// Backpropagation stuff using the Adam optimizer (Adaptive Moment Estimation)
			// Initialize the gradients of the weights and the biases to 0
			for (int layer = 1; layer < network->nb_layers; layer++) {
				error = clEnqueueFillBuffer(oc.command_queue, layers[layer].weights_gradients_flat, &zero, sizeof(nn_type), 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), 0, NULL, &events[events_count++]);
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while filling the weights_gradients_flat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
				error = clEnqueueFillBuffer(oc.command_queue, layers[layer].biases_gradients, &zero, sizeof(nn_type), 0, network->layers[layer].nb_neurons * sizeof(nn_type), 0, NULL, &events[events_count++]);
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while filling the biases_gradients buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			}

			// Copy the first activation values resulting of the feed forward to the actual activation values buffers
			error = clEnqueueCopyBuffer(oc.command_queue, outputs_buffers[first_sample], layers[network->nb_layers - 1].activation_values, 0, 0, network->output_layer->nb_neurons * sizeof(nn_type), 1, &events[last_feed_forward_event], &events[events_count++]);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while copying the activation values of the output layer with code %d / %s\n", error, getOpenCLErrorString(error));

			// Calculate the derivative of activation functions for the output layer
			size_t deri_dimensions[] = { network->output_layer->nb_neurons, 0, 0 };
			error = clEnqueueNDRangeKernel(oc.command_queue, layers[network->nb_layers - 1].act_derivative_kernel, 1, NULL, deri_dimensions, NULL, 1, &events[events_count - 1], &events[events_count]);
			events_count++;
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while launching the activation derivative kernel for the output layer with code %d / %s\n", error, getOpenCLErrorString(error));

			// Calculate the gradient of the loss function for the output layer
			for (int sample = 0; sample < nb_samples; sample++) {

				// Set the arguments of the kernel (prediction and target output buffers)
				error = clSetKernelArg(loss_kernel, 0, sizeof(cl_mem), &outputs_buffers[first_sample + sample]);
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
				error = clSetKernelArg(loss_kernel, 1, sizeof(cl_mem), &target_outputs_buffers[first_sample + sample]);
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));

				// Set possible missing argument of the kernel
				if (network->nb_layers == 2) {
					error = clSetKernelArg(loss_kernel, 3, sizeof(cl_mem), &inputs_buffers[first_sample + sample]);
					ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
				}

				// Launch the kernel
				error = clEnqueueNDRangeKernel(oc.command_queue, loss_kernel, 1, NULL, deri_dimensions, NULL, 1, &events[events_count - 1], &events[events_count]);
				events_count++;
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while launching the loss function derivative kernel for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
			}

			// Set possible missing argument of the kernel
			if (network->nb_layers > 2) {
					error = clSetKernelArg(layers[1].hidden_layer_gradient_kernel, 3, sizeof(cl_mem), &inputs_buffers[first_sample]);
					ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 3 for layer 1 with code %d / %s\n", error, getOpenCLErrorString(error));
				}

			// For each layer of the neural network (except the input layer) (in reverse order),
			for (int i = network->nb_layers - 2; i > 0; i--) {

				// Calculate the derivative of activation functions
				size_t global_dimensions[] = { network->layers[i].nb_neurons, 0, 0 };
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[i].act_derivative_kernel, 1, NULL, global_dimensions, NULL, 1, &events[events_count - 1], &events[events_count]);
				events_count++;
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while launching the activation derivative kernel for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));

				// Calculate the gradient
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[i].hidden_layer_gradient_kernel, 1, NULL, global_dimensions, NULL, 1, &events[events_count - 1], &events[events_count]);
				events_count++;
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while launching the hidden layer gradient kernel for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
			}

			// Update the weights and the biases
			int backpropagation_last_event = events_count - 1;
			for (int layer = 1; layer < network->nb_layers; layer++) {

				// Launch the kernel
				size_t global_dimensions[] = { network->layers[layer].nb_neurons, 0, 0 };
				error = clEnqueueNDRangeKernel(oc.command_queue, update_kernels[layer - 1], 1, NULL, global_dimensions, NULL, 1, &events[backpropagation_last_event], &events[events_count]);
				events_count++;
				ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while launching the weights and biases update kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			}

			// Wait for everything to be finished and release the events
			clFinish(oc.command_queue);
			for (int i = 0; i < events_count; i++)
				clReleaseEvent(events[i]);
			events_count = 0;

			// Update the Adam optimizer parameters
			size_t update_dimensions[] = { 1, 0, 0 };
			error = clEnqueueNDRangeKernel(oc.command_queue, update_parameters_kernel, 1, NULL, update_dimensions, NULL, 0, NULL, &events[events_count++]);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while launching the Adam optimizer parameters update kernel with code %d / %s\n", error, getOpenCLErrorString(error));
		}

		// Wait for everything to be finished and release the events
		clFinish(oc.command_queue);
		for (int i = 0; i < events_count; i++)
			clReleaseEvent(events[i]);
		events_count = 0;

		///// Use the test inputs to calculate the error
		// Feed forward
		error = FeedForwardGPUWithPreparedBuffers(network, &inputs_buffers[training_data.nb_inputs], &outputs_buffers[training_data.nb_inputs], nb_test_inputs, events, &events_count, layers);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while feeding forward with code %d / %s\n", error, getOpenCLErrorString(error));

		///// Calculate the error
		// Copy the outputs from outputs_buffers to the single output buffer
		for (int sample = 0; sample < nb_test_inputs; sample++) {
			error = clEnqueueCopyBuffer(oc.command_queue, outputs_buffers[training_data.nb_inputs + sample], tests_outputs_buffer, 0, sample * network->output_layer->nb_neurons * sizeof(nn_type), network->output_layer->nb_neurons * sizeof(nn_type), 0, NULL, &events[events_count++]);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while copying the outputs from the outputs buffer for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		}

		// Set the arguments of the kernel (prediction and target output buffers)
		error = clSetKernelArg(error_kernel, 0, sizeof(cl_mem), &tests_outputs_buffer);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 0 with code %d / %s\n", error, getOpenCLErrorString(error));
		error = clSetKernelArg(error_kernel, 1, sizeof(cl_mem), &tests_target_buffer);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while setting the kernel argument 1 with code %d / %s\n", error, getOpenCLErrorString(error));

		// Launch the kernel
		size_t output_dimensions[] = { nb_test_inputs, 0, 0 };
		error = clEnqueueNDRangeKernel(oc.command_queue, error_kernel, 1, NULL, output_dimensions, NULL, 1, &events[events_count - 1], &events[events_count]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while launching the error kernel with code %d / %s\n", error, getOpenCLErrorString(error));

		// Wait for everything to be finished and release the events
		clFinish(oc.command_queue);
		for (int i = 0; i < events_count; i++)
			clReleaseEvent(events[i]);
		events_count = 0;

		// Copy the errors from the device to the host to sum them up
		error = clEnqueueReadBuffer(oc.command_queue, errors_buffer, CL_TRUE, 0, sizeof(nn_type) * nb_test_inputs, errors_to_sum, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while reading the errors buffer with code %d / %s\n", error, getOpenCLErrorString(error));
		for (int i = 0; i < nb_test_inputs; i++)
			current_error += errors_to_sum[i];
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

		// Save the error of the epoch
		if (error_per_epoch != NULL)
			error_per_epoch[current_epoch - 1] = current_error;
	}

	// Copy the weights and the biases from the device to the host
	for (int layer = 1; layer < network->nb_layers; layer++) {
		error = clEnqueueReadBuffer(oc.command_queue, layers[layer].weights, CL_TRUE, 0, network->layers[layer].nb_neurons * network->layers[layer].nb_inputs_per_neuron * sizeof(nn_type), network->layers[layer].weights_flat, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while reading the weights buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clEnqueueReadBuffer(oc.command_queue, layers[layer].biases, CL_TRUE, 0, network->layers[layer].nb_neurons * sizeof(nn_type), network->layers[layer].biases, 0, NULL, NULL);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while reading the biases buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
	}

	// Free everything
	free(events);
	for (int sample = 0; sample < total_samples; sample++) {
		error = clReleaseMemObject(inputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the inputs buffer for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(target_outputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the target outputs buffer for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(outputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the outputs buffer for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
	}
	free(inputs_buffers);
	free(target_outputs_buffers);
	free(outputs_buffers);
	error = clReleaseMemObject(tests_target_buffer);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the tests target buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseMemObject(tests_outputs_buffer);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the tests outputs buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	for (int layer = 1; layer < network->nb_layers; layer++) {
		error = clReleaseMemObject(layers[layer].activation_values);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the activation values buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(layers[layer].weights);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the weights buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(layers[layer].biases);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the biases buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(layers[layer].biases_gradients);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the biases_gradients buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(layers[layer].weights_gradients_flat);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the weights_gradients_flat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseProgram(layers[layer].program);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseKernel(layers[layer].kernel);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		if (layers[layer].optionnal_2nd_program != NULL) {
			error = clReleaseProgram(layers[layer].optionnal_2nd_program);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the optionnal_2nd_program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clReleaseKernel(layers[layer].optionnal_2nd_kernel);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the optionnal_2nd_kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
		if (layer > 0) {
			error = clReleaseProgram(layers[layer].act_derivative_program);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the act_derivative_program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clReleaseKernel(layers[layer].act_derivative_kernel);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the act_derivative_kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
		if (layer > 0 && layer < network->nb_layers - 1) {
			error = clReleaseProgram(layers[layer].hidden_layer_gradient_program);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the hidden_layer_gradient_program for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clReleaseKernel(layers[layer].hidden_layer_gradient_kernel);
			ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the hidden_layer_gradient_kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}
	}
	free(layers);
	error = clReleaseProgram(loss_program);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the program for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseKernel(loss_kernel);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the kernel for the loss function with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseMemObject(beta1_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the beta1_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseMemObject(beta2_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the beta2_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseMemObject(minus_beta1_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the minus_beta1_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseMemObject(minus_beta2_t);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the minus_beta2_t buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	for (int layer = 1; layer < network->nb_layers; layer++) {
		error = clReleaseMemObject(m[layer - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the m buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(v[layer - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the v buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(m_hat[layer - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the m_hat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clReleaseMemObject(v_hat[layer - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the v_hat buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
	}
	free(m);
	free(v);
	free(m_hat);
	free(v_hat);
	error = clReleaseProgram(update_program);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the program for the weights and biases update with code %d / %s\n", error, getOpenCLErrorString(error));
	for (int i = 1; i < network->nb_layers; i++) {
		error = clReleaseKernel(update_kernels[i - 1]);
		ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the kernel for the weights and biases update for layer %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
	}
	free(update_kernels);
	error = clReleaseProgram(update_parameters_program);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the program for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseKernel(update_parameters_kernel);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the kernel for the Adam optimizer parameters update with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseProgram(error_program);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the program for the error calculation with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseKernel(error_kernel);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the kernel for the error calculation with code %d / %s\n", error, getOpenCLErrorString(error));
	error = clReleaseMemObject(errors_buffer);
	ERROR_HANDLE_INT_RETURN_INT(error, "TrainAdamGPU(): Error while releasing the errors buffer with code %d / %s\n", error, getOpenCLErrorString(error));
	free(errors_to_sum);

	// Verbose
	if (verbose > 0)
		DEBUG_PRINT("TrainAdamGPU(): Training done!\n");
	return current_epoch;
}






/**
 * @brief Train the neural network with the GPU by using
 * a batch of inputs and a batch of target outputs,
 * a number of epochs and a target error value
 * 
 * @param network					Pointer to the neural network
 * @param training_data				Training data structure (inputs, target outputs, number of inputs, batch size, test inputs percentage)
 * @param training_parameters		Training parameters structure (number of epochs, error target, optimizer, loss function, learning rate)
 * @param error_per_epoch			Pointer to the array of errors per epoch (can be NULL)
 * @param verbose					Verbose level (0: no verbose, 1: verbose, 2: very verbose, 3: all)
 * 
 * @return int						Number of epochs done, -1 if there is an error
 */
int TrainGPU(NeuralNetwork *network, TrainingData training_data, TrainingParameters training_parameters, nn_type *error_per_epoch, int verbose) {

	// Check if at least one of the two parameters is specified
	int boolean_parameters = training_parameters.nb_epochs != -1 || training_parameters.error_target != 0.0;	// 0 when none of the two parameters is specified, 1 otherwise
	ERROR_HANDLE_INT_RETURN_INT(boolean_parameters - 1, "TrainGPU(): At least the number of epochs or the error target must be specified!\n");

	// Make new training data pointers as the training data may be shuffled
	training_data.inputs = duplicateMemory(training_data.inputs, training_data.nb_inputs * sizeof(nn_type*), "TrainGPU(copy training_data.inputs)");
	training_data.targets = duplicateMemory(training_data.targets, training_data.nb_inputs * sizeof(nn_type*), "TrainGPU(copy training_data.targets)");

	// Setup OpenCL
	setupOpenCLGPU();

	// Launch the training depending on the chosen optimizer
	int code;
	/*if (strcmp(training_parameters.optimizer, "SGD") == 0 || strcmp(training_parameters.optimizer, "StochasticGradientDescent") == 0)
		code = TrainSGDGPU(network, training_data, training_parameters, error_per_epoch, verbose);

	else*/ if (strcmp(training_parameters.optimizer, "Adam") == 0 || strcmp(training_parameters.optimizer, "ADAM") == 0)
		code = TrainAdamGPU(network, training_data, training_parameters, error_per_epoch, verbose);

	// else if (strcmp(training_parameters.optimizer, "RMSProp") == 0 || strcmp(training_parameters.optimizer, "RMS") == 0)
	// 	return TrainRMSProp(network, training_data, training_parameters, verbose);

	else {
		ERROR_PRINT("TrainGPU(): Unknown optimizer: '%s'\n", training_parameters.optimizer);
		code = -1;
	}

	// Free the training data copy pointers and return
	free(training_data.inputs);
	free(training_data.targets);
	return code;
}

