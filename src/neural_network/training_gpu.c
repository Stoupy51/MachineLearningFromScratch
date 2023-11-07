
#include "training_gpu.h"
#include "activation_functions.h"
#include "loss_functions.h"
#include "training_utils.h"
#include "../universal_utils.h"
#include "../st_benchmark.h"

#include "../gpu/gpu_utils.h"



char feed_forward_function[1024] = ""
"kernel void feed_forward(global "NN_STRING"* previous_layer_activation_values, global "NN_STRING"* weights, global "NN_STRING"* biases, global "NN_STRING"* activation_values, int current_layer_size, int previous_layer_size) {"
	"int neuron = get_global_id(0);"
	"int weight_index = neuron * previous_layer_size;"
	""NN_STRING" sum = biases[neuron];"
	"for (int i = 0; i < previous_layer_size; i++) {"
		"sum += previous_layer_activation_values[i] * weights[weight_index + i];"
	"}"
	"activation_values[neuron] = __ACTIVATION_FUNCTION__;"
"}";

/**
 * @brief Get the feed forward function for the given activation function.
 * 
 * @param activation_function_name		Name of the activation function
 * 
 * @return The feed forward function string (must be freed)
 */
char* getFeedForwardFunction(char* activation_function_name) {

	// Copy the feed forward function
	char *feed_forward_function_copy = mallocBlocking(sizeof(feed_forward_function), "getFeedForwardFunction(feed_forward_function_copy)");
	strcpy(feed_forward_function_copy, feed_forward_function);

	// Depending on the activation function, replace the activation function in the feed forward function
	if (strcmp(activation_function_name, "sigmoid") == 0)
		replaceString(feed_forward_function_copy, sizeof(feed_forward_function), "__ACTIVATION_FUNCTION__", "1.0 / (1.0 + exp(-sum))");
	else if (strcmp(activation_function_name, "relu") == 0)
		replaceString(feed_forward_function_copy, sizeof(feed_forward_function), "__ACTIVATION_FUNCTION__", "sum > 0.0 ? sum : 0.0");
	else if (strcmp(activation_function_name, "tanh") == 0)
		replaceString(feed_forward_function_copy, sizeof(feed_forward_function), "__ACTIVATION_FUNCTION__", "tanh(sum)");
	else if (strcmp(activation_function_name, "identity") == 0)
		replaceString(feed_forward_function_copy, sizeof(feed_forward_function), "__ACTIVATION_FUNCTION__", "sum");
	else if (strcmp(activation_function_name, "softmax") == 0)
		replaceString(feed_forward_function_copy, sizeof(feed_forward_function), "__ACTIVATION_FUNCTION__", "exp(sum)");
	else {
		ERROR_PRINT("Activation function not found: '%s'\n", activation_function_name);
		exit(EXIT_FAILURE);
	}

	// Return the feed forward function
	return feed_forward_function_copy;
}

// Softmax activation function launched after "feed_forward_function" assuming that exp() is already applied to the activation values
char feed_forward_softmax[] = ""
"kernel void feed_forward_softmax(global "NN_STRING"* activation_values, int current_layer_size) {"
	""NN_STRING" sum = 0.0;"
	"for (int i = 0; i < current_layer_size; i++) {"
		"sum += activation_values[i];"
	"}"
	"for (int i = 0; i < current_layer_size; i++) {"
		"activation_values[i] /= sum;"
	"}"
"}";

int is_oc_setup = 0;
struct opencl_context_t oc;

/**
 * @brief Structure for each layer of the neural network.
 * 
 * @param activation_values			Activation values of the layer
 * @param weights					Weights of the layer
 * @param biases					Biases of the layer
 * 
 * @param program					Program of the layer
 * @param kernel					Kernel of the layer
 * @param optionnal_2nd_program		Optionnal second program of the layer (for softmax)
 * @param optionnal_2nd_kernel		Optionnal second kernel of the layer (for softmax)
*/
typedef struct LayerForGPU {
	cl_mem activation_values;
	cl_mem weights;
	cl_mem biases;

	cl_program program;
	cl_kernel kernel;
	cl_program optionnal_2nd_program;
	cl_kernel optionnal_2nd_kernel;
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

		// Create the program and the kernel
		char *feed_forward_function_copy = getFeedForwardFunction(network->layers[layer].activation_function_name);
		layers[layer].program = clCreateProgramWithSource(oc.context, 1, (const char**)&feed_forward_function_copy, NULL, &error);
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
		error = clSetKernelArg(layers[layer].kernel, 4, sizeof(int), &network->layers[layer].nb_neurons);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 4 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		error = clSetKernelArg(layers[layer].kernel, 5, sizeof(int), &network->layers[layer].nb_inputs_per_neuron);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 5 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

		// If the activation function is softmax, create the program and the kernel for softmax
		if (strcmp(network->layers[layer].activation_function_name, "softmax") == 0) {
			layers[layer].optionnal_2nd_program = clCreateProgramWithSource(oc.context, 1, (const char**)&feed_forward_softmax, NULL, &error);
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
			error = clSetKernelArg(layers[layer].optionnal_2nd_kernel, 0, sizeof(cl_mem), &layers[layer].activation_values);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 0 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			error = clSetKernelArg(layers[layer].optionnal_2nd_kernel, 1, sizeof(int), &network->layers[layer].nb_neurons);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 1 for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		}

		// Free the feed forward function copy
		free(feed_forward_function_copy);
	}

	// Wait for the preparation events to finish and release them
	error = clWaitForEvents(events_count, events);
	ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while waiting for the preparation events to finish with code %d / %s\n", error, getOpenCLErrorString(error));
	for (int i = 0; i < events_count; i++) {
		error = clReleaseEvent(events[i]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the preparation event %d with code %d / %s\n", i, error, getOpenCLErrorString(error));
	}
	events_count = 0;

	// For each sample, launch the feed forward kernel
	for (int sample = 0; sample < batch_size; sample++) {

		// Set the arguments of the kernel (input and output buffers)
		error = clSetKernelArg(layers[1].kernel, 0, sizeof(cl_mem), &inputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 0 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		error = clSetKernelArg(layers[network->nb_layers - 1].kernel, 3, sizeof(cl_mem), &outputs_buffers[sample]);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 0 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));

		// If the final activation function is softmax, use the output buffer as input
		if (layers[network->nb_layers - 1].optionnal_2nd_kernel != NULL) {
			error = clSetKernelArg(layers[network->nb_layers - 1].optionnal_2nd_kernel, 0, sizeof(cl_mem), &outputs_buffers[sample]);
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while setting the kernel argument 0 for sample %d with code %d / %s\n", sample, error, getOpenCLErrorString(error));
		}

		// Launch the feed forward kernel for each layer
		for (int layer = 1; layer < network->nb_layers; layer++) {

			// Launch the feed forward kernel
			size_t global_dimensions[] = { network->layers[layer].nb_neurons, 0, 0 };
			if (events_count == 0)
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[layer].kernel, 1, NULL, global_dimensions, NULL, 0, NULL, &events[0]);
			else
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[layer].kernel, 1, NULL, global_dimensions, NULL, 1, &events[events_count - 1], &events[events_count]);
			events_count++;
			ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while launching the feed forward kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));

			// If the final activation function is softmax, launch the softmax kernel
			if (layers[layer].optionnal_2nd_kernel != NULL) {
				error = clEnqueueNDRangeKernel(oc.command_queue, layers[layer].optionnal_2nd_kernel, 1, NULL, global_dimensions, NULL, 1, &events[events_count - 1], &events[events_count]);
				events_count++;
				ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while launching the softmax kernel for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
			}
		}
	}

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
	for (int layer = 0; layer < network->nb_layers; layer++) {
		error = clReleaseMemObject(layers[layer].activation_values);
		ERROR_HANDLE_INT_RETURN_INT(error, "FeedForwardGPU(): Error while releasing the activation values buffer for layer %d with code %d / %s\n", layer, error, getOpenCLErrorString(error));
		if (layer == 0)
			continue;
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
int TrainAdam(NeuralNetwork *network, TrainingData training_data, TrainingParameters training_parameters, nn_type *error_per_epoch, int verbose) {

	// TODO
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
	if (strcmp(training_parameters.optimizer, "SGD") == 0 || strcmp(training_parameters.optimizer, "StochasticGradientDescent") == 0)
		code = TrainSGD(network, training_data, training_parameters, error_per_epoch, verbose);

	else if (strcmp(training_parameters.optimizer, "Adam") == 0 || strcmp(training_parameters.optimizer, "ADAM") == 0)
		code = TrainAdam(network, training_data, training_parameters, error_per_epoch, verbose);

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

