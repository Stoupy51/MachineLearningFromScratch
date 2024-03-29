
#include "neural_utils.h"
#include "activation_functions.h"
#include "../utils/random_array_values.h"
#include "../universal_utils.h"

/**
 * @brief Function that creates a neural network
 * 
 * @param network					Pointer to the neural network to create
 * @param nb_layers					Number of layers in the neural network
 * @param nb_neurons_per_layer		Array of int representing the number of neurons per layer
 * @param activation_function_names	Array of strings representing the activation function names of each layer
 * 
 * @return int		0 if the neural network was saved successfully, 1 otherwise
 */
int initNeuralNetwork(NeuralNetwork *network, int nb_layers, int nb_neurons_per_layer[], char **activation_function_names) {

	// Init easy value
	network->nb_layers = nb_layers;

	// Calculate all required memory size
	long long required_memory_size = 0;
	for (int i = 0; i < nb_layers; i++) {
		required_memory_size += sizeof(NeuronLayer);	// NeuronLayer struct
		required_memory_size += (long long)nb_neurons_per_layer[i] * sizeof(nn_type);	// activations_values
		if (i == 0) continue;
		required_memory_size += (long long)nb_neurons_per_layer[i] * (long long)nb_neurons_per_layer[i - 1] * sizeof(nn_type);	// weights_flat
		required_memory_size += (long long)nb_neurons_per_layer[i] * sizeof(nn_type*);	// weights
		required_memory_size += (long long)nb_neurons_per_layer[i] * sizeof(nn_type);	// biases
	}
	network->memory_size = required_memory_size;

	// If the memory size is too big (> 1 GB), ask the user if he wants to continue
	if (network->memory_size > 1000000000) {

		// Print a warning message
		WARNING_PRINT("initNeuralNetwork(): The required memory of the neural network is huge (");
		if (network->memory_size < 1000) { PRINTER("%lld Bytes)\n", network->memory_size); }
		else if (network->memory_size < 1000000) { PRINTER("%.2Lf KB [%lld])\n", (long double)network->memory_size / 1000, network->memory_size); }
		else if (network->memory_size < 1000000000) { PRINTER("%.2Lf MB [%lld])\n", (long double)network->memory_size / 1000000, network->memory_size); }
		else { PRINTER("%.2Lf GB [%lld])\n", (long double)network->memory_size / 1000000000, network->memory_size); }

		// Ask the user if he wants to continue
		WARNING_PRINT("initNeuralNetwork(): Do you want to continue? (Y/n) ");
		char answer = getchar();
		if (answer != 'Y' && answer != 'y' && answer != '\n') {
			ERROR_PRINT("initNeuralNetwork(): Cancelled the creation of the neural network\n");
			exit(EXIT_FAILURE);
		}
	}

	// Allocate memory for the layers
	long long this_malloc_size = nb_layers * sizeof(NeuronLayer);
	network->layers = mallocBlocking(this_malloc_size, "initNeuralNetwork()");
	memset(network->layers, 0, this_malloc_size);

	// Create the next layers
	for (int i = 0; i < nb_layers; i++) {

		// Create the layer
		network->layers[i].nb_neurons = nb_neurons_per_layer[i];
		network->layers[i].nb_inputs_per_neuron = (i == 0) ? 0 : nb_neurons_per_layer[i - 1];	// Depends on the previous layer

		///// Allocate memory for the activations_values (nb_neurons * sizeof(nn_type))
		network->layers[i].activations_values = mallocBlocking(network->layers[i].nb_neurons * sizeof(nn_type), "initNeuralNetwork()");
		fillRandomFloatArray(network->layers[i].activations_values, network->layers[i].nb_neurons, -1.0, 1.0);
		network->layers[i].activation_function_name = activation_function_names[i] == NULL ? "NULL" : strdup(activation_function_names[i]);

		// Stop here if it's the first layer (no weights, biases, etc.)
		if (i == 0) continue;

		// Assign the activation function
		network->layers[i].activation_function = get_activation_function(activation_function_names[i]);
		network->layers[i].activation_function_derivative = get_activation_function_derivative(activation_function_names[i]);

		///// Allocate memory for the weights (nb_neurons * nb_inputs_per_neuron * sizeof(nn_type))
		network->layers[i].weights_flat = try2DFlatMatrixAllocation((void***)&network->layers[i].weights, network->layers[i].nb_neurons, network->layers[i].nb_inputs_per_neuron, sizeof(nn_type), "initNeuralNetwork()");
		fillRandomFloatArray(network->layers[i].weights_flat, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron, -1.0, 1.0);

		///// Allocate memory for the biases, and the deltas (nb_neurons * sizeof(nn_type))
		this_malloc_size = network->layers[i].nb_neurons * sizeof(nn_type);
		network->layers[i].biases = mallocBlocking(this_malloc_size, "initNeuralNetwork()");
		fillRandomFloatArray(network->layers[i].biases, network->layers[i].nb_neurons, -1.0, 1.0);
	}

	// Assign the input and output layers pointers
	network->input_layer = &(network->layers[0]);
	network->output_layer = &(network->layers[nb_layers - 1]);

	// Return the neural network
	return 0;
}


/**
 * @brief Function that prints a neural network
 * It only prints information about the neural network, not the weights, biases, etc.
 * 
 * @param network	Neural network to print
 */
void printNeuralNetwork(NeuralNetwork network) {
	INFO_PRINT("printNeuralNetwork():\n");

	// Print Layers information
	PRINTER(CYAN"- Number of layers: "YELLOW"%d"CYAN"\n", network.nb_layers);
	for (int i = 0; i < network.nb_layers; i++) {
		PRINTER("  - Layer "YELLOW"%2d"CYAN":\t"YELLOW"%4d"CYAN" neurons\t[Parameters: "YELLOW"%8d"CYAN" weights, "YELLOW"%4d"CYAN" biases]\t(Activation function: %s)\n",
			i,
			network.layers[i].nb_neurons,
			network.layers[i].nb_neurons * network.layers[i].nb_inputs_per_neuron,
			network.layers[i].nb_neurons,
			network.layers[i].activation_function_name
		);
	}

	// Print other information
	PRINTER(CYAN"- Input layer:\t\t"YELLOW"0x%p"CYAN" (nb_neurons: "YELLOW"%4d"CYAN", nb_inputs_per_neuron: "YELLOW"%4d"CYAN")\n", (void*)network.input_layer, network.input_layer->nb_neurons, network.input_layer->nb_inputs_per_neuron);
	PRINTER(CYAN"- Output layer:\t\t"YELLOW"0x%p"CYAN" (nb_neurons: "YELLOW"%4d"CYAN", nb_inputs_per_neuron: "YELLOW"%4d"CYAN")\n", (void*)network.output_layer, network.output_layer->nb_neurons, network.output_layer->nb_inputs_per_neuron);

	// Print total neurons and weights
	long long total_neurons = 0;
	long long total_weights = 0;
	for (int i = 0; i < network.nb_layers; i++) {
		total_neurons += network.layers[i].nb_neurons;
		total_weights += network.layers[i].nb_neurons * network.layers[i].nb_inputs_per_neuron;
	}
	PRINTER(CYAN"- Total neurons:\t"YELLOW"%lld\n", total_neurons);
	PRINTER(CYAN"- Total weights:\t");
	if (total_weights < 1000)
		{ PRINTER(YELLOW"%lld"CYAN"\n", total_weights); }
	else if (total_weights < 1000000)
		{ PRINTER(YELLOW"%5.2Lf"CYAN" K  ("YELLOW"%lld"CYAN" weights)\n", (long double)total_weights / 1000, total_weights); }
	else if (total_weights < 1000000000)
		{ PRINTER(YELLOW"%5.2Lf"CYAN" M  ("YELLOW"%lld"CYAN" weights)\n", (long double)total_weights / 1000000, total_weights); }
	else
		{ PRINTER(YELLOW"%5.2Lf"CYAN" B  ("YELLOW"%lld"CYAN" weights)\n", (long double)total_weights / 1000000000, total_weights); }

	// Print memory size
	if (network.memory_size < 1000)
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%lld"CYAN" Bytes\n", network.memory_size); }
	else if (network.memory_size < 1000000)
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%5.2Lf"CYAN" KB ("YELLOW"%lld"CYAN" bytes)\n", (long double)network.memory_size / 1000, network.memory_size); }
	else if (network.memory_size < 1000000000)
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%5.2Lf"CYAN" MB ("YELLOW"%lld"CYAN" bytes)\n", (long double)network.memory_size / 1000000, network.memory_size); }
	else
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%5.2Lf"CYAN" GB ("YELLOW"%lld"CYAN" bytes)\n", (long double)network.memory_size / 1000000000, network.memory_size); }

	// End of the function
	PRINTER(RESET"\n");
}

/**
 * @brief Function that prints the activations_values of a neural network
 * 
 * @param network	Neural network to print the activations_values of
 */
void printActivationValues(NeuralNetwork network) {
	INFO_PRINT("printActivationValues():\n");
	int max_neurons = 0;
	for (int i = 0; i < network.nb_layers; i++)
		if (network.layers[i].nb_neurons > max_neurons) max_neurons = network.layers[i].nb_neurons;
	for (int i = 0; i < max_neurons; i++) {
		for (int j = 0; j < network.nb_layers; j++) {
			if (i < network.layers[j].nb_neurons)
				{ PRINTER("%.2"NN_FORMAT"\t", network.layers[j].activations_values[i]); }
			else
				{ PRINTER("\t"); }
		}
		PRINTER("\n");
	}
}

/**
 * @brief Function that frees a neural network
 * and sets all the bytes of the neural network to 0.
 * 
 * @param network	Neural network to free
 */
void freeNeuralNetwork(NeuralNetwork *network) {
	if (network->nb_layers == 0) return;

	// Free the layers
	for (int i = 0; i < network->nb_layers; i++) {
		free(network->layers[i].activations_values);
		if (i == 0) continue;

		free2DFlatMatrix((void**)network->layers[i].weights, network->layers[i].weights_flat, network->layers[i].nb_neurons);
		free(network->layers[i].biases);
	}

	// Free the layers array and memset the network to 0
	free(network->layers);
	memset(network, 0, sizeof(NeuralNetwork));
}

/**
 * @brief Function that saves a neural network in a file
 * only by using binary data so the file is not human readable.
 * 
 * @param network							Neural network to save
 * @param filename							Filename to save the neural network to
 * @param generate_human_readable_file		1 if you also want to generate a human readable file, 0 otherwise
 * 
 * @return int		0 if the neural network was saved successfully, 1 otherwise
 */
int saveNeuralNetwork(NeuralNetwork network, char *filename, int generate_human_readable_file) {
	
	// Open the file
	FILE *file = fopen(filename, "wb");
	ERROR_HANDLE_PTR_RETURN_INT(file, "saveNeuralNetwork(): Could not open/create the file '%s'\n", filename);

	// Write the number of layers, and the memory size
	fwrite(&(network.nb_layers), sizeof(int), 1, file);
	fwrite(&(network.memory_size), sizeof(long long), 1, file);

	// For each layer of the neural network,
	for (int i = 0; i < network.nb_layers; i++) {

		// Write the number of neurons and the number of inputs per neuron
		fwrite(&(network.layers[i].nb_neurons), sizeof(int), 1, file);
		fwrite(&(network.layers[i].nb_inputs_per_neuron), sizeof(int), 1, file);

		// Write the activation function name
		int activation_function_name_length = strlen(network.layers[i].activation_function_name);
		fwrite(&activation_function_name_length, sizeof(int), 1, file);
		fwrite(network.layers[i].activation_function_name, sizeof(char) * activation_function_name_length, 1, file);
		if (i == 0) continue;

		// Write the weights_flat, and the biases
		fwrite(network.layers[i].weights_flat, network.layers[i].nb_neurons * network.layers[i].nb_inputs_per_neuron * sizeof(nn_type), 1, file);
		fwrite(network.layers[i].biases, network.layers[i].nb_neurons * sizeof(nn_type), 1, file);
	}

	// Close the file
	fclose(file);

	// If the user wants to generate a human readable file,
	if (generate_human_readable_file) {

		// Get new path
		char *new_path = mallocBlocking(strlen(filename) + sizeof("_human_readable.txt"), "saveNeuralNetworkD()");
		ERROR_HANDLE_PTR_RETURN_INT(new_path, "saveNeuralNetworkD(): Could not allocate memory for the new path\n");
		strcpy(new_path, filename);
		strcat(new_path, "_human_readable.txt");
		INFO_PRINT("saveNeuralNetworkD(): Generating human readable file '%s'\n", new_path);

		// Open the file
		file = fopen(new_path, "w");
		ERROR_HANDLE_PTR_RETURN_INT(file, "saveNeuralNetworkD(): Could not open/create the file '%s'\n", new_path);

		// Write the number of layers, and the memory size
		fprintf(file, "nb_layers: %d\n", network.nb_layers);
		fprintf(file, "memory_size: %lld\n", network.memory_size);

		// For each layer of the neural network,
		for (int i = 0; i < network.nb_layers; i++) {

			// Write the number of neurons and the number of inputs per neuron
			fprintf(file, "\n");
			fprintf(file, "layer: %d\n", i);
			fprintf(file, "nb_neurons: %d\n", network.layers[i].nb_neurons);
			fprintf(file, "nb_inputs_per_neuron: %d\n", network.layers[i].nb_inputs_per_neuron);

			// Write the activation function name
			fprintf(file, "activation_function_name: %s\n", network.layers[i].activation_function_name);
			if (i == 0) continue;

			// Write the formula for each neuron of the layer: "neuron[i] = (weight1 * input1) + (weight2 * input2) + ... + (weightN * inputN) + bias"
			fprintf(file, "formula for each neuron: ");
			for (int j = 0; j < network.layers[i].nb_neurons; j++) {
				fprintf(file, "neuron[%d] = (%.2"NN_FORMAT") + (", j, network.layers[i].biases[j]);
				for (int k = 0; k < network.layers[i].nb_inputs_per_neuron; k++) {
					fprintf(file, "%.2"NN_FORMAT" * %.2"NN_FORMAT"", network.layers[i].weights[j][k], network.layers[i - 1].activations_values[k]);
					if (k != network.layers[i].nb_inputs_per_neuron - 1)
						fprintf(file, " + ");
				}
				fprintf(file, ") = %.2"NN_FORMAT"\n", network.layers[i].activations_values[j]);
			}
		}

		// Close the file
		fclose(file);
	}

	// Return that everything went well
	return 0;
}

/**
 * @brief Function that loads a neural network from a file
 * only by using binary data so the file is not human readable.
 * 
 * @param network		Pointer to the neural network to load
 * @param filename		Filename to load the neural network from
 * 
 * @return NeuralNetwork		Neural network loaded
 */
int loadNeuralNetwork(NeuralNetwork *network, char *filename) {

	// Open the file
	FILE *file = fopen(filename, "rb");
	ERROR_HANDLE_PTR_RETURN_INT(file, "loadNeuralNetwork(): Could not open the file '%s'\n", filename);

	// Read the number of layers, and the memory size
	int code = fread(&(network->nb_layers), sizeof(int), 1, file);
	ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the number of layers\n");
	code = fread(&(network->memory_size), sizeof(long long), 1, file);
	ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the memory size\n");

	// Allocate memory for the layers
	long long this_malloc_size = network->nb_layers * sizeof(NeuronLayer);
	network->layers = mallocBlocking(this_malloc_size, "loadNeuralNetwork()");
	memset(network->layers, 0, this_malloc_size);

	// For each layer of the neural network,
	for (int i = 0; i < network->nb_layers; i++) {

		// Read the number of neurons and the number of inputs per neuron
		code = fread(&(network->layers[i].nb_neurons), sizeof(int), 1, file);
		ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the number of neurons of layer %d\n", i);
		code = fread(&(network->layers[i].nb_inputs_per_neuron), sizeof(int), 1, file);
		ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the number of inputs per neuron of layer %d\n", i);
		network->layers[i].activations_values = mallocBlocking(network->layers[i].nb_neurons * sizeof(nn_type), "loadNeuralNetwork()");
		
		// Read the activation function name
		int activation_function_name_length;
		code = fread(&activation_function_name_length, sizeof(int), 1, file);
		ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the activation function name length of layer %d\n", i);
		network->layers[i].activation_function_name = mallocBlocking(sizeof(char) * (activation_function_name_length + 1), "loadNeuralNetwork()");
		code = fread(network->layers[i].activation_function_name, sizeof(char) * activation_function_name_length, 1, file);
		ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the activation function name of layer %d\n", i);
		network->layers[i].activation_function_name[activation_function_name_length] = '\0';
		if (i == 0) continue;

		// Allocate memory for the weights_flat, the weights, the biases, the deltas and the errors
		this_malloc_size = (long long)network->layers[i].nb_neurons * (long long)network->layers[i].nb_inputs_per_neuron * sizeof(nn_type);
		network->layers[i].weights_flat = mallocBlocking(this_malloc_size, "loadNeuralNetwork()");
		network->layers[i].weights = mallocBlocking(network->layers[i].nb_neurons * sizeof(nn_type*), "loadNeuralNetwork()");
		network->layers[i].biases = mallocBlocking(network->layers[i].nb_neurons * sizeof(nn_type), "loadNeuralNetwork()");

		// Read the weights_flat, and the biases
		code = fread(network->layers[i].weights_flat, network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(nn_type), 1, file);
		ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the weights_flat of layer %d\n", i);
		code = fread(network->layers[i].biases, network->layers[i].nb_neurons * sizeof(nn_type), 1, file);
		ERROR_HANDLE_INT_RETURN_INT(code - 1, "loadNeuralNetwork(): Could not read the biases of layer %d\n", i);

		// Assign the weights_flat addresses to the weights
		for (int j = 0; j < network->layers[i].nb_neurons; j++)
			network->layers[i].weights[j] = &(network->layers[i].weights_flat[j * network->layers[i].nb_inputs_per_neuron]);
		
		// Assign the activation function
		network->layers[i].activation_function = get_activation_function(network->layers[i].activation_function_name);
		network->layers[i].activation_function_derivative = get_activation_function_derivative(network->layers[i].activation_function_name);
	}

	// Assign the input and output layers pointers
	network->input_layer = &(network->layers[0]);
	network->output_layer = &(network->layers[network->nb_layers - 1]);

	// Close the file
	fclose(file);

	// Return that everything went well
	return 0;
}

/**
 * @brief Function that clones a neural network (deep copy)
 * 
 * @param network_to_clone		Neural network to clone
 * @param cloned_network		Pointer to the cloned neural network
 */
void deepCloneNeuralNetwork(NeuralNetwork *network_to_clone, NeuralNetwork *cloned_network) {

	// Clone the easy values
	*cloned_network = *network_to_clone;

	// Duplicate the layers array
	long long this_malloc_size = cloned_network->nb_layers * sizeof(NeuronLayer);
	cloned_network->layers = duplicateMemory(network_to_clone->layers, this_malloc_size, "deepCloneNeuralNetwork()");

	// For each layer of the neural network,
	for (int i = 0; i < network_to_clone->nb_layers; i++) {

		// Clone the activation function name
		int activation_function_name_length = strlen(network_to_clone->layers[i].activation_function_name);
		cloned_network->layers[i].activation_function_name = duplicateMemory(network_to_clone->layers[i].activation_function_name, sizeof(char) * (activation_function_name_length + 1), "deepCloneNeuralNetwork()");

		// Stop here if it's the first layer (no weights, biases, etc.)
		if (i == 0) continue;

		// Duplicate the weights_flat, the weights, and the biases
		this_malloc_size = (long long)network_to_clone->layers[i].nb_neurons * (long long)network_to_clone->layers[i].nb_inputs_per_neuron * sizeof(nn_type);
		cloned_network->layers[i].weights_flat = duplicateMemory(network_to_clone->layers[i].weights_flat, this_malloc_size, "deepCloneNeuralNetwork()");
		this_malloc_size = network_to_clone->layers[i].nb_neurons * sizeof(nn_type);
		cloned_network->layers[i].weights = duplicateMemory(network_to_clone->layers[i].weights, this_malloc_size, "deepCloneNeuralNetwork()");
		cloned_network->layers[i].activations_values = duplicateMemory(network_to_clone->layers[i].activations_values, this_malloc_size, "deepCloneNeuralNetwork()");
		cloned_network->layers[i].biases = duplicateMemory(network_to_clone->layers[i].biases, this_malloc_size, "deepCloneNeuralNetwork()");

		// Assign the weights_flat addresses to the weights
		for (int j = 0; j < network_to_clone->layers[i].nb_neurons; j++)
			cloned_network->layers[i].weights[j] = &(cloned_network->layers[i].weights_flat[j * cloned_network->layers[i].nb_inputs_per_neuron]);
	}

	// Assign the input and output layers pointers
	cloned_network->input_layer = &(cloned_network->layers[0]);
	cloned_network->output_layer = &(cloned_network->layers[cloned_network->nb_layers - 1]);
}

