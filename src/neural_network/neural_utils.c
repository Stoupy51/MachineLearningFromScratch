
#include "neural_utils.h"

#include <fcntl.h>

// Generate a random double/float between min and max
double generateRandomDouble(double min, double max) { return (double)rand() / RAND_MAX * (max - min) + min; }
float generateRandomFloat(float min, float max) { return (float)rand() / RAND_MAX * (max - min) + min; }



/**
 * @brief Function that creates a neural network using double as type
 * 
 * @param nb_layers				Number of layers in the neural network
 * @param nb_neurons_per_layer	Array of int representing the number of neurons per layer
 * @param learning_rate			Learning rate of the neural network: how fast the network learns by adjusting the weights
 * 								(0.0 = no learning, 1.0 = full learning)
 * @param activation_function	Activation function of the neural network: how the network will activate the neurons
 * 
 * @return NeuralNetworkD		Neural network created
 */
NeuralNetworkD createNeuralNetworkD(int nb_layers, int nb_neurons_per_layer[], double learning_rate, double (*activation_function)(double)) {

	// Create the neural network
	NeuralNetworkD network;
	network.nb_layers = nb_layers;
	network.learning_rate = learning_rate;
	network.activation_function = activation_function;
	
	// Allocate memory for the layers
	size_t this_malloc_size = nb_layers * sizeof(NeuronLayerD);
	network.memory_size = this_malloc_size;
	network.layers = (NeuronLayerD*)malloc(this_malloc_size);
	if (network.layers == NULL) {
		ERROR_PRINT("createNeuralNetworkD(): Could not allocate memory for the layers.\n");
		exit(EXIT_FAILURE);
	}

	// Memset the layers to 0
	memset(network.layers, 0, this_malloc_size);

	// Create the next layers
	for (int i = 0; i < nb_layers; i++) {
		
		// Create the layer
		network.layers[i].nb_neurons = nb_neurons_per_layer[i];
		network.layers[i].nb_inputs_per_neuron = (i == 0) ? 0 : nb_neurons_per_layer[i - 1];	// Depends on the previous layer

		///// Allocate memory for the activations_values (nb_neurons * sizeof(double))
		this_malloc_size = network.layers[i].nb_neurons * sizeof(double);
		network.memory_size += this_malloc_size;
		network.layers[i].activations_values = (double*)malloc(this_malloc_size);
		if (network.layers[i].activations_values == NULL) {
			ERROR_PRINT("createNeuralNetworkD(): Could not allocate memory for the activations_values.\n");
			exit(EXIT_FAILURE);
		}

		// Initialize the activations_values to random values
		for (int j = 0; j < network.layers[i].nb_neurons; j++)
			network.layers[i].activations_values[j] = generateRandomDouble(-1.0, 1.0);
		
		// Stop here if it's the first layer (no weights, biases, etc.)
		if (i == 0) continue;
		
		///// Allocate memory for the weights (nb_neurons * nb_inputs_per_neuron * sizeof(double))
		// Allocate memory for the weights_flat
		this_malloc_size = network.layers[i].nb_neurons * network.layers[i].nb_inputs_per_neuron * sizeof(double);
		network.memory_size += this_malloc_size;
		network.layers[i].weights_flat = (double*)malloc(this_malloc_size);
		if (network.layers[i].weights_flat == NULL) {
			ERROR_PRINT("createNeuralNetworkD(): Could not allocate memory for the weights.\n");
			exit(EXIT_FAILURE);
		}

		// Allocate memory for the weights (2D array)
		this_malloc_size = network.layers[i].nb_neurons * sizeof(double*);
		network.memory_size += this_malloc_size;
		network.layers[i].weights = (double**)malloc(this_malloc_size);
		if (network.layers[i].weights == NULL) {
			ERROR_PRINT("createNeuralNetworkD(): Could not allocate memory for the weights.\n");
			exit(EXIT_FAILURE);
		}

		// Assign the weights_flat addresses to the weights
		for (int j = 0; j < network.layers[i].nb_neurons; j++)
			network.layers[i].weights[j] = &(network.layers[i].weights_flat[j * network.layers[i].nb_inputs_per_neuron]);

		///// Allocate memory for the biases (nb_neurons * sizeof(double))
		this_malloc_size = network.layers[i].nb_neurons * sizeof(double);
		network.memory_size += this_malloc_size;
		network.layers[i].biases = (double*)malloc(this_malloc_size);
		if (network.layers[i].biases == NULL) {
			ERROR_PRINT("createNeuralNetworkD(): Could not allocate memory for the biases.\n");
			exit(EXIT_FAILURE);
		}

		// Initialize the weights and biases with random values
		for (int j = 0; j < network.layers[i].nb_neurons; j++) {
			for (int k = 0; k < network.layers[i].nb_inputs_per_neuron; k++)
				network.layers[i].weights[j][k] = generateRandomDouble(-1.0, 1.0);
			network.layers[i].biases[j] = generateRandomDouble(-1.0, 1.0);
		}

		///// Allocate memory for the deltas (nb_neurons * sizeof(double))
		this_malloc_size = network.layers[i].nb_neurons * sizeof(double);
		network.memory_size += this_malloc_size;
		network.layers[i].deltas = (double*)malloc(this_malloc_size);
		if (network.layers[i].deltas == NULL) {
			ERROR_PRINT("createNeuralNetworkD(): Could not allocate memory for the deltas.\n");
			exit(EXIT_FAILURE);
		}
	}

	// Assign the input and output layers pointers
	network.input_layer = &(network.layers[0]);
	network.output_layer = &(network.layers[nb_layers - 1]);

	// Return the neural network
	return network;
}

/**
 * @brief Function that prints a neural network using double as type
 * It only prints information about the neural network, not the weights, biases, etc.
 * 
 * @param network	Neural network to print
 * 
 * @return void
 */
void printNeuralNetworkD(NeuralNetworkD network) {
	INFO_PRINT("printNeuralNetworkD():\n");
	PRINTER(CYAN"- Number of layers:\t"YELLOW"%d"CYAN"\n", network.nb_layers);
	for (int i = 0; i < network.nb_layers; i++)
		{ PRINTER("  - Number of neurons in layer "YELLOW"%d"CYAN":\t"YELLOW"%d"CYAN"\n", i, network.layers[i].nb_neurons); }
	PRINTER(CYAN"- Learning rate:\t"YELLOW"%f\n", network.learning_rate);
	PRINTER(CYAN"- Input layer:\t\t"YELLOW"0x%p"CYAN" (nb_neurons: "YELLOW"%d"CYAN", nb_inputs_per_neuron: "YELLOW"%d"CYAN")\n", (void*)network.input_layer, network.input_layer->nb_neurons, network.input_layer->nb_inputs_per_neuron);
	PRINTER(CYAN"- Output layer:\t\t"YELLOW"0x%p"CYAN" (nb_neurons: "YELLOW"%d"CYAN", nb_inputs_per_neuron: "YELLOW"%d"CYAN")\n", (void*)network.output_layer, network.output_layer->nb_neurons, network.output_layer->nb_inputs_per_neuron);
	if (network.memory_size < 1000)
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%zu"CYAN" Bytes\n", network.memory_size); }
	else if (network.memory_size < 1000000)
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%.2Lf"CYAN" KB\n", (long double)network.memory_size / 1000); }
	else if (network.memory_size < 1000000000)
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%.2Lf"CYAN" MB\n", (long double)network.memory_size / 1000000); }
	else
		{ PRINTER(CYAN"- Memory size:\t\t"YELLOW"%.2Lf"CYAN" GB\n", (long double)network.memory_size / 1000000000); }
	int total_neurons = 0;
	int total_weights = 0;
	for (int i = 0; i < network.nb_layers; i++) {
		total_neurons += network.layers[i].nb_neurons;
		total_weights += network.layers[i].nb_neurons * network.layers[i].nb_inputs_per_neuron;
	}
	PRINTER(CYAN"- Total neurons:\t"YELLOW"%d\n", total_neurons);
	PRINTER(CYAN"- Total weights:\t"YELLOW"%d\n", total_weights);
	PRINTER(RESET"\n");
}

/**
 * @brief Function that frees a neural network using double as type
 * and sets all the bytes of the neural network to 0.
 * 
 * @param network	Neural network to free
 * 
 * @return void
 */
void freeNeuralNetworkD(NeuralNetworkD *network) {

	// Free the layers
	for (int i = 0; i < network->nb_layers; i++) {
		free(network->layers[i].weights_flat);
		free(network->layers[i].weights);
		free(network->layers[i].activations_values);
		free(network->layers[i].biases);
	}

	// Free the layers array
	free(network->layers);

	// Memset the network to 0
	memset(network, 0, sizeof(NeuralNetworkD));
}

/**
 * @brief Function that saves a neural network using double as type in a file
 * only by using binary data so the file is not human readable.
 * 
 * @param network							Neural network to save
 * @param filename							Filename to save the neural network to
 * @param generate_human_readable_file		1 if you also want to generate a human readable file, 0 otherwise
 * 
 * @return int		0 if the neural network was saved successfully, 1 otherwise
 */
int saveNeuralNetworkD(NeuralNetworkD network, char *filename, int generate_human_readable_file) {
	
	// Open the file
	int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	ERROR_HANDLE_INT_RETURN_INT(fd, "saveNeuralNetworkD(): Could not open/create the file '%s'.\n", filename);

	// Write the number of layers, the learning rate, and the memory size
	ERROR_HANDLE_INT_RETURN_INT(write(fd, &(network.nb_layers), sizeof(int)), "saveNeuralNetworkD(): Could not write the number of layers in the file '%s'.\n", filename);
	ERROR_HANDLE_INT_RETURN_INT(write(fd, &(network.learning_rate), sizeof(double)), "saveNeuralNetworkD(): Could not write the learning rate in the file '%s'.\n", filename);
	ERROR_HANDLE_INT_RETURN_INT(write(fd, &(network.memory_size), sizeof(size_t)), "saveNeuralNetworkD(): Could not write the memory size in the file '%s'.\n", filename);

	// For each layer of the neural network, write the number of neurons
	for (int i = 0; i < network.nb_layers; i++) {
		ERROR_HANDLE_INT_RETURN_INT(write(fd, &(network.layers[i].nb_neurons), sizeof(int)), "saveNeuralNetworkD(): Could not write the number of neurons in the file '%s'.\n", filename);
	}

	// For each layer of the neural network,
	for (int i = 0; i < network.nb_layers; i++) {

		// Write the weights_flat
		ERROR_HANDLE_INT_RETURN_INT(write(fd, network.layers[i].weights_flat, network.layers[i].nb_neurons * network.layers[i].nb_inputs_per_neuron * sizeof(double)), "saveNeuralNetworkD(): Could not write the weights_flat in the file '%s'.\n", filename);

		// Write the activations_values, and the biases
		ERROR_HANDLE_INT_RETURN_INT(write(fd, network.layers[i].activations_values, network.layers[i].nb_neurons * sizeof(double)), "saveNeuralNetworkD(): Could not write the activations_values in the file '%s'.\n", filename);
		ERROR_HANDLE_INT_RETURN_INT(write(fd, network.layers[i].biases, network.layers[i].nb_neurons * sizeof(double)), "saveNeuralNetworkD(): Could not write the biases in the file '%s'.\n", filename);
	}

	// Close the file
	ERROR_HANDLE_INT_RETURN_INT(close(fd), "saveNeuralNetworkD(): Could not close the file '%s'.\n", filename);

	// If the user wants to generate a human readable file,
	if (generate_human_readable_file == 1) {

		// Get new path
		char *new_path = (char*)malloc(strlen(filename) + sizeof("_human_readable.txt"));
		ERROR_HANDLE_PTR_RETURN_INT(new_path, "saveNeuralNetworkD(): Could not allocate memory for the new path.\n");
		strcpy(new_path, filename);
		strcat(new_path, "_human_readable.txt");

		// Open the file
		FILE *file = fopen("neural_network_readable.txt", "w");
		ERROR_HANDLE_PTR_RETURN_INT(file, "saveNeuralNetworkD(): Could not open/create the file '%s'.\n", new_path);

		// Write the number of layers, the learning rate, and the memory size
		fprintf(file, "\n");
		fprintf(file, "Number of layers:\t%d\n", network.nb_layers);
		fprintf(file, "Learning rate:\t\t%f\n", network.learning_rate);
		fprintf(file, "Memory size:\t\t%zu Bytes\n\n", network.memory_size);

		// For each layer of the neural network, write the number of neurons
		for (int i = 0; i < network.nb_layers; i++) {
			fprintf(file, "Number of neurons in layer %d:\t%d\n", i, network.layers[i].nb_neurons);
		}

		// For each layer of the neural network, write the number of inputs per neuron
		fprintf(file, "\n");
		for (int i = 0; i < network.nb_layers; i++) {
			fprintf(file, "Number of inputs per neuron in layer %d:\t%d\n", i, network.layers[i].nb_inputs_per_neuron);
		}

		// For each layer of the neural network,
		fprintf(file, "\n");
		for (int i = 0; i < network.nb_layers; i++) {

			// Write the weights_flat
			fprintf(file, "Weights_flat in layer %d:\n", i);
			for (int j = 0; j < network.layers[i].nb_neurons; j++) {
				for (int k = 0; k < network.layers[i].nb_inputs_per_neuron; k++) {
					fprintf(file, "%.1f\t", network.layers[i].weights[j][k]);
				}
				fprintf(file, "\n");
			}

			// Write the activations_values, and the biases
			fprintf(file, "\n");
			fprintf(file, "Activations_values in layer %d:\n", i);
			for (int j = 0; j < network.layers[i].nb_neurons; j++) {
				fprintf(file, "%.1f\t", network.layers[i].activations_values[j]);
			}
			fprintf(file, "\n\n");
			fprintf(file, "Biases in layer %d:\n", i);
			for (int j = 0; j < network.layers[i].nb_neurons; j++) {
				fprintf(file, "%.1f\t", network.layers[i].biases[j]);
			}
			fprintf(file, "\n\n");

			// Write the deltas
			fprintf(file, "Deltas in layer %d:\n", i);
			for (int j = 0; j < network.layers[i].nb_neurons; j++) {
				fprintf(file, "%.1f\t", network.layers[i].deltas[j]);
			}
			fprintf(file, "\n\n\n");
		}

		// Close the file
		fclose(file);
	}

	// Return that everything went well
	return 0;
}

/**
 * @brief Function that loads a neural network using double as type from a file
 * only by using binary data so the file is not human readable.
 * 
 * @param filename				Filename to load the neural network from
 * @param activation_function	Activation function of the neural network: how the network will activate the neurons
 * 								(because the function can't be saved in the file)
 * 
 * @return NeuralNetworkD		Neural network loaded
 */
NeuralNetworkD* loadNeuralNetworkD(char *filename, double (*activation_function)(double)) {

	// Open the file
	int fd = open(filename, O_RDONLY);
	ERROR_HANDLE_INT_RETURN_NULL(fd, "loadNeuralNetworkD(): Could not open the file '%s'\n", filename);

	// Create the neural network structure
	NeuralNetworkD *network = (NeuralNetworkD*)malloc(sizeof(NeuralNetworkD));
	memset(network, 0, sizeof(NeuralNetworkD));

	// Read the number of layers, the learning rate, and the memory size
	ERROR_HANDLE_INT_RETURN_NULL(read(fd, &(network->nb_layers), sizeof(int)), "loadNeuralNetworkD(): Could not read the number of layers in the file '%s'.\n", filename);
	ERROR_HANDLE_INT_RETURN_NULL(read(fd, &(network->learning_rate), sizeof(double)), "loadNeuralNetworkD(): Could not read the learning rate in the file '%s'.\n", filename);
	ERROR_HANDLE_INT_RETURN_NULL(read(fd, &(network->memory_size), sizeof(size_t)), "loadNeuralNetworkD(): Could not read the memory size in the file '%s'.\n", filename);

	// Allocate memory for the layers
	size_t this_malloc_size = network->nb_layers * sizeof(NeuronLayerD);
	network->layers = (NeuronLayerD*)malloc(this_malloc_size);
	ERROR_HANDLE_PTR_RETURN_NULL(network->layers, "loadNeuralNetworkD(): Could not allocate memory for the layers.\n");

	// Memset the layers to 0
	memset(network->layers, 0, this_malloc_size);

	// For each layer of the neural network, read the number of neurons
	for (int i = 0; i < network->nb_layers; i++) {
		ERROR_HANDLE_INT_RETURN_NULL(read(fd, &(network->layers[i].nb_neurons), sizeof(int)), "loadNeuralNetworkD(): Could not read the number of neurons in the file '%s'.\n", filename);
	}

	// For each layer of the neural network,
	for (int i = 0; i < network->nb_layers; i++) {

		// Calculate the number of inputs per neuron (depends on the previous layer)
		network->layers[i].nb_inputs_per_neuron = (i == 0) ? 0 : network->layers[i - 1].nb_neurons;

		// Read the weights_flat
		this_malloc_size = network->layers[i].nb_neurons * network->layers[i].nb_inputs_per_neuron * sizeof(double);
		network->layers[i].weights_flat = (double*)malloc(this_malloc_size);
		ERROR_HANDLE_PTR_RETURN_NULL(network->layers[i].weights_flat, "loadNeuralNetworkD(): Could not allocate memory for the weights_flat.\n");
		ERROR_HANDLE_INT_RETURN_NULL(read(fd, network->layers[i].weights_flat, this_malloc_size), "loadNeuralNetworkD(): Could not read the weights_flat in the file '%s'.\n", filename);

		// Allocate memory for the weights (2D array)
		this_malloc_size = network->layers[i].nb_neurons * sizeof(double*);
		network->layers[i].weights = (double**)malloc(this_malloc_size);
		ERROR_HANDLE_PTR_RETURN_NULL(network->layers[i].weights, "loadNeuralNetworkD(): Could not allocate memory for the weights.\n");

		// Assign the weights_flat addresses to the weights
		for (int j = 0; j < network->layers[i].nb_neurons; j++)
			network->layers[i].weights[j] = &(network->layers[i].weights_flat[j * network->layers[i].nb_inputs_per_neuron]);
		
		// Read the activations_values, and the biases
		this_malloc_size = network->layers[i].nb_neurons * sizeof(double);
		network->layers[i].activations_values = (double*)malloc(this_malloc_size);
		ERROR_HANDLE_PTR_RETURN_NULL(network->layers[i].activations_values, "loadNeuralNetworkD(): Could not allocate memory for the activations_values.\n");
		ERROR_HANDLE_INT_RETURN_NULL(read(fd, network->layers[i].activations_values, this_malloc_size), "loadNeuralNetworkD(): Could not read the activations_values in the file '%s'.\n", filename);
		network->layers[i].biases = (double*)malloc(this_malloc_size);
		ERROR_HANDLE_PTR_RETURN_NULL(network->layers[i].biases, "loadNeuralNetworkD(): Could not allocate memory for the biases.\n");
		ERROR_HANDLE_INT_RETURN_NULL(read(fd, network->layers[i].biases, this_malloc_size), "loadNeuralNetworkD(): Could not read the biases in the file '%s'.\n", filename);

		// Allocate memory for the deltas
		network->layers[i].deltas = (double*)malloc(this_malloc_size);
		ERROR_HANDLE_PTR_RETURN_NULL(network->layers[i].deltas, "loadNeuralNetworkD(): Could not allocate memory for the deltas.\n");
	}

	// Assign the input and output layers pointers
	network->input_layer = &(network->layers[0]);
	network->output_layer = &(network->layers[network->nb_layers - 1]);

	// Assign the activation function
	network->activation_function = activation_function;

	// Close the file
	ERROR_HANDLE_INT_RETURN_NULL(close(fd), "loadNeuralNetworkD(): Could not close the file '%s'.\n", filename);

	// Return the neural network
	return network;
}

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

