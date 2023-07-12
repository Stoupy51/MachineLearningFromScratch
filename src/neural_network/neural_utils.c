
#include "neural_utils.h"

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

	// Create the input layer
	network.layers[0].nb_neurons = nb_neurons_per_layer[0];
	network.layers[0].nb_inputs_per_neuron = 0;	// No inputs for the input layer, seems logical ;-;

	// Create the next layers
	for (int i = 1; i < nb_layers; i++) {
		
		// Create the layer
		network.layers[i].nb_neurons = nb_neurons_per_layer[i];
		network.layers[i].nb_inputs_per_neuron = nb_neurons_per_layer[i - 1];	// Depends on the previous layer
		
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

		///// Allocate memory for the activations_values (nb_neurons * sizeof(double))
		this_malloc_size = network.layers[i].nb_neurons * sizeof(double);
		network.memory_size += this_malloc_size;
		network.layers[i].activations_values = (double*)malloc(this_malloc_size);
		if (network.layers[i].activations_values == NULL) {
			ERROR_PRINT("createNeuralNetworkD(): Could not allocate memory for the activations_values.\n");
			exit(EXIT_FAILURE);
		}

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
	PRINTER(CYAN"- Number of layers:\t"MAGENTA"%d\n", network.nb_layers);
	for (int i = 0; i < network.nb_layers; i++)
		{ PRINTER("  - Number of neurons in layer "MAGENTA"%d"CYAN":\t"MAGENTA"%d\n", i, network.layers[i].nb_neurons); }
	PRINTER(CYAN"- Learning rate:\t"MAGENTA"%f\n", network.learning_rate);
	PRINTER(CYAN"- Input layer:\t"MAGENTA"0x%p"CYAN" (nb_neurons: "MAGENTA"%d"CYAN", nb_inputs_per_neuron: "MAGENTA"%d"CYAN")\n", (void*)network.input_layer, network.input_layer->nb_neurons, network.input_layer->nb_inputs_per_neuron);
	PRINTER(CYAN"- Output layer:\t"MAGENTA"0x%p"CYAN" (nb_neurons: "MAGENTA"%d"CYAN", nb_inputs_per_neuron: "MAGENTA"%d"CYAN")\n", (void*)network.output_layer, network.output_layer->nb_neurons, network.output_layer->nb_inputs_per_neuron);
	if (network.memory_size < 1000)
		{ PRINTER(CYAN"- Memory size:\t\t"MAGENTA"%zu"CYAN" Bytes\n", network.memory_size); }
	else if (network.memory_size < 1000000)
		{ PRINTER(CYAN"- Memory size:\t\t"MAGENTA"%.2Lf"CYAN" KB\n", (long double)network.memory_size / 1000); }
	else if (network.memory_size < 1000000000)
		{ PRINTER(CYAN"- Memory size:\t\t"MAGENTA"%.2Lf"CYAN" MB\n", (long double)network.memory_size / 1000000); }
	else
		{ PRINTER(CYAN"- Memory size:\t\t"MAGENTA"%.2Lf"CYAN" GB\n", (long double)network.memory_size / 1000000000); }
	int total_neurons = 0;
	int total_weights = 0;
	for (int i = 0; i < network.nb_layers; i++) {
		total_neurons += network.layers[i].nb_neurons;
		total_weights += network.layers[i].nb_neurons * network.layers[i].nb_inputs_per_neuron;
	}
	PRINTER(CYAN"- Total neurons:\t"MAGENTA"%d\n", total_neurons);
	PRINTER(CYAN"- Total weights:\t"MAGENTA"%d\n", total_weights);
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

