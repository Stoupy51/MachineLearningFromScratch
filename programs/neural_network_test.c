
#include <sys/types.h>

#include "../src/universal_utils.h"
#include "../src/math/sigmoid.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training.h"
#include "../src/utils/random_array_values.h"

#define NEURAL_NETWORK_PATH "bin/neural_network_test.bin"

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

	// Print end of program
	INFO_PRINT("exitProgram(): End of program, press enter to exit.\n");
	getchar();
	exit(0);
}

/**
 * This program is an introduction test to Neural Networks.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'neural_network_test' program.\n");
	atexit(exitProgram);

	// Try to load a neural network
	NeuralNetworkD *loaded_network = loadNeuralNetworkD(NEURAL_NETWORK_PATH, sigmoid);
	NeuralNetworkD network;
	if (loaded_network == NULL) {
		WARNING_PRINT("main(): No neural network found, creating a new one.\n");

		// Create a neural network using double as type
		int nb_neurons_per_layer[] = {128*128, 4096, 4096, 4096, 256*256};
		//int nb_neurons_per_layer[] = {16*16, 4096, 4096, 4096, 32*32};
		int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
		network = createNeuralNetworkD(nb_layers, nb_neurons_per_layer, 0.1, sigmoid);
	} else {
		INFO_PRINT("main(): Neural network found, using it.\n");
		network = *loaded_network;
		free(loaded_network);
	}

	// Print the neural network information
	printNeuralNetworkD(network);

	// Create an input array and an excepted output array
	double *input = (double*)malloc(network.input_layer->nb_neurons * sizeof(double));
	double *excepted_output = (double*)malloc(network.output_layer->nb_neurons * sizeof(double));

	// Make random input and excepted output
	fillRandomDoubleArray(input, network.input_layer->nb_neurons, 0.0, 1.0);
	fillRandomDoubleArray(excepted_output, network.output_layer->nb_neurons, 0.0, 1.0);
	
	// Train the neural network
	NeuralNetworkDtrain(&network, input, excepted_output);

	// Run the neural network with the input array and get the output array
	NeuralNetworkDfeedForward(&network, input);

	// Free the input and excepted output arrays
	free(input);
	free(excepted_output);

	// Save the neural network to a file and another human readable file
	//saveNeuralNetworkD(network, NEURAL_NETWORK_PATH, 0);

	// Final print and return
	INFO_PRINT("main(): End of program.\n");
	return 0;
}

