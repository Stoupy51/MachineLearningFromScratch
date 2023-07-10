
#include <sys/types.h>

#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"

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
 * @brief Function that returns the exponential of x
 * 
 * @param x		Value to apply the exponential function to
 * 
 * @return double	Exponential of x
*/
double exp(double x) {
	double result = 1.0;
	for (int i = 0; i < x; i++)
		result *= 2.71828182845904523536028747135266249775724709369995;
	return result;
}

/**
 * @brief Function that returns the sigmoid of x
 * 
 * @param x		Value to apply the sigmoid function to
 * 
 * @return double	Sigmoid of x
*/
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
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

	// Create a neural network using double as type
	int nb_neurons_per_layer[] = {1024, 4096, 4096, 4096, 2048};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	NeuralNetworkD network = createNeuralNetworkD(nb_layers, nb_neurons_per_layer, 0.1, sigmoid);

	// Print the neural network information
	printNeuralNetworkD(network);

	// Free the neural network
	freeNeuralNetworkD(&network);

	// Final print and return
	INFO_PRINT("main(): End of program.\n");
	return 0;
}

