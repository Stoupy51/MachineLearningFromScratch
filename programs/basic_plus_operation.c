
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

	// Print end of program
	INFO_PRINT("exitProgram(): End of program, press enter to exit\n");
	getchar();
	exit(0);
}

int doubleToInt(double d) {
	return (int)(d + 0.5);
}

/**
 * @brief Convert an integer to a binary array of
 * double values (0.0 or 1.0) for the neural network.
 * 
 * @param value The integer to convert
 * @param array The array to fill
 * @param offset The offset in the array
 * 
 * @return void
 */
void convertIntToBinaryDoubleArray(int value, double* array, int array_offset) {
	for (int i = 0; i < 32; i++)
		array[array_offset + i] = (value & (1 << i)) ? 1.0 : 0.0;
}

/**
 * @brief Convert a binary array of double values (0.0 or 1.0) to an integer.
 * 
 * @param array The array to convert
 * @param offset The offset in the array
 * 
 * @return The converted integer
 */
int convertBinaryDoubleArrayToInt(double* array, int array_offset) {
	int value = 0;
	for (int i = 0; i < 32; i++)
		value |= doubleToInt(array[array_offset + i]) << i;
	return value;
}

/**
 * This program is an introduction to basic 'plus' operation using a neural network.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'basic_plus_operation' program\n");
	atexit(exitProgram);

	// Create a neural network to learn the '+' function
	int nb_neurons_per_layer[] = {64, 48, 32};
	char *activation_functions[] = {NULL, "sigmoid", "sigmoid"};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	NeuralNetwork network_plus;
	int code = initNeuralNetwork(&network_plus, nb_layers, nb_neurons_per_layer, activation_functions, 0.1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network_plus);
	printActivationValues(network_plus);

	// TODO

	///// Final part
	// Free the neural network
	freeNeuralNetwork(&network_plus);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

