
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

int doubleToInt(nn_type d) {
	return (int)(d + 0.5);
}

/**
 * @brief Convert an integer to a binary array of
 * nn_type values (0.0 or 1.0) for the neural network.
 * 
 * @param value The integer to convert
 * @param array The array to fill
 * @param offset The offset in the array
 * 
 * @return void
 */
void convertIntToBinaryDoubleArray(int value, nn_type* array, int array_offset) {
	for (int i = 0; i < 32; i++)
		array[array_offset + i] = (value & (1 << i)) ? 1.0 : 0.0;
}

/**
 * @brief Convert a binary array of nn_type values (0.0 or 1.0) to an integer.
 * 
 * @param array The array to convert
 * @param offset The offset in the array
 * 
 * @return The converted integer
 */
int convertBinaryDoubleArrayToInt(nn_type* array, int array_offset) {
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
	int nb_neurons_per_layer[] = {64, 48, 48, 48, 32};
	char *activation_functions[] = {NULL, "sigmoid", "sigmoid", "sigmoid", "sigmoid"};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	if (nb_layers != sizeof(activation_functions) / sizeof(char*))
		ERROR_HANDLE_INT_RETURN_INT(-1, "main(): Error, the number of layers and the number of activation functions must be the same\n");
	NeuralNetwork network_plus;
	int code = initNeuralNetwork(&network_plus, nb_layers, nb_neurons_per_layer, activation_functions, "MSE", 0.1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network_plus);

	///// Create the training data
	#define NB_TOTAL_DATA 1000
	#define NB_TEST_DATA_PERCENTAGE 20
	#define BATCH_SIZE 1
	#define NB_EPOCHS 200
	#define ERROR_TARGET 0.000001
	#define VERBOSE 3
	nn_type **inputs = mallocBlocking(NB_TOTAL_DATA * sizeof(nn_type*), "main()");
	nn_type **expected = mallocBlocking(NB_TOTAL_DATA * sizeof(nn_type*), "main()");
	for (int i = 0; i < NB_TOTAL_DATA; i++) {
		inputs[i] = mallocBlocking(network_plus.input_layer->nb_neurons * sizeof(nn_type), "main()");
		expected[i] = mallocBlocking(network_plus.output_layer->nb_neurons * sizeof(nn_type), "main()");
	}

	// Fill the training data
	#define MAX_VALUE (200 / 2)
	for (int i = 0; i < NB_TOTAL_DATA; i++) {
		int a = rand() % MAX_VALUE;
		int b = rand() % MAX_VALUE;
		int c = a + b;
		convertIntToBinaryDoubleArray(a, inputs[i], 0);
		convertIntToBinaryDoubleArray(b, inputs[i], 32);
		convertIntToBinaryDoubleArray(c, expected[i], 0);
	}

	// Train the neural network
	code = NeuralNetworkTrainCPUMultiThreads(&network_plus, inputs, expected,
		NB_TOTAL_DATA,
		NB_TEST_DATA_PERCENTAGE,
		BATCH_SIZE,
		NB_EPOCHS,
		ERROR_TARGET,
		VERBOSE
	);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while training the neural network\n");

	///// Test the neural network
	int nb_test_data = NB_TOTAL_DATA * NB_TEST_DATA_PERCENTAGE / 100;
	nn_type **test_inputs = &inputs[NB_TOTAL_DATA - nb_test_data];
	nn_type **test_expected = &expected[NB_TOTAL_DATA - nb_test_data];
	int nb_errors = 0;
	for (int i = 0; i < nb_test_data; i++) {

		// Feed forward
		nn_type *test_output = mallocBlocking(network_plus.output_layer->nb_neurons * sizeof(nn_type), "main()");
		NeuralNetworkFeedForwardCPUMultiThreads(&network_plus, test_inputs[i], test_output);

		// Print the test results
		int a = convertBinaryDoubleArrayToInt(test_inputs[i], 0);
		int b = convertBinaryDoubleArrayToInt(test_inputs[i], 32);
		int c = convertBinaryDoubleArrayToInt(test_output, 0);
		int d = convertBinaryDoubleArrayToInt(test_expected[i], 0);
		if (c != d) {
			nb_errors++;
			ERROR_PRINT("main(): Error for %d + %d = %d (expected %d)\n", a, b, c, d);
		}
	}
	INFO_PRINT("main(): Success rate: %d/%d (%.2f%%)\n", nb_test_data - nb_errors, nb_test_data, (nb_test_data - nb_errors) * 100.0 / nb_test_data);

	///// Final part
	// Free the neural network
	freeNeuralNetwork(&network_plus);

	// Free the training data
	for (int i = 0; i < NB_TOTAL_DATA; i++) {
		free(inputs[i]);
		free(expected[i]);
	}
	free(inputs);
	free(expected);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

