
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/neural_network/training_utils.h"
#include "../src/st_benchmark.h"

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 */
void exitProgram() {

	// Print end of program
	INFO_PRINT("exitProgram(): End of program, press enter to exit\n");
	getchar();
	exit(0);
}

/**
 * @brief Convert an integer to a binary array of
 * nn_type values (0.0 or 1.0) for the neural network.
 * 
 * @param value The integer to convert
 * @param array The array to fill
 * @param offset The offset in the array
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
	NeuralNetwork network_plus;
	int code = initNeuralNetwork(&network_plus, sizeof(nb_neurons_per_layer) / sizeof(int), nb_neurons_per_layer, activation_functions);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network_plus);

	///// Create the training data
	#define NB_TOTAL_DATA 1000
	nn_type **inputs;
	nn_type **expected;
	nn_type *inputs_flat_matrix = try2DFlatMatrixAllocation((void***)&inputs, NB_TOTAL_DATA, network_plus.input_layer->nb_neurons, sizeof(nn_type), "main()");
	nn_type *outputs_flat_matrix = try2DFlatMatrixAllocation((void***)&expected, NB_TOTAL_DATA, network_plus.output_layer->nb_neurons, sizeof(nn_type), "main()");

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
	TrainingData training_data = {
		.inputs = inputs,
		.targets = expected,
		.nb_inputs = NB_TOTAL_DATA,
		.batch_size = 1,
		.test_inputs_percentage = 20
	};
	TrainingParameters training_parameters = {
		.nb_epochs = 200,
		.error_target = 0.00001,
		.optimizer = "SGD",				// StochasticGradientDescent
		.loss_function_name = "MSE",	// MeanSquaredError
		.learning_rate = 0.1
	};
	struct timeval start, end;
	st_gettimeofday(start, NULL);
	code = TrainCPU(&network_plus, training_data, training_parameters, NULL, 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while training the neural network\n");
	st_gettimeofday(end, NULL);
	INFO_PRINT("main(): Total training time: "STR_YELLOW_R("%.3f")"s\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0);

	///// Test the neural network
	nn_type **test_inputs = &inputs[NB_TOTAL_DATA - NB_TOTAL_DATA];
	nn_type **test_expected = &expected[NB_TOTAL_DATA - NB_TOTAL_DATA];
	nn_type **test_outputs;
	nn_type *test_outputs_flat_matrix = try2DFlatMatrixAllocation((void***)&test_outputs, NB_TOTAL_DATA, network_plus.output_layer->nb_neurons, sizeof(nn_type), "main()");
	FeedForwardCPU(&network_plus, test_inputs, test_outputs, NB_TOTAL_DATA);
	int nb_errors = 0;
	for (int i = 0; i < NB_TOTAL_DATA; i++) {
		int a = convertBinaryDoubleArrayToInt(test_inputs[i], 0);
		int b = convertBinaryDoubleArrayToInt(test_inputs[i], 32);
		int c = convertBinaryDoubleArrayToInt(test_outputs[i], 0);
		int d = convertBinaryDoubleArrayToInt(test_expected[i], 0);
		if (c != d && nb_errors++ < 5)
			ERROR_PRINT("main(): Error for %d + %d = %d (expected %d)\n", a, b, c, d);
	}
	INFO_PRINT("main(): Success rate: %d/%d (%.2f%%)\n", NB_TOTAL_DATA - nb_errors, NB_TOTAL_DATA, (double)(NB_TOTAL_DATA - nb_errors) / NB_TOTAL_DATA * 100.0);

	///// Final part
	// Free the neural network
	freeNeuralNetwork(&network_plus);

	// Free the training data
	free2DFlatMatrix((void**)inputs, inputs_flat_matrix, NB_TOTAL_DATA);
	free2DFlatMatrix((void**)expected, outputs_flat_matrix, NB_TOTAL_DATA);
	free2DFlatMatrix((void**)test_outputs, test_outputs_flat_matrix, NB_TOTAL_DATA);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

