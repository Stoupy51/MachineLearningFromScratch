
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/neural_network/training_gpu.h"
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
 * This program is an introduction to basic 'plus' operation using a neural network.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'basic_plus_operation' program\n");
	atexit(exitProgram);

	// Create a neural network to learn the '+' function
	int nb_neurons_per_layer[] = {2, 1};
	char *activation_functions[] = {NULL, "relu"};
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
		inputs[i][0] = a;
		inputs[i][1] = b;
		expected[i][0] = c;
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
		.nb_epochs = 100,
		.error_target = 0.00001,
		.optimizer = "Adam",			// Adaptive Moment Estimation
		.loss_function_name = "MSE",	// Mean Squared Error
		.learning_rate = 0.001
	};
	struct timeval start, end;
	st_gettimeofday(start, NULL);
	code = TrainGPU(&network_plus, training_data, training_parameters, NULL, 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while training the neural network\n");
	st_gettimeofday(end, NULL);
	INFO_PRINT("main(): Total training time: "STR_YELLOW_R("%.3f")"s\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0);

	///// Test the neural network
	nn_type **test_inputs = &inputs[NB_TOTAL_DATA - NB_TOTAL_DATA];
	nn_type **test_expected = &expected[NB_TOTAL_DATA - NB_TOTAL_DATA];
	nn_type **test_outputs;
	nn_type *test_outputs_flat_matrix = try2DFlatMatrixAllocation((void***)&test_outputs, NB_TOTAL_DATA, network_plus.output_layer->nb_neurons, sizeof(nn_type), "main()");
	FeedForwardGPU(&network_plus, test_inputs, test_outputs, NB_TOTAL_DATA);
	int nb_errors = 0;
	for (int i = 0; i < NB_TOTAL_DATA; i++) {
		int a = doubleToInt(test_inputs[i][0]);
		int b = doubleToInt(test_inputs[i][1]);
		int c = doubleToInt(test_outputs[i][0]);
		int d = doubleToInt(test_expected[i][0]);
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

