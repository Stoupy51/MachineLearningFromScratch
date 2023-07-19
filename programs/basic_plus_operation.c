
#include "../src/universal_utils.h"
#include "../src/math/sigmoid.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/neural_network/training_gpu.h"

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

	// Free private GPU buffers
	stopNeuralNetworkGpuOpenCL();
	stopNeuralNetworkGpuBuffersOpenCL();

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
	mainInit("main(): Launching 'image_upscaler_training' program\n");
	atexit(exitProgram);

	// Create a neural network to learn the '+' function
	WARNING_PRINT("main(): No neural network found, creating a new one\n");
	int nb_neurons_per_layer[] = {64, 48, 32};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	NeuralNetworkD network_plus = createNeuralNetworkD(nb_layers, nb_neurons_per_layer, 1.0, sigmoid);

	// Print the neural network information
	printNeuralNetworkD(network_plus);
	printActivationValues(network_plus);

	// Training dataset (input pairs and corresponding outputs)
	#define nb_training_data 10000
	double** inputs = (double**)malloc(nb_training_data * sizeof(double*));
	double** outputs = (double**)malloc(nb_training_data * sizeof(double*));
	for (int i = 0; i < nb_training_data; i++) {
		inputs[i] = (double*)malloc(64 * sizeof(double));
		outputs[i] = (double*)malloc(32 * sizeof(double));

		// Generate random inputs
		int a = rand() % 100;
		int b = rand() % 100;
		int c = a + b;

		// Convert the inputs to binary
		convertIntToBinaryDoubleArray(a, inputs[i], 0);
		convertIntToBinaryDoubleArray(b, inputs[i], 32);

		// Convert the outputs to binary
		convertIntToBinaryDoubleArray(c, outputs[i], 0);
	}
		


	///// Training part
	// First training
	INFO_PRINT("main(): First training\n");
	for (int i = 0; i < nb_training_data; i++) {
		NeuralNetworkDtrainCPU(&network_plus, inputs[i], outputs[i]);
	}
	INFO_PRINT("main(): First training done\n");

	// Train the neural network
	double error = 1.0;
	int tries = 0;
	while (error > 0.000001) {
		tries++;
		error = 0.0;
		for (int i = 0; i < nb_training_data; i++) {
			NeuralNetworkDtrainCPU(&network_plus, inputs[i], outputs[i]);
			double local_error = 0.0;
			for (int j = 0; j < network_plus.output_layer->nb_neurons; j++) {
				double delta = network_plus.output_layer->deltas[j];
				local_error += delta * delta;
			}
			error += local_error;
		}
		error /= nb_training_data;
		if (tries < 4 || tries % 250 == 0)
			INFO_PRINT("Trie nb %d, error: %.16f (%.16f)\n", tries, error, error * nb_training_data);
	}
	INFO_PRINT("main(): Training done in %d tries\n", tries);



	///// Testing part
	// Test the neural network
	WARNING_PRINT("main(): Testing the trained neural network with new inputs\n");
	int nb_errors = 0;
	for (int i = 0; i < nb_training_data; i++) {
		int a = rand() % 100;
		int b = rand() % 100;
		convertIntToBinaryDoubleArray(a, inputs[i], 0);
		convertIntToBinaryDoubleArray(b, inputs[i], 32);
		NeuralNetworkDfeedForwardCPU(&network_plus, inputs[i]);
		int c = convertBinaryDoubleArrayToInt(network_plus.output_layer->activations_values, 0);
		if ((a + b) != c) {
			ERROR_PRINT("main(): %d + %d = %d\n", a, b, c);
			nb_errors++;
		}
	}
	INFO_PRINT("main(): %d errors on %d tests (ratio: %.2f%%)\n", nb_errors, nb_training_data, 100.0 - ((double)nb_errors / (double)nb_training_data * 100.0));



	///// Final part
	// Save the neural network
	WARNING_PRINT("main(): Saving the neural network\n");
	int code = saveNeuralNetworkD(network_plus, "bin/plus.nn", 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while saving the neural network\n");

	// Free the neural network & free private GPU buffers
	freeNeuralNetworkD(&network_plus);
	stopNeuralNetworkGpuOpenCL();
	stopNeuralNetworkGpuBuffersOpenCL();

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

