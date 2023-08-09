
#include "../src/universal_utils.h"
#include "../src/math/sigmoid.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/neural_network/training_gpu.h"

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
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

/**
 * This program is an introduction to basic 'and' operation using a neural network.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'image_upscaler_training' program\n");
	atexit(exitProgram);

	// Create a neural network to learn the AND function
	WARNING_PRINT("main(): No neural network found, creating a new one\n");
	int nb_neurons_per_layer[] = {2, 1};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	NeuralNetwork network_and = initNeuralNetwork(nb_layers, nb_neurons_per_layer, 1.0, sigmoid);

	// Print the neural network information
	printNeuralNetwork(network_and);
	printActivationValues(network_and);

	// Train the neural network while the error is not low enough
	WARNING_PRINT("main(): Training the neural network\n");
	double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double outputs[4][1] = {{0}, {0}, {0}, {1}};
	int nb_training_data = sizeof(inputs) / sizeof(inputs[0]);
	double error = 1.0;
	int tries = 0;
	while (error > 0.0001) {
		tries++;
		error = 0.0;
		for (int i = 0; i < nb_training_data; i++) {
			NeuralNetworktrainCPU(&network_and, inputs[i], outputs[i]);
			//NeuralNetworktrainStepByStepGPU(&network_and, inputs[i], outputs[i], 1);
			//NeuralNetworktrainGPU(&network_and, inputs[i], outputs[i], 1);

			double local_error = 0.0;
			for (int j = 0; j < network_and.output_layer->nb_neurons; j++) {
				double delta = network_and.output_layer->deltas[j];
				local_error += delta * delta;
			}
			error += local_error;
		}
		error /= nb_training_data;
		if (tries < 4 || tries % 2500 == 0)
			INFO_PRINT("Trie nb %d, error: %f (%f)\n", tries, error, error * nb_training_data);
	}
	INFO_PRINT("main(): Training done in %d tries\n", tries);

	// Test the neural network
	WARNING_PRINT("main(): Testing the neural network\n");
	for (int i = 0; i < nb_training_data; i++) {
		FeedForwardCPU(&network_and, inputs[i]);
		//NeuralNetworkfeedForwardGPU(&network_and, inputs[i], 1);
		printActivationValues(network_and);
	}

	// Free the neural network & free private GPU buffers
	freeNeuralNetwork(&network_and);
	stopNeuralNetworkGpuOpenCL();
	stopNeuralNetworkGpuBuffersOpenCL();

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

