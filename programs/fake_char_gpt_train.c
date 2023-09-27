
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/utils/text_file.h"
#include "../src/st_benchmark.h"

#define TRAINING_FOLDER_PATH "data/words/"
#define NEURAL_NETWORK_PATH "bin/fake_char_gpt.nn"

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
 * This program is a try to mimic GPT (Generative Pre-trained Transformer).
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'fake_char_gpt' program\n");
	atexit(exitProgram);

	// Load the training data into one big string
	INFO_PRINT("main(): Loading training data\n");
	char *training_text = readTextFromFolder(TRAINING_FOLDER_PATH);
	ERROR_HANDLE_PTR_RETURN_INT(training_text, "main(): Error while loading training data '%s'\n", TRAINING_FOLDER_PATH);
	int nb_characters = strlen(training_text);

	// Create vocabulary from the training data (list of characters)
	int vocabulary_size = 0;
	char *vocabulary = generateCharVocabularyFromText(training_text, &vocabulary_size);
	INFO_PRINT("main(): %d characters in the vocabulary: %s\n", vocabulary_size, vocabulary);
	vocabulary_size++; // Add the '\0' character

	// Convert the list of tokens into chunks of tokens
	int chunk_size = 8;	// Maximum context length
	int nb_chunks = nb_characters - chunk_size;
	if (nb_chunks > 50000) nb_chunks = 50000;
	char **chunks = selectRandomChunksFromCharArray(training_text, nb_characters, nb_chunks, chunk_size);
	INFO_PRINT("main(): %d chunks of %d characters: [[%d, %d, ...], [%d, %d, ...], ...]\n", nb_chunks, chunk_size, chunks[0][0], chunks[0][1], chunks[1][0], chunks[1][1]);

	// Create the neural network
	int nb_neurons_per_layer[] = {chunk_size, 256, vocabulary_size};
	char *activation_functions[] = {NULL, "relu", "softmax"};
	NeuralNetwork network;
	int code = initNeuralNetwork(&network, sizeof(nb_neurons_per_layer) / sizeof(int), nb_neurons_per_layer, activation_functions, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network);

	// Prepare the training data
	nn_type **inputs;
	nn_type **targets;
	nn_type *inputs_flat_matrix = try2DFlatMatrixAllocation((void***)&inputs, nb_chunks, chunk_size, sizeof(nn_type), "main()");
	nn_type *targets_flat_matrix = try2DFlatMatrixAllocation((void***)&targets, nb_chunks, vocabulary_size, sizeof(nn_type), "main()");
	for (int i = 0; i < nb_chunks; i++) {

		for (int j = 0; j < chunk_size; j++)
			inputs[i][j] = (nn_type)(chunks[i][j]);

		memset(targets[i], 0, vocabulary_size * sizeof(nn_type));
		int targeted_char_index = (int)(chunks[i][chunk_size - 1]);
		targets[i][targeted_char_index] = 1;
	}

	// Train the neural network
	TrainingData training_data = {
		.inputs = inputs,
		.targets = targets,
		.nb_inputs = nb_chunks,
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
	code = TrainCPU(&network, training_data, training_parameters, 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while training the neural network\n");
	st_gettimeofday(end, NULL);
	INFO_PRINT("main(): Total training time: "STR_YELLOW_R("%.3f")"s\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0);

	// Save the neural network
	code = saveNeuralNetwork(network, NEURAL_NETWORK_PATH, 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while saving the neural network\n");

	// Free the training data and the vocabulary
	free(training_text);
	free2DFlatMatrix((void**)inputs, inputs_flat_matrix, nb_chunks);
	free2DFlatMatrix((void**)targets, targets_flat_matrix, nb_chunks);
	free(vocabulary);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

