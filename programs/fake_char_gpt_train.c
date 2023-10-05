
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
	INFO_PRINT("main(): %d characters in the vocabulary: \\0%s\n", vocabulary_size, vocabulary + 1);

	// Convert the list of tokens into chunks of tokens
	int chunk_size = 8;	// Maximum context length
	int nb_chunks = nb_characters - chunk_size;
	if (nb_chunks > 50000) nb_chunks = 50000;
	char **chunks = selectRandomChunksFromCharArray(training_text, nb_characters, nb_chunks, chunk_size);
	INFO_PRINT("main(): %d chunks of %d characters: [[%d, %d, ...], [%d, %d, ...], ...]\n", nb_chunks, chunk_size, chunks[0][0], chunks[0][1], chunks[1][0], chunks[1][1]);

	// Create the neural network
	int nb_neurons_per_layer[] = {chunk_size, 384, 384, vocabulary_size};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	char *activation_functions[] = {NULL, "relu", "relu", "softmax"};
	NeuralNetwork network;
	int code = initNeuralNetwork(&network, nb_layers, nb_neurons_per_layer, activation_functions, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network);

	// Prepare the vocabulary correspondance array for optimization
	int *vocabulary_correspondance_array = correspondanceArrayWithVocabularyIndex(vocabulary, vocabulary_size);

	// Prepare the training data
	nn_type **inputs;
	nn_type **targets;
	nn_type *inputs_flat_matrix = try2DFlatMatrixAllocation((void***)&inputs, nb_chunks, chunk_size, sizeof(nn_type), "main()");
	nn_type *targets_flat_matrix = try2DFlatMatrixAllocation((void***)&targets, nb_chunks, vocabulary_size, sizeof(nn_type), "main()");
	for (int i = 0; i < nb_chunks; i++) {

		for (int j = 0; j < chunk_size; j++)
			inputs[i][j] = (nn_type)(chunks[i][j]);

		// Use the correspondance array to get the index of the chracter in the vocabulary
		// Example: 'a' -> 97 -> 1 (in the vocabulary)
		memset(targets[i], 0, vocabulary_size * sizeof(nn_type));
		int targeted_char_index = vocabulary_correspondance_array[(int)chunks[i][chunk_size]];
		targets[i][targeted_char_index] = 1.0;
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
		.learning_rate = 0.01
	};

	struct timeval start, end;
	st_gettimeofday(start, NULL);
	code = TrainCPU(&network, training_data, training_parameters, 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while training the neural network\n");
	st_gettimeofday(end, NULL);
	INFO_PRINT("main(): Total training time: "STR_YELLOW_R("%.3f")"s\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0);

	// Test the neural network
	INFO_PRINT("main(): Testing the neural network\n");
	nn_type **predictions;
	nn_type *predictions_flat = try2DFlatMatrixAllocation((void***)&predictions, nb_chunks, vocabulary_size, sizeof(nn_type), "main()");
	FeedForwardCPU(&network, inputs, predictions, nb_chunks);
	int nb_errors = 0;
	for (int i = 0; i < nb_chunks; i++) {
		int predicted_char_index = getIndexOfMaxFromDoubleArray(predictions[i], vocabulary_size);
		int targeted_char_index = getIndexOfMaxFromDoubleArray(targets[i], vocabulary_size);
		if (predicted_char_index != targeted_char_index && nb_errors++ < 10)
			ERROR_PRINT("main(): Input '%8s', predicted '%c' instead of '%c'\n", chunks[i], predicted_char_index == 0 ? '0' : vocabulary[predicted_char_index], targeted_char_index == 0 ? '0' : vocabulary[targeted_char_index]);
	}
	INFO_PRINT("main(): Success rate: %d/%d (%.2f%%)\n", nb_chunks - nb_errors, nb_chunks, (double)(nb_chunks - nb_errors) / (double)nb_chunks * 100.0);

	// Save the neural network
	code = saveNeuralNetwork(network, NEURAL_NETWORK_PATH, 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while saving the neural network\n");

	// Free the training data and the vocabulary
	free(training_text);
	free2DFlatMatrix((void**)inputs, inputs_flat_matrix, nb_chunks);
	free2DFlatMatrix((void**)targets, targets_flat_matrix, nb_chunks);
	free2DFlatMatrix((void**)predictions, predictions_flat, nb_chunks);
	free(vocabulary);
	free(chunks);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

