
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_utils.h"
#include "../src/neural_network/activation_functions.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/utils/text_file.h"
#include "../src/st_benchmark.h"

#define TRAINING_FOLDER_PATH "data/gpt_train/"
#define NEURAL_NETWORK_PATH "bin/char_gpt.nn"

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
 * This program is an introduction to GPT (Generative Pre-trained Transformer).
 * A transformer is a neural network that uses attention mechanisms to process sequences of words,
 * and a pre-trained transformer is a transformer that has been trained on a large corpus of text,
 * typically on a large-scale unsupervised language modeling task, such as predicting the next word in a sequence.
 * 
 * This GPT works character by character, not word by word.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'char_gpt_train' program\n");
	atexit(exitProgram);

	// Load the training data into one big string
	INFO_PRINT("main(): Loading training data\n");
	char *training_data = readTextFromFolder(TRAINING_FOLDER_PATH);
	ERROR_HANDLE_PTR_RETURN_INT(training_data, "main(): Error while loading training data '%s'\n", TRAINING_FOLDER_PATH);
	int nb_characters = strlen(training_data);
	INFO_PRINT("main(): %d characters in the training data: [%d, %d, ...]\n", nb_characters, (int)training_data[0], (int)training_data[1]);

	// Create vocabulary from the training data (list of characters)
	INFO_PRINT("main(): Creating vocabulary\n");
	char *vocabulary = mallocBlocking(256 * sizeof(char), "main(vocabulary)");
	memset(vocabulary, 0, 256 * sizeof(char));
	int vocabulary_size = 0;
	for (int i = 0; i < nb_characters; i++) {
		char c = training_data[i];
		if (strchr(vocabulary, c) == NULL) {
			vocabulary[vocabulary_size] = c;
			vocabulary_size++;
		}
	}

	// Sort the vocabulary
	for (int i = 0; i < vocabulary_size; i++) {
		for (int j = i + 1; j < vocabulary_size; j++) {
			if (vocabulary[i] > vocabulary[j]) {
				char tmp = vocabulary[i];
				vocabulary[i] = vocabulary[j];
				vocabulary[j] = tmp;
			}
		}
	}
	INFO_PRINT("main(): %d characters in the vocabulary: %s\n", vocabulary_size, vocabulary);

	// Convert the list of tokens into chunks of tokens
	int chunk_size = 1;	// Maximum context length for the transformer predictions
	int size_of_char = sizeof(char);
	int nb_chunks = nb_characters - chunk_size;
	char **chunks = selectRandomChunksFromCharArray(training_data, nb_characters, nb_chunks, chunk_size);
	INFO_PRINT("main(): %d chunks of %d characters: [[%d, %d, ...], [%d, %d, ...], ...]\n", nb_chunks, chunk_size, chunks[0][0], chunks[0][1], chunks[1][0], chunks[1][1]);

	// Prepare the training data
	INFO_PRINT("main(): Preparing training data\n");
	nn_type **xb;
	nn_type **yb;
	nn_type *xb_flat = try2DFlatMatrixAllocation((void***)&xb, nb_chunks, chunk_size * size_of_char, sizeof(nn_type), "main(xb_flat)");
	nn_type *yb_flat = try2DFlatMatrixAllocation((void***)&yb, nb_chunks, size_of_char, sizeof(nn_type), "main(yb_flat)");
	for (int i = 0; i < nb_chunks; i++) {
		for (int j = 0; j < chunk_size; j++) {
			xb[i][j] = (nn_type)(chunks[i][j]) / vocabulary_size;
		}
		yb[i][0] = (nn_type)(chunks[i][chunk_size]) / vocabulary_size;
	}
	INFO_PRINT("main(): Training data prepared: [[%.4f, ...] => [%.4f], [%.4f, ...] => [%.4f], ...]\n", (double)xb[0][0], (double)yb[0][0], (double)xb[1][0], (double)yb[1][0]);

	// Create the neural network
	#define NB_TEST_DATA_PERCENTAGE 10
	#define BATCH_SIZE 1
	#define NB_EPOCHS 100
	#define ERROR_TARGET 0.000001
	#define VERBOSE 1
	int input_size = chunk_size * size_of_char;
	int nb_neurons_per_layer[] = {input_size, 128, size_of_char};
	char *activation_functions[] = {NULL, "tanh", "softmax"};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	NeuralNetwork network;
	int code = initNeuralNetwork(&network, nb_layers, nb_neurons_per_layer, activation_functions, "MSE", 0.01, 1);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network);

	// Train the neural network
	char buffer[16];
	ST_BENCHMARK_SOLO_COUNT(buffer, {
		code = TrainCPU(&network, xb, yb,
			nb_chunks,
			NB_TEST_DATA_PERCENTAGE,
			BATCH_SIZE,
			NB_EPOCHS,
			ERROR_TARGET,
			VERBOSE,
			"SGD"
		);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while training the neural network\n");
	}, "", 1, 1);
	INFO_PRINT("main(): Total training time: "STR_YELLOW_R("%s")"s\n", buffer);






	// Free the training data
	free2DFlatMatrix((void**)xb, xb_flat, nb_chunks);
	free2DFlatMatrix((void**)yb, yb_flat, nb_chunks);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

