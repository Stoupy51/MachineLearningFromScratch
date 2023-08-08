
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/list/token_dictionary.h"
#include "../src/utils/text_file.h"

#define TRAINING_FOLDER_PATH "data/gpt_train/"
#define NEURAL_NETWORK_PATH "bin/s_gpt.nn"

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
	mainInit("main(): Launching 'gpt_train' program\n");
	atexit(exitProgram);

	// Load the training data into one big string
	INFO_PRINT("main(): Loading training data\n");
	char *training_data = readTextFromFolder(TRAINING_FOLDER_PATH);
	ERROR_HANDLE_PTR_RETURN_INT(training_data, "main(): Error while loading training data '%s'\n", TRAINING_FOLDER_PATH);
	int nb_characters = strlen(training_data);
	INFO_PRINT("main(): %d characters in the training data: [%d, %d, ...]\n", nb_characters, (int)training_data[0], (int)training_data[1]);

	// Convert the list of tokens into chunks of tokens
	int chunk_size = 8;	// Maximum context length for the transformer predictions
	int nb_chunks = nb_characters / chunk_size + 1;
	char **chunks = selectRandomChunksFromCharArray(training_data, nb_characters, nb_chunks, chunk_size);
	PRINTER("main(): %d chunks of %d characters: [[%d, %d, ...], [%d, %d, ...], ...]\n", nb_chunks, chunk_size, chunks[0][0], chunks[0][1], chunks[1][0], chunks[1][1]);

	// Create the neural network
	int number_of_bits = sizeof(char) * 8;
	int input_layer_size = number_of_bits * chunk_size;
	int nb_neurons_per_layer[] = {input_layer_size, input_layer_size, input_layer_size, number_of_bits};
	char *activation_functions[] = {NULL, "relu", "relu", "softmax"};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	NeuralNetwork network;
	int code = initNeuralNetwork(&network, nb_layers, nb_neurons_per_layer, activation_functions, "cross_entropy", 0.1, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

