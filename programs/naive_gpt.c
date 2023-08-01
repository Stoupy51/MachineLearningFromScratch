
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
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

int doubleToInt(nn_type d) {
	return (int)(d + 0.5);
}

/**
 * This program is an introduction to GPT (Generative Pre-trained Transformer) using a neural network.
 * Naive and first implementation without real research all thinking by myself.
 * 
 * Plan:
 * - Words are represented in a binary way as tokens (" " = 1, "bonjour" = 2, "salut" = 3, etc.) Binary up to 2^24 = 16777216 different tokens.
 * - The words are automatically assigned with a token in the file "words_tokens.txt" (generated with the program "generate_words_tokens.c").
 * - The neural network is trained to predict the next word in a sentence.
 * - The sentence is represented as a sequence of 100 words (100 tokens) where the token 0 means no word or end of sentence.
 * 
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'naive_gpt' program\n");
	atexit(exitProgram);

	// Macros
	#define NB_WORDS 100
	#define SIZE_WORD 24
	#define INPUT_SIZE (NB_WORDS * SIZE_WORD)
	#define HIDDEN_LAYER_SIZE (int)(INPUT_SIZE * 0.75)

	// Create a neural network
	int nb_neurons_per_layer[] = {INPUT_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, SIZE_WORD};
	char *activation_functions[] = {NULL, "sigmoid", "sigmoid", "sigmoid", "sigmoid"};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	if (nb_layers != sizeof(activation_functions) / sizeof(char*)) ERROR_HANDLE_INT_RETURN_INT(-1, "main(): Error, the number of layers and the number of activation functions must be the same\n");
	NeuralNetwork network;
	int code = initNeuralNetwork(&network, nb_layers, nb_neurons_per_layer, activation_functions, "MSE", 0.1, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

