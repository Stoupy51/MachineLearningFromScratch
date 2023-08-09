
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/list/token_dictionary.h"
#include "../src/utils/text_file.h"
#include "../src/st_benchmark.h"

#define DICTIONARY_TOKEN_PATH "data/token_dictionary.txt"
#define NEURAL_NETWORK_PATH "bin/naive_gpt.nn"

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

int doubleToInt(nn_type d) { return (int)(d + 0.5); }

void convertXbIntToBinaryDoubleArray(int n, nn_type *binary, int number_of_bits) {
	for (int i = 0; i < number_of_bits; i++)
		binary[i] = (n & (1 << i)) ? 1.0 : 0.0;
}

int convertBinaryDoubleArrayToXbInt(nn_type *binary, int number_of_bits) {
	int n = 0;
	for (int i = 0; i < number_of_bits; i++)
		n |= doubleToInt(binary[i]) << i;
	return n;
}


/**
 * This program is an introduction to GPT (Generative Pre-trained Transformer) using a neural network.
 * Naive and first implementation without real research all thinking by myself.
 * 
 * Plan:
 * - Words are represented in a binary way as tokens (" " = 1, "bonjour" = 2, "salut" = 3, etc.) Binary up to 2^24 = 16777216 different tokens.
 * - The words are automatically assigned with a token in the file "token_dictionary.txt" (generated with the program "create_token_dictionary.c").
 * - The neural network is trained to predict the next word in a sentence.
 * - The sentence is represented as a sequence of 100 words (100 tokens) where the token 0 means no word or end of sentence.
 * 
 * Reality:
 * - The binary size of the tokens is dynamically calculated depending on the number of words in the dictionary.
 * - The number of tokens is not 100 but defined by the macro NB_WORDS (making it adjustable).
 * 
 * Observations:
 * - Below 20 context words, smaller the input layer is, worse the neural network is.
 * - MSE Error is stable at about 3 epochs.
 * - The more tokens there are, the less the neural network is good.
 * - 100 epochs & NB_WORDS = 50: 21.74% success rate	(600 seconds of training)
 * - 100 epochs & NB_WORDS = 250: ???
 * 
 * 
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'naive_gpt' program\n");
	atexit(exitProgram);

	// Load the token dictionary
	token_dictionary_t token_dictionary;
	int code = token_dict_load(&token_dictionary, DICTIONARY_TOKEN_PATH);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while loading the token dictionary\n");

	// Get the number of bits depending on the number of words
	int number_of_bits = 0;
	int nb_words = token_dictionary.size * 2;	// *2 to add a bit, to easily interpret new tokens
	while (nb_words > 0) {
		number_of_bits++;
		nb_words >>= 1;
	}
	INFO_PRINT("main(): Number of bits: %d\n", number_of_bits);

	// Macros
	#define NB_WORDS 250
	#define INPUT_SIZE (NB_WORDS * number_of_bits)

	// Create a neural network
	int nb_neurons_per_layer[] = {INPUT_SIZE, INPUT_SIZE / 2, INPUT_SIZE / 4, INPUT_SIZE / 8, number_of_bits};
	char *activation_functions[] = {NULL, "sigmoid", "sigmoid", "sigmoid", "sigmoid"};
	int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
	if (nb_layers != sizeof(activation_functions) / sizeof(char*)) ERROR_HANDLE_INT_RETURN_INT(-1, "main(): Error, the number of layers and the number of activation functions must be the same\n");
	NeuralNetwork network;
	code = initNeuralNetwork(&network, nb_layers, nb_neurons_per_layer, activation_functions, "MSE", 0.1, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while initializing the neural network\n");

	// Print the neural network information
	printNeuralNetwork(network);

	// Prepare sentences
	char **sentences;
	int nb_sentences;
	code = generateSentencesFromFolderForGPT("data/words/", &sentences, 100000, NB_WORDS + 1, &nb_sentences);
	INFO_PRINT("main(): %d sentences\n", nb_sentences);

	// Prepare the training data (output = last word of the sentence)
	nn_type **inputs;
	nn_type **expected;
	nn_type *inputs_flat_matrix = try2DFlatMatrixAllocation((void***)&inputs, nb_sentences, network.input_layer->nb_neurons, sizeof(nn_type), "main()");
	nn_type *outputs_flat_matrix = try2DFlatMatrixAllocation((void***)&expected, nb_sentences, network.output_layer->nb_neurons, sizeof(nn_type), "main()");
	for (int i = 0; i < nb_sentences; i++) {
		
		// Convert the sentence to a sequence of tokens
		int sentence_tokens[NB_WORDS];
		int nb_sentence_tokens = 0;
		convertSentenceToTokensArray(&token_dictionary, sentences[i], sentence_tokens, &nb_sentence_tokens);

		// Convert the sentence tokens to a binary array
		nn_type *sentence_binary = inputs[i];
		memset(sentence_binary, 0, INPUT_SIZE * sizeof(nn_type));
		for (int j = 0; j < nb_sentence_tokens - 1; j++) {
			convertXbIntToBinaryDoubleArray(sentence_tokens[j], sentence_binary, number_of_bits);
			sentence_binary += number_of_bits;
		}

		// Convert the expected sentence tokens to a binary array
		nn_type *expected_word_binary = expected[i];
		if (nb_sentence_tokens > 0)
			convertXbIntToBinaryDoubleArray(sentence_tokens[nb_sentence_tokens - 1], expected_word_binary, number_of_bits);
		else
			memset(expected_word_binary, 0, number_of_bits * sizeof(nn_type));
	}

	// Train the neural network
	#define NB_TEST_DATA_PERCENTAGE 20
	#define BATCH_SIZE 1
	#define NB_EPOCHS 100
	#define ERROR_TARGET 0.000001
	#define VERBOSE 1
	char buffer[16];
	ST_BENCHMARK_SOLO_COUNT(buffer, {
		code = TrainCPUSingleThread(&network, inputs, expected,
			nb_sentences,
			NB_TEST_DATA_PERCENTAGE,
			BATCH_SIZE,
			NB_EPOCHS,
			ERROR_TARGET,
			VERBOSE
		);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while training the neural network\n");
	}, "", 1, 1);
	INFO_PRINT("main(): Total training time: "STR_YELLOW_R("%s")"s\n", buffer);


	///// Test the neural network
	INFO_PRINT("main(): Testing the neural network\n");
	nn_type **test_inputs = inputs;
	nn_type **test_expected = expected;
	nn_type **test_outputs;
	nn_type *test_outputs_flat_matrix = try2DFlatMatrixAllocation((void***)&test_outputs, nb_sentences, network.output_layer->nb_neurons, sizeof(nn_type), "main()");
	FeedForwardBatchCPUSingleThread(&network, test_inputs, test_outputs, nb_sentences, 0);
	int nb_errors = 0;
	for (int i = 0; i < nb_sentences; i++) {

		// Convert the binary arrays to tokens
		int expected_token = convertBinaryDoubleArrayToXbInt(test_expected[i], number_of_bits);
		int output_token = convertBinaryDoubleArrayToXbInt(test_outputs[i], number_of_bits);
		
		// If there is no input token, continue
		if (convertBinaryDoubleArrayToXbInt(test_inputs[i], number_of_bits) == 0) continue;

		// If the output token is not the expected token, increment the number of errors
		if (expected_token != output_token)
			nb_errors++;
	}
	INFO_PRINT("main(): Success rate: %d/%d (%.2f%%)\n", nb_sentences - nb_errors, nb_sentences, (double)(nb_sentences - nb_errors) / nb_sentences * 100.0);
	free2DFlatMatrix((void**)test_outputs, test_outputs_flat_matrix, nb_sentences);

	///// Final part
	// Save the neural network
	INFO_PRINT("main(): Saving the neural network\n");
	code = saveNeuralNetwork(network, NEURAL_NETWORK_PATH, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while saving the neural network\n");

	// Free the neural network
	freeNeuralNetwork(&network);

	// Free the training data
	free2DFlatMatrix((void**)inputs, inputs_flat_matrix, nb_sentences);
	free2DFlatMatrix((void**)expected, outputs_flat_matrix, nb_sentences);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}
