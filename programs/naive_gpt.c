
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/list/token_dictionary.h"
#include "../src/st_benchmark.h"

#define DICTIONNARY_TOKEN_PATH "data/words_tokens.txt"

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

	// Load the token dictionary
	token_dictionary_t token_dictionary;
	int code = token_dict_load(&token_dictionary, DICTIONNARY_TOKEN_PATH);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while loading the token dictionary\n");

	// Get the number of bits depending on the number of words
	int number_of_bits = 0;
	int nb_words = token_dictionary.size;
	while (nb_words > 0) {
		number_of_bits++;
		nb_words >>= 1;
	}
	INFO_PRINT("main(): Number of bits: %d\n", number_of_bits);

	// Macros
	#define NB_WORDS 10
	#define INPUT_SIZE (NB_WORDS * number_of_bits)
	#define HIDDEN_LAYER_SIZE (int)(INPUT_SIZE * 0.5)

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
	// TODO: Add sentences dynamically
	char* sentences[] = {
		"bonjour comment ca va aujourd'hui",
		"salut comment ca va aujourd'hui",
		"bonjour comment ca va",
		"salut comment ca va",
		"bonjour comment ca",
		"salut comment ca",
		"bonjour comment",
		"salut comment",
		"bonjour",
		"salut"
	};
	int nb_sentences = sizeof(sentences) / sizeof(char*);
	INFO_PRINT("main(): %d sentences\n", nb_sentences);

	// Prepare the training data
	nn_type **inputs;
	nn_type **expected;
	nn_type *inputs_flat_matrix = try2DFlatMatrixAllocation((void***)&inputs, nb_sentences, network.input_layer->nb_neurons, sizeof(nn_type), "main()");
	nn_type *outputs_flat_matrix = try2DFlatMatrixAllocation((void***)&expected, nb_sentences, network.output_layer->nb_neurons, sizeof(nn_type), "main()");
	for (int i = 0; i < nb_sentences; i++) {
		
		// Convert the sentence to a sequence of tokens
		int sentence_tokens[NB_WORDS];
		int nb_sentence_tokens = 0;
		char* sentence = sentences[i];
		DEBUG_PRINT("main(): Sentence: '%s'\n", sentence);
		char* word = strtok(strdup(sentence), " ");
		while (word != NULL) {

			// Get the token id of the word
			token_t token;
			token.size = strlen(word);
			token.str = word;
			token_t *token_ptr = token_dict_search(token_dictionary, token);
			int token_id = token_ptr == NULL ? 0 : token_ptr->token_id;
			if (token_id == 0) WARNING_PRINT("main(): Unknown word: '%s'\n", word);

			// Add the token to the sentence tokens and get the next word
			sentence_tokens[nb_sentence_tokens++] = token_id;
			word = strtok(NULL, " ");
		}

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
	#define NB_EPOCHS 200
	#define ERROR_TARGET 0.000001
	#define VERBOSE 1
	INFO_PRINT("main(): Training the neural network\n");
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

		// Print the expected and output tokens
		if (expected_token != output_token) {
			ERROR_PRINT("main(): input tokens:");
			for (int j = 0; j < NB_WORDS; j++) {
				int token = convertBinaryDoubleArrayToXbInt(test_inputs[i] + j * number_of_bits, number_of_bits);
				PRINTER(" "STR_YELLOW_R("%d"), token);
				if (token == 0) break;
			}
			PRINTER(",\texpected token: "STR_YELLOW_R("%d")", output token: "STR_YELLOW_R("%d")"\n", expected_token, output_token);
			nb_errors++;
		}
		else {
			INFO_PRINT("main(): input tokens:");
			for (int j = 0; j < NB_WORDS; j++) {
				int token = convertBinaryDoubleArrayToXbInt(test_inputs[i] + j * number_of_bits, number_of_bits);
				PRINTER(" "STR_GREEN_R("%d"), token);
				if (token == 0) break;
			}
			PRINTER(",\texpected token: "STR_GREEN_R("%d")", output token: "STR_GREEN_R("%d")"\n", expected_token, output_token);
		}
	}
	INFO_PRINT("main(): Success rate: %d/%d (%.2f%%)\n", nb_sentences - nb_errors, nb_sentences, (double)(nb_sentences - nb_errors) / nb_sentences * 100.0);



	///// Final part
	// Save the neural network
	// INFO_PRINT("main(): Saving the neural network\n");
	// code = saveNeuralNetwork(network, "naive_gpt.nn", 1);
	// ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while saving the neural network\n");

	// Free the neural network
	freeNeuralNetwork(&network);

	// Free the training data
	free2DFlatMatrix((void**)inputs, inputs_flat_matrix, nb_sentences);
	free2DFlatMatrix((void**)expected, outputs_flat_matrix, nb_sentences);
	free2DFlatMatrix((void**)test_outputs, test_outputs_flat_matrix, nb_sentences);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

