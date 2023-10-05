
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

	// Create vocabulary from the training data (list of characters)
	int vocabulary_size = 0;
	char *vocabulary = generateCharVocabularyFromText(training_text, &vocabulary_size);
	INFO_PRINT("main(): %d characters in the vocabulary: %s\n", vocabulary_size, vocabulary);

	// Load the neural network from file
	NeuralNetwork network;
	int code = loadNeuralNetwork(&network, NEURAL_NETWORK_PATH);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while loading neural network from file '%s'\n", NEURAL_NETWORK_PATH);
	INFO_PRINT("main(): Neural network loaded from file '%s'\n", NEURAL_NETWORK_PATH);

	// Print the neural network information
	printNeuralNetwork(network);

	///// Use the neural network to generate text
	// Get the context size and the vocabulary size (input and output layer sizes)
	int context_size = network.input_layer->nb_neurons;
	vocabulary_size = network.output_layer->nb_neurons;

	// Prepare a pattern buffer (getting safe input from the user)
	char *pattern_buffer = calloc(context_size + 2, sizeof(char));
	sprintf(pattern_buffer, "%%.%ds", context_size);

	// Allocate memory for the context
	char *context = calloc(context_size + 1, sizeof(char));

	// Loop until the user wants to quit
	char loop = 1;
	while (loop != 'q') {

		// Ask the user to write a context
		INFO_PRINT("main(): Write a starting context of maximum %d characters (or 'q' to quit): ", context_size);
		scanf(pattern_buffer, context);
		if (context[0] == 'q' && context[1] == '\0') break;

		// Generate text until the neural network outputs a '\0' character
		#define generation_limit 512
		char generated_text[generation_limit] = "";
		int generated_text_index = 0;
		while (generated_text_index < generation_limit) {

			// Put the "context" into the neural network
			memset(network.input_layer->activations_values, 0, context_size * sizeof(nn_type));
			for (int i = 0; i < context_size; i++) {
				if (context[i] == '\0') break;
				network.input_layer->activations_values[i] = (nn_type)context[i];
			}

			// Compute the neural network
			FeedForwardCPUNoInput(&network);

			// Get the next character & stop if it is a '\0' character
			int predicted_index = getIndexOfMaxFromDoubleArray(network.output_layer->activations_values, vocabulary_size);
			if (predicted_index == 0) break;
			char next_char = vocabulary[predicted_index];

			// Add the next character to the generated text
			generated_text[generated_text_index++] = next_char;
			printf("%c", next_char);

			// Shift the context if needed
			int real_context_size = strlen(context);
			if (real_context_size < context_size) {
				context[real_context_size] = next_char;
				context[real_context_size + 1] = '\0';
			} else {
				for (int i = 0; i < context_size - 1; i++)
					context[i] = context[i + 1];
				context[context_size - 1] = next_char;
			}
		}
		(void)generated_text[0];
		printf("\n");
	}

	// Free the memory
	free(pattern_buffer);
	free(context);
	free(training_text);
	free(vocabulary);
	freeNeuralNetwork(&network);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

