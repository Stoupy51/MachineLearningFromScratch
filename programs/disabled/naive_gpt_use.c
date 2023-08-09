
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/list/token_dictionary.h"
#include "../src/utils/text_file.h"
#include "../src/st_benchmark.h"

#define DICTIONARY_TOKEN_PATH "data/token_dictionary.txt"
#define NEURAL_NETWORK_PATH "bin/naive_gpt.nn"
#define OUTPUT_FILE_PATH "data/naive_gpt_output.txt"

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

	// Load the neural network
	NeuralNetwork network;
	code = loadNeuralNetwork(&network, NEURAL_NETWORK_PATH);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while loading the neural network\n");
	int number_of_bits = network.output_layer->nb_neurons;

	// Print the neural network
	printNeuralNetwork(network);

	// Open the output file
	FILE *output_file = fopen(OUTPUT_FILE_PATH, "w");
	ERROR_HANDLE_PTR_RETURN_INT(output_file, "main(): Error while opening the output file\n");

	// While the user wants to continue
	while (1) {

		// Ask the user for a sentence
		char sentence[512];
		printf("Enter a sentence (or \"quit\"): ");
		scanf("%[^\n]%*c", sentence);
		sentence[511] = '\0';

		// If the user wants to quit
		if (strcmp(sentence, "quit") == 0)
			break;
		
		// Write the sentence in the output file
		fprintf(output_file, "\nUtilisateur: %s\n", sentence);

		// For each word in the sentence, convert it to a token
		memset(network.input_layer->activations_values, 0, network.input_layer->nb_neurons * sizeof(nn_type));
		char *word = strtok(sentence, " ");
		int i = 0;
		while (word != NULL) {

			// Get the token of the word
			token_t token;
			token.size = strlen(word);
			token.str = word;
			token_t *found_token = token_dict_search(token_dictionary, token);

			// If the token is not found, fill the input with 0
			nn_type *input = &network.input_layer->activations_values[i * number_of_bits];
			if (found_token == NULL)
				memset(input, 0, number_of_bits * sizeof(nn_type));
			else
				convertXbIntToBinaryDoubleArray(found_token->token_id, input, number_of_bits);
			
			// Next word
			word = strtok(NULL, " ");
			i++;
		}

		// Feed forward
		FeedForwardCPU(&network, network.input_layer->activations_values);

		// While the output is filled with 0, feed forward again
		INFO_PRINT("main(): Output:\n");
		fprintf(output_file, "\nStoup-GPT:\n");
		while (1) {

			// Print the word associated with the output
			int output_token = convertBinaryDoubleArrayToXbInt(network.output_layer->activations_values, number_of_bits);
			token_t *found_token = token_dict_search_id(token_dictionary, output_token);
			if (found_token == NULL) {
				PRINTER("\n\nFinished with output_token = %d\n", output_token);
				fprintf(output_file, "\n\nFinished with output_token = %d\n", output_token);
				break;
			}
			else {
				PRINTER("%s ", found_token->str);
				fprintf(output_file, "%s ", found_token->str);
			}
			
			///// Feed forward again with the output
			// Offset the input
			memmove(network.input_layer->activations_values + number_of_bits, network.input_layer->activations_values, (network.input_layer->nb_neurons - number_of_bits) * sizeof(nn_type));

			// Copy the output to the input
			memcpy(network.input_layer->activations_values, network.output_layer->activations_values, number_of_bits * sizeof(nn_type));

			// Feed forward
			FeedForwardCPU(&network, network.input_layer->activations_values);
		}
	}

	// Close the output file
	fclose(output_file);

	// Free the neural network
	freeNeuralNetwork(&network);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

