
#include "../src/universal_utils.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/list/token_dictionary.h"
#include "../src/utils/text_file.h"

#define DICTIONARY_TOKEN_PATH "data/token_dictionary.bin"
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
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'gpt_train' program\n");
	atexit(exitProgram);

	// Load the token dictionary
	INFO_PRINT("main(): Loading token dictionary\n");
	token_dictionary_t token_dict;
	int code = token_dict_load(&token_dict, DICTIONARY_TOKEN_PATH);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while loading token dictionary '%s'\n", DICTIONARY_TOKEN_PATH);

	// Load the training data into one big string
	INFO_PRINT("main(): Loading training data\n");
	char *training_data = readTextFromFolder(TRAINING_FOLDER_PATH);
	ERROR_HANDLE_PTR_RETURN_INT(training_data, "main(): Error while loading training data '%s'\n", TRAINING_FOLDER_PATH);
	
	// Convert the training data into a list of tokens
	INFO_PRINT("main(): Converting training data into tokens\n");
	int nb_tokens = 0;
	int *tokens = token_dict_convert_text_to_token_list(&token_dict, training_data, &nb_tokens);
	INFO_PRINT("main(): Number of tokens in the training data: %d [%d, %d, ...]\n", nb_tokens, tokens[0], tokens[1]);


	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

