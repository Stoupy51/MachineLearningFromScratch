
#include "../src/universal_utils.h"
#include "../src/list/token_dictionary.h"
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

#define WORDS_FOLDER_PATH "data/words"
#define WORDS_TOKENS_FILE_PATH "data/words_tokens.txt"

/**
 * This program create a token dictionary for the GPT (Generative Pre-trained Transformer)
 * by assigning a token to each word in the files of the folder "words".
 * 
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'create_token_dictionary' program\n");
	atexit(exitProgram);

	// Create a token dictionary
	token_dictionary_t token_dictionary;
	token_dict_init(&token_dictionary, 100000);
	INFO_PRINT("main(): Token dictionary created with memory size: %.2f MB\n", (double)(sizeof(token_dictionary_t) + token_dictionary.hash_table_size * sizeof(token_list_t)) / 1024.0 / 1024.0);

	// For each file in the folder "words"
	FILE *pipe = popen("ls " WORDS_FOLDER_PATH, "r");
	ERROR_HANDLE_PTR_RETURN_INT(pipe, "main(): Can't open pipe to list files in folder '%s'\n", WORDS_FOLDER_PATH);
	#ifdef _WIN32
		system("powershell -command \"\"");
	#endif

	// Measure time of the program
	char buffer[100];
	ST_BENCHMARK_SOLO_COUNT(buffer, {

	char filename[256];
	while (fgets(filename, sizeof(filename), pipe) != NULL) {

		// Remove the \n at the end of the file_path
		filename[strlen(filename) - 1] = '\0';

		// Create the file path
		char real_filepath[256];
		sprintf(real_filepath, "%s/%s", WORDS_FOLDER_PATH, filename);
		DEBUG_PRINT("main(): Loading file: '%s'\n", real_filepath);

		// Open the file
		FILE *file = fopen(real_filepath, "r");
		if (file == NULL) {
			ERROR_PRINT("main(): Can't open file '%s', skipping.\n", real_filepath);
			continue;
		}

		// For each word in the file
		char word[256];
		while (fscanf(file, "%255s", word) != EOF) {

			// Create the token
			token_t token;
			token.size = strlen(word);
			token.str = mallocBlocking((token.size + 1) * sizeof(char), "main()");
			token.str[token.size] = '\0';
			memcpy(token.str, word, token.size);

			// Add the token to the token dictionary if it doesn't exist
			if (token_dict_search(token_dictionary, token) == NULL)
				token_dict_add(&token_dictionary, token);
		}

		// Close the file
		int code = fclose(file);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Can't close file '%s'\n", real_filepath);
	}
	INFO_PRINT("main(): Token dictionary filled\n");

	// Close the pipe
	ERROR_HANDLE_INT_RETURN_INT(pclose(pipe), "main(): Can't close pipe to list files in folder '%s'\n", WORDS_FOLDER_PATH);

	// Print the token dictionary
	token_dict_print(token_dictionary, 10);

	// Save the token dictionary and free it
	int code = token_dict_save(&token_dictionary, WORDS_TOKENS_FILE_PATH);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Can't save token dictionary to file '%s'\n", WORDS_TOKENS_FILE_PATH);
	token_dict_free(&token_dictionary);

	}, "create_token_dictionary main()", 1, 0);
	PRINTER(buffer);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

