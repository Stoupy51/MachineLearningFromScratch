
#include "text_file.h"
#include "../universal_utils.h"

/**
 * @brief Function to read a text file and return its content as a string.
 * 
 * @param file	The file to read.
 * 
 * @return char*	The content of the file as a string (automatically allocated).
 */
char *readTextFromFile(FILE *file) {

	// Get the size of the file
	size_t file_size = get_file_size(fileno(file));

	// Allocate the buffer
	char *buffer = mallocBlocking(file_size * sizeof(char), "readTextFromFile(buffer)");
	memset(buffer, 0, file_size * sizeof(char));

	// Read the file and return the buffer
	int code = fread(buffer, sizeof(char), file_size, file);
	WARNING_HANDLE_INT(code - 1, "readTextFromFile(): Error while reading the file\n");
	return buffer;
}

/**
 * @brief Function to read a text file and return its content as a string.
 * 
 * @param filename	The file to read.
 * 
 * @return char*	The content of the file as a string (automatically allocated).
 */
char *readTextFromTextFile(char *filename) {
	FILE *file = fopen(filename, "r");
	ERROR_HANDLE_PTR_RETURN_NULL(file, "readTextFromTextFile(): Error while opening the file '%s'\n", filename);
	char *buffer = readTextFromFile(file);
	fclose(file);
	return buffer;
}

/**
 * @brief Function to read all the text files in a folder and return their content as a string.
 * 
 * @param folder	The folder to read (must end with a '/').
 * 
 * @return char*	The content of the files as a string (automatically allocated).
 */
char *readTextFromFolder(char *folder) {

	// Get the list of files in the folder using a pipe with ls
	char command[512];
	sprintf(command, "ls %s", folder);
	FILE *pipe = popen(command, "r");
	ERROR_HANDLE_PTR_RETURN_NULL(pipe, "readTextFromFolder(): Error while opening the pipe for the path '%s'\n", folder);
	#ifdef _WIN32
		system("powershell -command \"\"");
	#endif

	// Read the files one by one
	char filename[512];
	char *buffer = NULL;
	size_t buffer_size = 0;
	while (fgets(filename, 512, pipe) != NULL) {

		// Remove the \n at the end of the filename
		int filename_length = strlen(filename);
		if (filename[filename_length - 1] == '\n') filename[filename_length - 1] = '\0';

		// Add the folder to the filename
		char full_filename[1024];
		sprintf(full_filename, "%s%s", folder, filename);

		// Read the file
		char *file_buffer = readTextFromTextFile(full_filename);
		if (file_buffer == NULL) {
			ERROR_PRINT("readTextFromFolder(): Error while reading the file '%s'\n", full_filename);
			continue;
		}

		// Add the file content to the buffer (realloc, memcpy and free)
		size_t file_buffer_size = strlen(file_buffer);
		buffer = reallocBlocking(buffer, (buffer_size + file_buffer_size + 1) * sizeof(char), "readTextFromFolder(buffer)");
		memcpy(buffer + buffer_size, file_buffer, file_buffer_size * sizeof(char));
		buffer_size += file_buffer_size;
		buffer[buffer_size] = '\0';
		free(file_buffer);
	}

	// Close the pipe
	int code = pclose(pipe);
	ERROR_HANDLE_INT_RETURN_NULL(code, "readTextFromFolder(): Error while closing the pipe for the path '%s'\n", folder);

	// Return the buffer
	return buffer;
}


// Private function to compare two characters
int compareChar(const void *a, const void *b) {
	return *(char*)a - *(char*)b;
}

/**
 * @brief Function to generate a char vocabulary from a text.
 * 
 * @param text				The text to read.
 * @param vocabulary_size	[out] The size of the vocabulary generated.
 * 
 * @return char*			The sorted vocabulary generated (automatically allocated), ex: ['a', 'b', 'c', ...]
 */
char *generateCharVocabularyFromText(const char *text, int *vocabulary_size) {

	// Initialize the vocabulary (257 instead of 256 to add another '\0' at the end)
	char *vocabulary = mallocBlocking(257 * sizeof(char), "main(vocabulary)");
	memset(vocabulary, '\0', 257 * sizeof(char));
	
	// Create vocabulary from the text
	*vocabulary_size = 1;
	int text_size = strlen(text);
	for (int i = 0; i < text_size; i++) {

		// Add the character to the vocabulary if it is not already in it
		char c = text[i];
		if (strchr(vocabulary + 1, c) == NULL && c != '\0') {
			vocabulary[*vocabulary_size] = c;
			(*vocabulary_size)++;
		}
	}

	// Sort the vocabulary
	qsort(vocabulary + 1, *vocabulary_size - 1, sizeof(char), compareChar);

	// Return the vocabulary
	return vocabulary;
}



/**
 * @brief Function to generate sentences from a text file.
 * The sentences can be used to train a neural network such as GPT.
 * 
 * @param file						The file to read.
 * @param sentences					[out] The sentences generated (automatically allocated)
 * @param max_sentences				The maximum number of sentences to generate.
 * @param max_words_per_sentence	The maximum number of words in a sentence.
 * @param total_sentences			[out] Pointer to the total number of sentences generated.
 * 
 * @return int						0 if no error, -1 otherwise.
 */
int generateSentencesFromFileForGPT(FILE *file, char ***sentences, int max_sentences, int max_words_per_sentence, int *total_sentences) {

	// Allocate the sentences array
	*sentences = mallocBlocking(max_sentences * sizeof(char*), "generateSentencesFromFileForGPT(sentences)");
	
	// Prepare the sentence buffer
	int max_sentence_length = max_words_per_sentence * 100;
	char *sentence = mallocBlocking(max_sentence_length * sizeof(char), "generateSentencesFromFileForGPT(buffer 1)");
	memset(sentence, 0, max_sentence_length * sizeof(char));
	int sentence_length = 0;
	int nb_words_in_sentence = 0;

	// Read the file until the end or the maximum number of sentences
	int sentence_index = 0;
	int current_alloc_size = max_sentence_length;
	char c;
	while (sentence_index < max_sentences && !feof(file)) {

		// Read the next character
		c = fgetc(file);

		// If the sentence_length is too big, reallocate the sentence buffer
		if (sentence_length >= current_alloc_size - 1) {
			current_alloc_size *= 2;
			sentence = reallocBlocking(sentence, current_alloc_size * sizeof(char), "generateSentencesFromFileForGPT(buffer 2)");
			memset(sentence + sentence_length, 0, (current_alloc_size - sentence_length) * sizeof(char));
		}

		// If the character is a new line or the sentence has too many words,
		if (c == '\n' || nb_words_in_sentence >= max_words_per_sentence) {

			// If the sentence is not empty
			if (sentence_length > 0) {

				// Add the sentence to the sentences array
				(*sentences)[sentence_index] = sentence;
				sentence_index++;

				// Allocate a new sentence buffer
				current_alloc_size = max_sentence_length;
				sentence = mallocBlocking(current_alloc_size * sizeof(char), "generateSentencesFromFileForGPT(buffer 3)");
				memset(sentence, 0, current_alloc_size * sizeof(char));
				sentence_length = 0;
				nb_words_in_sentence = 0;

				// If the character is not a new line, add it to the new sentence buffer
				if (c != '\n') sentence[sentence_length++] = c;
			}
		}

		// Else, add the character to the sentence buffer
		else {
			sentence[sentence_length++] = c;
			if (c == ' ') nb_words_in_sentence++;
		}
	}

	// Free the last sentence buffer
	free(sentence);

	// Set the total number of sentences generated
	*total_sentences = sentence_index;

	// Return success
	return 0;
}


/**
 * @brief Function to generate sentences from a text file.
 * The sentences can be used to train a neural network such as GPT.
 * 
 * @param filename					The file to read.
 * @param sentences					[out] The sentences generated (automatically allocated)
 * @param max_sentences				The maximum number of sentences to generate.
 * @param max_words_per_sentence	The maximum number of words in a sentence.
 * @param total_sentences			[out] Pointer to the total number of sentences generated.
 * 
 * @return int						0 if no error, -1 otherwise.
 */
int generateSentencesFromTextFileForGPT(char *filename, char ***sentences, int max_sentences, int max_words_per_sentence, int *total_sentences) {
	FILE *file = fopen(filename, "r");
	ERROR_HANDLE_PTR_RETURN_INT(file, "generateSentencesFromTextFileForGPT(): Error while opening the file '%s'\n", filename);
	int code = generateSentencesFromFileForGPT(file, sentences, max_sentences, max_words_per_sentence, total_sentences);
	fclose(file);
	return code;
}


/**
 * @brief Function to generate sentences from a folder by reading all the files in the folder.
 * The sentences can be used to train a neural network such as GPT.
 * 
 * @param folder					The folder to read (must end with a '/')
 * @param sentences					[out] The sentences generated (automatically allocated)
 * @param max_sentences				The maximum number of sentences to generate.
 * @param max_words_per_sentence	The maximum number of words in a sentence.
 * @param total_sentences			[out] Pointer to the total number of sentences generated.
 * 
 * @return int						0 if at least one file was read, -1 otherwise.
*/
int generateSentencesFromFolderForGPT(char *folder, char ***sentences, int max_sentences, int max_words_per_sentence, int *total_sentences) {

	// Get the list of files in the folder using a pipe with ls
	char command[512];
	sprintf(command, "ls %s", folder);
	FILE *pipe = popen(command, "r");
	ERROR_HANDLE_PTR_RETURN_INT(pipe, "generateSentencesFromFolderForGPT(): Error while opening the pipe for the path '%s'\n", folder);
	#ifdef _WIN32
		system("powershell -command \"\"");
	#endif

	// Read the files one by one
	char filename[512];
	*total_sentences = 0;
	*sentences = NULL;
	while ((*total_sentences) < max_sentences && fgets(filename, 512, pipe) != NULL) {

		// Remove the \n at the end of the filename
		int filename_length = strlen(filename);
		if (filename[filename_length - 1] == '\n') filename[filename_length - 1] = '\0';

		// Add the folder to the filename
		char full_filename[1024];
		sprintf(full_filename, "%s%s", folder, filename);

		// Generate the sentences from the file
		int nb_sentences;
		char **local_sentences;
		int code = generateSentencesFromTextFileForGPT(full_filename, &local_sentences, max_sentences - (*total_sentences), max_words_per_sentence, &nb_sentences);
		if (code < 0) {
			ERROR_PRINT("generateSentencesFromFolderForGPT(): Error while generating sentences from the file '%s'\n", full_filename);
			continue;
		}

		// Add the sentences to the sentences array (realloc, memcpy and free)
		*sentences = reallocBlocking(*sentences, ((*total_sentences) + nb_sentences) * sizeof(char*), "generateSentencesFromFolderForGPT(sentences)");
		memcpy(*sentences + (*total_sentences), local_sentences, nb_sentences * sizeof(char*));
		*total_sentences += nb_sentences;
		free(local_sentences);
	}

	// Close the pipe
	int code = pclose(pipe);
	ERROR_HANDLE_INT_RETURN_INT(code, "generateSentencesFromFolderForGPT(): Error while closing the pipe for the path '%s'\n", folder);

	// Return success if at least one file was read
	return (*total_sentences) > 0 ? 0 : -1;
}

