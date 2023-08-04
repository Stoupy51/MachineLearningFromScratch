
#include "text_file.h"
#include "../universal_utils.h"

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


