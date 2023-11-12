
#include "training_utils.h"
#include "../universal_utils.h"

/**
 * @brief Utility function to shuffle the training data
 * 
 * @param inputs		Pointer to the inputs array
 * @param targets		Pointer to the target outputs array
 * @param batch_size	Number of samples in the batch
 */
void shuffleTrainingData(void **inputs, void **targets, int batch_size) {

	// Prepare a new array of pointers to the inputs and the target outputs
	void **new_inputs = mallocBlocking(batch_size * sizeof(void *), "shuffleTrainingData()");
	void **new_targets = mallocBlocking(batch_size * sizeof(void *), "shuffleTrainingData()");
	int new_size = 0;

	// While there are samples in the batch,
	int nb_samples = batch_size;
	while (nb_samples > 0) {

		// Select a random sample
		int random_index = rand() % nb_samples;

		// Add the random sample to the new array
		new_inputs[new_size] = inputs[random_index];
		new_targets[new_size] = targets[random_index];
		new_size++;

		// Remove the random sample from the old array by replacing it with the last sample
		inputs[random_index] = inputs[nb_samples - 1];
		targets[random_index] = targets[nb_samples - 1];
		nb_samples--;
	}

	// Copy the new array to the old array
	memcpy(inputs, new_inputs, batch_size * sizeof(void *));
	memcpy(targets, new_targets, batch_size * sizeof(void *));

	// Free the new array
	free(new_inputs);
	free(new_targets);
}


/**
 * @brief Convert a double to an int (rounding to the nearest integer)
 * 
 * @param d		The double to convert
 * 
 * @return int	The converted int
 */
int doubleToInt(nn_type d) {
	return (int)(d + 0.5);
}

/**
 * @brief Convert an int to a binary array of double
 * 
 * @param n					The int to convert
 * @param binary			The binary array of double to fill
 * @param number_of_bits	The number of bits to convert
 */
void convertXbIntToBinaryDoubleArray(int n, nn_type *binary, int number_of_bits) {
	for (int i = 0; i < number_of_bits; i++)
		binary[i] = (n & (1 << i)) ? 1.0 : 0.0;
}

/**
 * @brief Convert a binary array of double to an int
 * 
 * @param binary			The binary array of double to convert
 * @param number_of_bits	The number of bits to convert
 * 
 * @return int				The converted int
 */
int convertBinaryDoubleArrayToXbInt(nn_type *binary, int number_of_bits) {
	int n = 0;
	for (int i = 0; i < number_of_bits; i++)
		n |= doubleToInt(binary[i]) << i;
	return n;
}

/**
 * @brief Get the index of the maximum value in a double array
 * 
 * @param array			The array to search in
 * @param array_size	The size of the array
 * 
 * @return int			The index of the maximum value
 */
int getIndexOfMaxFromDoubleArray(nn_type *array, int array_size) {
	int index = 0;
	for (int i = 1; i < array_size; i++)
		if (array[i] > array[index])
			index = i;
	return index;
}


/**
 * @brief Create an array of correspondance between a vocabulary and its indexes
 * 
 * @param vocabulary		The vocabulary to create the correspondance array from
 * @param vocabulary_size	The size of the vocabulary
 * @param verbose			Whether to print debug messages or not
 * 
 * @return int*				256 indexes corresponding to the vocabulary indexes
 */
int* correspondanceArrayWithVocabularyIndex(char* vocabulary, int vocabulary_size, int verbose) {
	if (verbose)
		DEBUG_PRINT("correspondanceArrayWithVocabularyIndex(): Creating correspondance array for vocabulary of size %d\n", vocabulary_size);

	// Allocate the array
	int *array = mallocBlocking(sizeof(int) * 256, "correspondanceArrayWithVocabularyIndex()");
	memset(array, 0, sizeof(int) * 256);

	// For each character in the vocabulary, set the index of the character in the array
	for (int i = 0; i < vocabulary_size; i++) {
		unsigned char c = vocabulary[i]; // Cast to unsigned char to avoid negative indexes
		array[(int)c] = i;
		if (verbose)
			DEBUG_PRINT("%d == '%c' -> %d\n", (int)c, c, i);
	}
	return array;
}



/**
 * @brief Links an array of random character chunks of a given size from a given array
 * 
 * @param array				The array to select chunks from
 * @param array_size		The size of the array
 * @param number_of_chunks	The number of chunks to select
 * @param chunk_size		The size of the chunks
 * 
 * @return char**			The array of linked chunks (only free the array, not the chunks)
 */
char** selectRandomChunksFromCharArray(char *array, int array_size, int number_of_chunks, int chunk_size) {

	// Allocate the array of chunks
	char **chunks = mallocBlocking(sizeof(char*) * number_of_chunks, "selectRandomChunksFromCharArray()");
	int current_nb_chunks = 0;

	// Select random chunks from the array
	while (current_nb_chunks < number_of_chunks) {

		// Select a random index in the array, but continue if the chunk is too close to the end of the array
		int random_index = rand() % array_size;
		if (random_index + chunk_size >= (array_size - 1)) continue;

		// Link the chunk to the array of chunks
		chunks[current_nb_chunks++] = array + random_index;
	}

	// Return the array of chunks
	return chunks;
}

/**
 * @brief Links an array of random integer chunks of a given size from a given array
 * 
 * @param array				The array to select chunks from
 * @param array_size		The size of the array
 * @param number_of_chunks	The number of chunks to select
 * @param chunk_size		The size of the chunks
 * 
 * @return int**			The array of linked chunks (only free the array, not the chunks)
 */
int** selectRandomChunksFromIntArray(int *array, int array_size, int number_of_chunks, int chunk_size) {

	// Allocate the array of chunks
	int **chunks = mallocBlocking(sizeof(int*) * number_of_chunks, "selectRandomChunksFromIntArray()");
	int current_nb_chunks = 0;

	// Select random chunks from the array
	while (current_nb_chunks < number_of_chunks) {

		// Select a random index in the array, but continue if the chunk is too close to the end of the array
		int random_index = rand() % array_size;
		if (random_index + chunk_size >= (array_size - 1)) continue;

		// Link the chunk to the array of chunks
		chunks[current_nb_chunks++] = array + random_index;
	}

	// Return the array of chunks
	return chunks;
}


/**
 * @brief Replace a string in a source string
 * 
 * @param source		The source string
 * @param source_size	The size of the source string
 * @param old			The string to replace
 * @param new			The string to replace with
 * 
 * @throw EXIT_FAILURE	If the source string is too small
 */
void replaceString(char *source, size_t source_size, char *old, char *new) {
	
	// Get the size of the old and new strings
	size_t old_size = strlen(old);
	size_t new_size = strlen(new);
	size_t difference = new_size - old_size;
	size_t source_length = strlen(source);

	// While the old string is found in the source string
	char *old_string_position = strstr(source, old);
	while (old_string_position != NULL) {

		// If the source string is too small, exit
		if (source_length + difference >= source_size) {
			ERROR_PRINT("replaceString(): Source string is too small\n");
			exit(EXIT_FAILURE);
		}

		// Replace the old string with the new string
		memmove(old_string_position + new_size, old_string_position + old_size, strlen(old_string_position + old_size) + 1);
		memcpy(old_string_position, new, new_size);

		// Update the source length
		source_length += difference;

		// Search for the old string again
		old_string_position = strstr(old_string_position + new_size, old);
	}
}

