
#include "training_utils.h"
#include "../universal_utils.h"


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

