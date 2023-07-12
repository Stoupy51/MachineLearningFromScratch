
#include <stdio.h>
#include <stdlib.h>

#include "vectors.h"
#include "../universal_utils.h"

/**
 * @brief Create a random vector of integers.
 * size or vector_size_bytes must be set but not both.
 * 
 * @param min Minimum value of the integers
 * @param max Maximum value of the integers
 * @param size Size of the vector
 * @param vector_size_bytes Size of the vector in bytes
 * 
 * @return int* Pointer to the vector
 */
int* create_random_vector(int min, int max, int size, size_t vector_size_bytes) {
	
	// Allocate memory for the vector
	int* vector = malloc(vector_size_bytes);
	ERROR_HANDLE_PTR_RETURN_NULL(vector, "create_random_vector(): Error allocating memory for the vector.\n");

	// Fill the vector with random integers
	for (size--; size >= 0; size--)
		vector[size] = rand() % max + min;

	// Return the vector
	return vector;
}

/**
 * @brief Fill a vector with random integers.
 * 
 * @param vector Pointer to the vector
 * @param min Minimum value of the integers
 * @param max Maximum value of the integers
 * @param size Size of the vector
 * 
 * @return void
 */
void fill_random_vector(int* vector, int min, int max, int size) {
	for (size--; size >= 0; size--)
		vector[size] = rand() % max + min;
}

/**
 * @brief Print a vector of integers.
 * (No line break at the end: [1, 2, 3])
 * 
 * @param vector Pointer to the vector
 * @param size Size of the vector
 * 
 * @return void
 */
void print_vector(int* vector, int size) {

	// Print [] if the vector is empty
	if (size == 0) {
		printf("[]");
		return;
	}

	// Print [ and the first element
	printf("[%d", vector[0]);

	// Print the rest of the elements
	int i;
	for (i = 1; i < size; i++) 
		printf(", %d", vector[i]);

	// Print ]
	printf("]");
}


