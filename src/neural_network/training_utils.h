
#ifndef __TRAINING_UTILS_H__
#define __TRAINING_UTILS_H__

#include "neural_config.h"

#include <stdlib.h>

// Has to be a macro because of pointer arithmetic
#define shuffleTrainingData(inputs, targets, batch_size, type) { \
	type* new_inputs = mallocBlocking(batch_size * sizeof(type), "shuffleTrainingData()"); \
	type* new_targets = mallocBlocking(batch_size * sizeof(type), "shuffleTrainingData()"); \
	int new_size = 0; \
	int nb_samples = batch_size; \
	while (nb_samples > 0) { \
		int random_index = rand() % nb_samples; \
		new_inputs[new_size] = inputs[random_index]; \
		new_targets[new_size] = targets[random_index]; \
		new_size++; \
		inputs[random_index] = inputs[nb_samples - 1]; \
		targets[random_index] = targets[nb_samples - 1]; \
		nb_samples--; \
	} \
	memcpy(inputs, new_inputs, batch_size * sizeof(type)); \
	memcpy(targets, new_targets, batch_size * sizeof(type)); \
	free(new_inputs); \
	free(new_targets); \
}

int doubleToInt(nn_type d);
void convertXbIntToBinaryDoubleArray(int n, nn_type *binary, int number_of_bits);
int convertBinaryDoubleArrayToXbInt(nn_type *binary, int number_of_bits);

int getIndexOfMaxFromDoubleArray(nn_type *array, int array_size);
int* correspondanceArrayWithVocabularyIndex(char* vocabulary, int vocabulary_size, int verbose);

char** selectRandomChunksFromCharArray(char *array, int array_size, int number_of_chunks, int chunk_size);
int** selectRandomChunksFromIntArray(int *array, int array_size, int number_of_chunks, int chunk_size);

void replaceString(char *source, size_t source_size, char *old, char *new);

#endif

