
#ifndef __INTEGER_UTILS_H__
#define __INTEGER_UTILS_H__

#include "neural_config.h"

int doubleToInt(nn_type d);
void convertXbIntToBinaryDoubleArray(int n, nn_type *binary, int number_of_bits);
int convertBinaryDoubleArrayToXbInt(nn_type *binary, int number_of_bits);

int getIndexOfMaxFromDoubleArray(nn_type *array, int array_size);
int* correspondanceArrayWithVocabularyIndex(char* vocabulary, int vocabulary_size);

char** selectRandomChunksFromCharArray(char *array, int array_size, int number_of_chunks, int chunk_size);
int** selectRandomChunksFromIntArray(int *array, int array_size, int number_of_chunks, int chunk_size);

#endif

