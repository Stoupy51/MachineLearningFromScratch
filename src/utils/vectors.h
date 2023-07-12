
#ifndef __VECTORS_H__
#define __VECTORS_H__

#include <stddef.h>

// Utils functions
int* create_random_vector(int min, int max, int size, size_t vector_size_bytes);
void fill_random_vector(int* vector, int min, int max, int size);
void print_vector(int* vector, int size);

#endif

