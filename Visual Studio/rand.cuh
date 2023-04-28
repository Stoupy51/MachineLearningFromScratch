
#pragma once

#include "includes.cuh"

#define SIZE 1000000000
#define NUMBER_OF_THREADS_PER_BLOCK 512
#define NUMBER_OF_BLOCKS SIZE / NUMBER_OF_THREADS_PER_BLOCK
#define DEBUG_MODE 0

__global__ void gpu_routine(int* device_vector, int size);
int randMain();

