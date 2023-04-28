
#pragma once

#include "includes.cuh"

#define SIZE 1000000
#define NUMBER_OF_THREADS 512

__global__ void gpu_routine(int* device_vector, long* thread_pipe, int size);
int sumMain();


