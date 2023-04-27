
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void gpu_routine(int *device_vector, int *thread_pipe, int size) {
	
	// Get the starting index of the thread in the vector
	int start = threadIdx.x * size / blockDim.x;

	// Get the ending index of the thread in the vector
	int end = (threadIdx.x + 1) * size / blockDim.x;

	// Compute the sum of the elements in the vector
	long sum = 0;
	for (int i = start; i < end; i++)
		sum += device_vector[i];
	
	// Store the sum in the thread pipe
	thread_pipe[threadIdx.x] = sum;
}

/**
 * This program is an introduction test to CUDA programming.
 * It is a simple program that manipulates a vector of integers.
 * The vector is initialized with random values and then the
 * program computes the sum of all the elements in the vector.
 * The program is executed in the GPU.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
*/
int main() {

	// Variables
	int i;

	// Size of the vector
	#define SIZE 10000000

	// Get number of GPU threads available (for example GTX 1060 has 1280)
	int number_of_gpu_threads;
	cudaDeviceGetAttribute(&number_of_gpu_threads, cudaDevAttrMultiProcessorCount, 0);
	printf("Number of GPU threads available: %d\n", number_of_gpu_threads);

	return 0;
}

