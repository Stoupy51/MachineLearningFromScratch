
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 100000
#define NUMBER_OF_THREADS 512

__global__ void gpu_routine(int* device_vector, long* thread_pipe, int size) {

	// Get the starting index of the thread in the vector
	int start = threadIdx.x * (size / NUMBER_OF_THREADS);

	// Get the ending index of the thread in the vector
	int end = (threadIdx.x + 1) * (size / NUMBER_OF_THREADS);
	printf("[GPU-Thread #%d] Start from %d t %d\n", threadIdx.x, start, end);

	// Compute the sum of the elements in the vector
	long sum = 0;
	for (int i = start; i < end; i++) {
		for (int j = i; j < end; j++)
			sum += device_vector[i] + j;
	}
	printf("[GPU-Thread #%d] Calculated sum: %d\n", threadIdx.x, sum);

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

	// Create a vector pipe to store the threads return values
	long* host_thread_pipes = (long*)malloc(SIZE * sizeof(long));
	long* device_thread_pipes;
	cudaMalloc((void**)&device_thread_pipes, NUMBER_OF_THREADS * sizeof(long));

	// Vector of integers
	int* host_vector = (int*)malloc(SIZE * sizeof(int));
	for (i = 0; i < SIZE; i++)
		host_vector[i] = rand() % 10;

	// Create the device vector and copy the host vector into it
	int* device_vector;
	cudaMalloc((void**)&device_vector, SIZE * sizeof(int));
	cudaMemcpy(device_vector, host_vector, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the kernel
	gpu_routine << < 1, NUMBER_OF_THREADS >> > (device_vector, device_thread_pipes, SIZE);

	// Wait for the kernel to finish
	cudaDeviceSynchronize();

	// Copy the thread pipes into the host vector
	cudaMemcpy(host_thread_pipes, device_thread_pipes, NUMBER_OF_THREADS * sizeof(long), cudaMemcpyDeviceToHost);

	// Compute the sum of the thread pipes
	long sum = 0;
	for (i = 0; i < NUMBER_OF_THREADS; i++)
		sum += host_thread_pipes[i];

	// Print the result
	printf("Sum of the vector: %ld\n", sum);

	// Free the memory
	cudaFree(device_thread_pipes);
	cudaFree(device_vector);
	free(host_thread_pipes);
	free(host_vector);

	return 0;
}

