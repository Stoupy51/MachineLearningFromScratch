
#include "sum.cuh"

/**
 * @brief Function that is executed in the GPU.
 * It computes the sum of the elements in the vector.
 * The vector is divided in NUMBER_OF_THREADS parts and
 * each thread computes the sum of its part.
 * 
 * @param device_vector The vector of integers
 * @param thread_pipe The thread pipe to store the results
 * @param size The size of the vector
 * 
 * @return void
 */
__global__ void gpu_routine(int* device_vector, long* thread_pipe, int size) {

	// Get the starting index of the thread in the vector
	int start = threadIdx.x * (size / NUMBER_OF_THREADS);

	// Get the ending index of the thread in the vector
	int end = (threadIdx.x + 1) * (size / NUMBER_OF_THREADS);
#if DEBUG_MODE == 1
	printf("[GPU-Thread #%d] Start from %d t %d\n", threadIdx.x, start, end);
#endif

	// Compute the sum of the elements in the vector
	long sum = 0;
	for (int i = start; i < end; i++) {
		for (int j = i; j < end; j++)
			sum += device_vector[i] + j;
	}


#if DEBUG_MODE == 1
	printf("[GPU-Thread #%d] Calculated sum: %d\n", threadIdx.x, sum);
#endif

	// Store the sum in the thread pipe
	thread_pipe[threadIdx.x] = sum;
}


/**
 * @brief Function that is executed in the CPU.
 * It creates a vector of integers and launches the kernel.
 * It waits for the kernel to finish and then computes the
 * sum of the thread pipes.
*/
int sumMain() {

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
	gpu_routine<<<1,NUMBER_OF_THREADS>>>(device_vector, device_thread_pipes, SIZE);

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

