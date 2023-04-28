
#include "rand.cuh"

/**
 * @brief GPU routine that fills a vector with random numbers
 * 
 * @param device_vector Vector to be filled
 * @param size Size of the vector
 * 
 * @return void
*/
__global__ void gpu_routine(int* device_vector, int size) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < size) {
		device_vector[tid] = tid;
	}
}

/**
 * @brief Function that calls the GPU routine to fill a vector with random numbers
*/
int randMain() {

	// Message
	INFO_PRINT("randMain() : Starting random number generation on GPU...\n");

	// Take current time in us
	// TODO: Use CUDA events instead
	struct timespec start;
	gettimeofday(&start, NULL);
	



	// Host variables
	int* host_vector = (int*)malloc(SIZE * sizeof(int));

	// Device variables
	int* device_vector;
	cudaMalloc((void**)&device_vector, SIZE * sizeof(int));

	// Initialize host vector and pipe
	memset(host_vector, 0, SIZE * sizeof(int));

	// Copy host vector to device vector
	cudaMemcpy(device_vector, host_vector, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Launch kernel on GPU
	gpu_routine<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>> (device_vector, SIZE);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Copy device vector to host vector
	cudaMemcpy(host_vector, device_vector, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(device_vector);
	free(host_vector);

	return 0;
}

