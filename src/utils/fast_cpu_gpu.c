
#include "fast_cpu_gpu.h"

#include "../universal_utils.h"
#include "../gpu/gpu_utils.h"
#include "../universal_pthread.h"


// Arbitrary values to determine if the GPU or the CPU should be used
// The values are based on my own tests, and may not be optimal for your own hardware
#define MIN_SIZE_FOR_GPU 100000
#define MIN_SIZE_FOR_CPU_THREADS 1000

/**
 * @brief Thread function to fill an array of double with random values
 * between min and max, using all the cores of the CPU.
 * 
 * @param args			The arguments of the thread
 * 
 * @return 0
 */
thread_return_type fillRandomDoubleArrayCPUThreadsThread(thread_param_type args) {

	// Get the arguments
	double* array = ((void**)args)[0];
	int size = ((int*)args)[1];
	double min = ((double*)args)[2];
	double max_minus_min = ((double*)args)[3] - min;
	int nb_cores = ((int*)args)[4];
	int thread_id = ((int*)args)[5];

	// Calculate the part of the array to fill
	int start = thread_id * size / nb_cores;
	int end = (thread_id + 1) * size / nb_cores;
	end = end > size ? size : end;

	// Fill the array
	for (int i = start; i < end; i++)
		array[i] = (double)rand() / RAND_MAX * max_minus_min + min;

	// Return
	return 0;
}

/**
 * @brief This function fills an array of double with random values
 * between min and max, using all the cores of the CPU.
 * 
 * @param array			The array to fill
 * @param size			The size of the array
 * @param min			The minimum value
 * @param max			The maximum value
 * 
 * @return int			0 if everything went well, -1 otherwise
 */
int fillRandomDoubleArrayCPUThreads(double* array, int size, double min, double max) {
	
	// Get the number of cores
	int nb_cores;
	#ifdef _WIN32
		#include <windows.h>
		SYSTEM_INFO sysinfo;
		GetSystemInfo(&sysinfo);
		nb_cores = sysinfo.dwNumberOfProcessors;
	#else
		#include <unistd.h>
		nb_cores = (int)sysconf(_SC_NPROCESSORS_CONF);
	#endif

	// Prepare the arguments for the threads
	void* args = malloc(sizeof(double*) + sizeof(int) + sizeof(double) + sizeof(double) + sizeof(int) + sizeof(int));
	((void**)args)[0] = array;
	((int*)args)[1] = size;
	((double*)args)[2] = min;
	((double*)args)[3] = max;
	((int*)args)[4] = nb_cores;
	((int*)args)[5] = 0;	// Thread ID (will be set later)

	// Create the threads
	pthread_t* threads = malloc(nb_cores * sizeof(pthread_t));
	for (int i = 0; i < nb_cores; i++) {
		((int*)args)[5] = i;	// Set the thread ID
		pthread_create(&threads[i], NULL, &fillRandomDoubleArrayCPUThreadsThread, args);
	}

	// Wait for the threads to finish
	for (int i = 0; i < nb_cores; i++)
		pthread_join(threads[i], NULL);
	
	// Free the memory
	free(args);
	free(threads);

	// Return
	return 0;
}

/**
 * @brief This function fills an array of double with random values
 * between min and max, using CPU or GPU depending on the
 * size of the array to maximize performances.
 * 
 * @param array			The array to fill
 * @param size			The size of the array
 * @param min			The minimum value
 * @param max			The maximum value
 * 
 * @return void
 */
void fillRandomDoubleArray(double* array, int size, double min, double max) {
	
	// If the array is big enough, try to use the GPU
	if (size > MIN_SIZE_FOR_GPU && fillRandomDoubleArrayGPU(array, size, min, max) == 0)
		return;

	// If the array is big enough, try to use the CPU with threads
	if (size > MIN_SIZE_FOR_CPU_THREADS && fillRandomDoubleArrayCPUThreads(array, size, min, max) == 0)
		return;

	// Otherwise, use 1 core CPU
	int max_minus_min = max - min;
	for (int i = 0; i < size; i++)
		array[i] = (double)rand() / RAND_MAX * (max_minus_min) + min;
}


