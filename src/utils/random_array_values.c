
#include "random_array_values.h"

#include "../universal_utils.h"
#include "../gpu/gpu_utils.h"
#include "../universal_pthread.h"


// Arbitrary values to determine if the GPU or the CPU should be used
// The values are based on my own tests, and may not be optimal for your own hardware
#define MIN_SIZE_FOR_CPU_THREADS 500000

struct frdacptt_args_t {
	nn_type* array;
	unsigned long long size;
	nn_type min;
	nn_type max_minus_min;
	int nb_cores;
};
struct frdacptt_args_t frdacptt_args;

/**
 * @brief Thread function to fill an array of nn_type with random values
 * between min and max, using all the cores of the CPU.
 * 
 * @param args			The arguments of the thread
 * 
 * @return 0
 */
thread_return_type fillRandomFloatArrayCPUThreadsThread(thread_param_type args) {

	// Get the argument
	int thread_id = *(int*)args;

	// Calculate the part of the array to fill
	unsigned long long start = (unsigned long long)thread_id * frdacptt_args.size / frdacptt_args.nb_cores;
	unsigned long long end = (unsigned long long)(thread_id + 1) * frdacptt_args.size / frdacptt_args.nb_cores;
	end = end > frdacptt_args.size ? frdacptt_args.size : end;

	// Fill the array
	for (unsigned long long i = start; i < end; i++)
		frdacptt_args.array[i] = (nn_type)rand() / RAND_MAX * frdacptt_args.max_minus_min + frdacptt_args.min;

	// Return
	return 0;
}

/**
 * @brief This function fills an array of nn_type with random values
 * between min and max, using all the cores of the CPU.
 * 
 * @param array			The array to fill
 * @param size			The size of the array
 * @param min			The minimum value
 * @param max			The maximum value
 * 
 * @return int			0 if everything went well, -1 otherwise
 */
int fillRandomFloatArrayCPUThreads(nn_type* array, unsigned long long size, nn_type min, nn_type max) {
	
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
	frdacptt_args.array = array;
	frdacptt_args.size = size;
	frdacptt_args.min = min;
	frdacptt_args.max_minus_min = max - min;
	frdacptt_args.nb_cores = nb_cores;
	int *threads_ids = malloc(nb_cores * sizeof(int));
	ERROR_HANDLE_PTR_RETURN_INT(threads_ids, "fillRandomFloatArrayCPUThreads(): Failed to allocate memory for the threads_ids\n");

	// Create the threads
	pthread_t* threads = malloc(nb_cores * sizeof(pthread_t));
	for (int i = 0; i < nb_cores; i++) {
		threads_ids[i] = i;
		pthread_create(&threads[i], NULL, fillRandomFloatArrayCPUThreadsThread, &threads_ids[i]);
	}

	// Wait for the threads to finish
	for (int i = 0; i < nb_cores; i++)
		pthread_join(threads[i], NULL);
	
	// Free the memory
	free(threads_ids);
	free(threads);

	// Return
	return 0;
}

/**
 * @brief This function fills an array of nn_type with random values
 * between min and max, using CPU or GPU depending on the
 * size of the array to maximize performances.
 * 
 * @param array			The array to fill
 * @param size			The size of the array
 * @param min			The minimum value
 * @param max			The maximum value
 */
void fillRandomFloatArray(nn_type* array, unsigned long long size, nn_type min, nn_type max) {

	// If the array is big enough, try to use the CPU with threads
	if (size > MIN_SIZE_FOR_CPU_THREADS && fillRandomFloatArrayCPUThreads(array, size, min, max) == 0)
		return;

	// Otherwise, use 1 core CPU
	int max_minus_min = max - min;
	for (unsigned long long i = 0; i < size; i++)
		array[i] = (nn_type)rand() / RAND_MAX * (max_minus_min) + min;
}

