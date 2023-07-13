
#include "random_array_values.h"

#include "../universal_utils.h"
#include "../gpu/gpu_utils.h"
#include "../universal_pthread.h"


// Arbitrary values to determine if the GPU or the CPU should be used
// The values are based on my own tests, and may not be optimal for your own hardware
#define MIN_SIZE_FOR_CPU_THREADS 500000

struct frdacptt_args_t {
	double* array;
	int size;
	double min;
	double max_minus_min;
	int nb_cores;
	int thread_id;
};
struct frdacptt_args_ptr_t {
	struct frdacptt_args_t* args;
	int thread_id;
};

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
	int thread_id = ((struct frdacptt_args_ptr_t*)args)->thread_id;
	struct frdacptt_args_t* args_struct = ((struct frdacptt_args_ptr_t*)args)->args;

	// Calculate the part of the array to fill
	int start = thread_id * args_struct->size / args_struct->nb_cores;
	int end = (thread_id + 1) * args_struct->size / args_struct->nb_cores;
	end = end > args_struct->size ? args_struct->size : end;

	// Fill the array
	for (int i = start; i < end; i++)
		args_struct->array[i] = (double)rand() / RAND_MAX * args_struct->max_minus_min + args_struct->min;

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
	struct frdacptt_args_t args;
	args.array = array;
	args.size = size;
	args.min = min;
	args.max_minus_min = max - min;
	args.nb_cores = nb_cores;
	struct frdacptt_args_ptr_t *args_ptr = malloc(nb_cores * sizeof(struct frdacptt_args_ptr_t));
	ERROR_HANDLE_PTR_RETURN_INT(args_ptr, "fillRandomDoubleArrayCPUThreads(): Failed to allocate memory for the args_ptr\n");
	for (int i = 0; i < nb_cores; i++) {
		args_ptr[i].args = &args;
		args_ptr[i].thread_id = i;
	}

	// Create the threads
	pthread_t* threads = malloc(nb_cores * sizeof(pthread_t));
	for (int i = 0; i < nb_cores; i++) {
		pthread_create(&threads[i], NULL, fillRandomDoubleArrayCPUThreadsThread, &args_ptr[i]);
	}

	// Wait for the threads to finish
	for (int i = 0; i < nb_cores; i++)
		pthread_join(threads[i], NULL);
	
	// Free the memory
	free(args_ptr);
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

	// If the array is big enough, try to use the CPU with threads
	if (size > MIN_SIZE_FOR_CPU_THREADS && fillRandomDoubleArrayCPUThreads(array, size, min, max) == 0)
		return;

	// Otherwise, use 1 core CPU
	int max_minus_min = max - min;
	for (int i = 0; i < size; i++)
		array[i] = (double)rand() / RAND_MAX * (max_minus_min) + min;
}

