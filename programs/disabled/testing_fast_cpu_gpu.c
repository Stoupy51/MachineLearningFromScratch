
#include "../src/universal_utils.h"
#include "../src/utils/random_array_values.h"
#include "../src/gpu/gpu_utils.h"
#include "../src/st_benchmark.h"

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * Conclusions:
 * - The GPU should not be used to generate random values because it is slower than 12 CPU threads.
 * - Single core CPU should be used to generate random values if the size is less than 500000.
 * - Otherwise, multiple CPU threads should be used to generate random values if the size is greater than 500000.
 */
void exitProgram() {

	// Print end of program
	INFO_PRINT("exitProgram(): End of program, press enter to exit\n");
	getchar();
	exit(0);
}

/**
 * This program benchmarks the CPU and GPU to determine
 * which one is the fastest for a given array size.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'testing_random_array_values' program\n");
	atexit(exitProgram);

	// Create an array of double
	#define ARRAY_SIZE 1000000000 // (1 GB)
	double* array = malloc(ARRAY_SIZE * sizeof(double));
	ERROR_HANDLE_PTR_RETURN_INT(array, "main(): Failed to allocate memory for the array\n");

	// Sizes to test
	int sizes[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
	int nb_sizes = sizeof(sizes) / sizeof(int);

	// Min and max values for the random values
	double min = -1.0;
	double max = 1.0;
	double max_minus_min = max - min;

	// For each size,
	for (int i = 0; i < nb_sizes; i++) {

		// Print the size
		PRINTER("\n");
		INFO_PRINT("main(): Testing size %d\n", sizes[i]);
		char buffer[2048];

		// Benchmark a single CPU thread
		ST_BENCHMARK_SOLO_TIME(buffer,
			{
				for (int j = 0; j < sizes[i]; j++)
					array[j] = (double)rand() / RAND_MAX * (max_minus_min) + min;
			},
			"singleCpuThread", 5	// 5 seconds at least
		);
		PRINTER(buffer);

		// Benchmark multiple CPU threads
		ST_BENCHMARK_SOLO_TIME(buffer,
			{
				fillRandomFloatArrayCPUThreads(array, sizes[i], min, max);
			},
			"multipleCpuThreads", 5	// 5 seconds at least
		);
		PRINTER(buffer);

		// Benchmark the GPU
		ST_BENCHMARK_SOLO_TIME(buffer,
			{
				fillRandomFloatArrayGPU(array, sizes[i], min, max);
			},
			"GPU", 5	// 5 seconds at least
		);
		PRINTER(buffer);
	}

	// Free the array
	free(array);

	// Final print and return
	INFO_PRINT("main(): End of program\n");
	return 0;
}

