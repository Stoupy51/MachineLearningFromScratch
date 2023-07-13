
/**
 * @brief Fills an array with random double values.
 * Function called in gpu_utils.c by fillRandomDoubleArrayGPU().
 * 
 * @param array		The array to fill
 * @param size		The size of the array
 * @param min		The minimum value of the random values
 * @param max		The maximum value of the random values
 * 
 * @return void
 */
kernel void fillRandomDoubleArrayGPU(global double* array, int size, int min, int max) {

	// Get the index of the current element
	int i = get_global_id(0);

	// Fill the array with random values: rand() function does not exist in OpenCL
	if (i < size) {

		// Generate a random looking seed
		uint seed = ((uint)(i + get_global_size(0) + 1)) * 1103515245 + 12345;

		// Generate a random value between min and max
		array[i] = min + (max - min) * ((double)seed / (double)UINT_MAX);
	}
}

