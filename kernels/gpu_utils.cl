
/**
 * @brief Fills an array with random double values.
 * Function called in gpu_utils.c by fillRandomDoubleArrayGPU().
 * 
 * @param array				The array to fill
 * @param size				The size of the array
 * @param min				The minimum value of the random values
 * @param max_minus_min		The maximum value of the random values minus the minimum value (max - min)
 * 
 * @return void
 */
kernel void fillRandomDoubleArrayGPU(global double* array, double size, double min, double max_minus_min) {

	// Get the index of the current element
	int i = get_global_id(0);

	// Fill the array with random values: rand() function does not exist in OpenCL
	if (i < size) {

		// Generate a random looking seed
		uint seed = ((uint)(i + get_global_size(0) + 1)) * 1103515245 + 12345;

		// Generate a random value between min and max
		array[i] = min + (max_minus_min) * ((double)seed / (double)UINT_MAX);
	}
}

