
/**
 * @brief Compute the first vector to the power of the second vector
 * using the naive method of exponentiation.
 * The output is stored in the first vector.
 * 
 * @param first_vector		The first vector
 * @param second_vector		The second vector
 * 
 * @return void
 */
kernel void computePowerNaiveExponentiation(global int* first_vector, global int* second_vector, int n) {

	// Get the index of the current element
	int i = get_global_id(0);

	// Compute the power using Fast exponentiation
	if (i < n) {

		// Get the power and the value
		int result = 1;
		int power = second_vector[i];
		int value = first_vector[i];
		int j = 0;

		// Multiply the value by itself power times
		for (j = 0; j < power; j++)
			result *= value;

		// Store the result in the first vector
		first_vector[i] = result;
	}
}

/**
 * @brief Compute the first vector to the power of the second vector
 * using the fast method of exponentiation.
 * The output is stored in the first vector.
 * 
 * @param first_vector		The first vector
 * @param second_vector		The second vector
 * 
 * @return void
 */
kernel void computePowerFastExponentiation(global int* first_vector, global int* second_vector, int n) {

	// Get the index of the current element
	int i = get_global_id(0);

	// Compute the power using Fast exponentiation
	if (i < n) {

		// Get the power and the value
		int result = 1;
		int power = second_vector[i];
		int value = first_vector[i];

		// While the power is not 0
		while (power > 0) {

			// If the power is odd, multiply the result by the value
			if (power % 2 == 1)
				result *= value;

			// In every case, square the value and divide the power by 2
			value *= value;
			power /= 2;
		}

		// Store the result in the first vector
		first_vector[i] = result;
	}
}


