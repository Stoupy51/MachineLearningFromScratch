
#include "loss_functions.h"

/**
 * @brief Calculate the mean squared error of the output layer of the neural network
 * 
 * @param predictions		Pointer to the predictions array (nn_type), must be the same size as the output layer
 * @param excepted_values	Pointer to the excepted values array (nn_type), must be the same size as the output layer
 * 
 * @return nn_type			Mean squared error of the output layer of the neural network
 */
nn_type mean_squared_error_f(nn_type *predictions, nn_type *excepted_values, int size) {
	nn_type error = 0.0;
	for (int i = 0; i < size; i++) {
		nn_type diff = excepted_values[i] - predictions[i];
		error += diff * diff;
	}
	return error / size;
}


