
#include "loss_functions.h"
#include "../universal_utils.h"
#include <math.h>

/**
 * @param predictions		Pointer to the predictions array (nn_type), must be the same size as the output layer
 * @param excepted_values	Pointer to the excepted values array (nn_type), must be the same size as the output layer
 * @param size				Size of the arrays
**/

// Calculate the mean absolute error of the neural network
nn_type mean_absolute_error_f(nn_type *predictions, nn_type *excepted_values, int size) {
	nn_type error = 0.0;
	for (int i = 0; i < size; i++) {
		nn_type diff = excepted_values[i] - predictions[i];
		error += diff < 0 ? -diff : diff;
	}
	return error / size;
}

// Calculate the mean squared error of the neural network
nn_type mean_squared_error_f(nn_type *predictions, nn_type *excepted_values, int size) {
	nn_type error = 0.0;
	for (int i = 0; i < size; i++) {
		nn_type diff = excepted_values[i] - predictions[i];
		error += diff * diff;
	}
	return error / size;
}

// Calculate the huber loss of the neural network: a combination of the mean absolute error and the mean squared error
nn_type huber_loss_f(nn_type *predictions, nn_type *excepted_values, int size) {
	nn_type error = 0.0;
	for (int i = 0; i < size; i++) {
		nn_type diff = excepted_values[i] - predictions[i];
		error += diff < 1 ? diff * diff / 2 : diff - 0.5;
	}
	return error / size;
}

// Calculate the cross entropy of the neural network / logaritmic loss / log loss / logistic loss
nn_type cross_entropy_f(nn_type *predictions, nn_type *excepted_values, int size) {
	nn_type error = 0.0;
	for (int i = 0; i < size; i++)
		#if NN_TYPE == 0
		error += excepted_values[i] * logf(predictions[i]) + (1 - excepted_values[i]) * logf(1 - predictions[i]);
		#elif NN_TYPE == 1
		error += excepted_values[i] * log(predictions[i]) + (1 - excepted_values[i]) * log(1 - predictions[i]);
		#else
		error += excepted_values[i] * logl(predictions[i]) + (1 - excepted_values[i]) * logl(1 - predictions[i]);
		#endif
	return -error / size;
}

// Calculate the relative entropy of the neural network / Kullback-Leibler divergence / KL divergence / KL distance / information divergence / information gain
nn_type relative_entropy_f(nn_type *predictions, nn_type *excepted_values, int size) {
	nn_type error = 0.0;
	for (int i = 0; i < size; i++)
		#if NN_TYPE == 0
		error += excepted_values[i] * logf(excepted_values[i] / predictions[i]) + (1 - excepted_values[i]) * logf((1 - excepted_values[i]) / (1 - predictions[i]));
		#elif NN_TYPE == 1
		error += excepted_values[i] * log(excepted_values[i] / predictions[i]) + (1 - excepted_values[i]) * log((1 - excepted_values[i]) / (1 - predictions[i]));
		#else
		error += excepted_values[i] * logl(excepted_values[i] / predictions[i]) + (1 - excepted_values[i]) * logl((1 - excepted_values[i]) / (1 - predictions[i]));
		#endif
	return error / size;
}

// Calculate the squared hinge of the neural network: max(0, 1 - y * y_pred) * max(0, 1 - y * y_pred)
nn_type squared_hinge_f(nn_type *predictions, nn_type *excepted_values, int size) {
	nn_type error = 0.0;
	for (int i = 0; i < size; i++) {
		nn_type diff = 1 - excepted_values[i] * predictions[i];
		error += diff < 0 ? 0 : diff * diff;
	}
	return error / size;
}


/**
 * @brief Function to get a loss function from its name
 * 
 * @param loss_function_name	Name of the loss function
 * 
 * @return The loss function corresponding to the name
 */
nn_type (*get_loss_function(char *loss_function_name))(nn_type*, nn_type*, int) {
	if (strcmp(loss_function_name, "MAE") == 0 || strcmp(loss_function_name, "mean_absolute_error") == 0)
		return mean_absolute_error_f;
	else if (strcmp(loss_function_name, "MSE") == 0 || strcmp(loss_function_name, "mean_squared_error") == 0)
		return mean_squared_error_f;
	else if (strcmp(loss_function_name, "huber_loss") == 0)
		return huber_loss_f;
	else if (strcmp(loss_function_name, "cross_entropy") == 0)
		return cross_entropy_f;
	else if (strcmp(loss_function_name, "relative_entropy") == 0)
		return relative_entropy_f;
	else if (strcmp(loss_function_name, "squared_hinge") == 0)
		return squared_hinge_f;
	else {
		ERROR_PRINT("Loss function not found: '%s'", loss_function_name);
		exit(EXIT_FAILURE);
	}
}

