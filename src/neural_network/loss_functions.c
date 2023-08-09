
#include "loss_functions.h"
#include "../universal_utils.h"
#include <math.h>

/**
 * @param prediction		Value predicted by the neural network
 * @param target_value		Value that the neural network should have predicted
**/

#if NN_TYPE == 0
#define nn_type_log logf
#elif NN_TYPE == 1
#define nn_type_log log
#else
#define nn_type_log logl
#endif

// Calculate the mean absolute error of the neural network
nn_type mean_absolute_error_f(nn_type prediction, nn_type target_value) {
	nn_type diff = target_value - prediction;
	return diff < 0 ? -diff : diff;
}

// Derivative of the mean absolute error of the neural network
nn_type mean_absolute_error_derivative(nn_type prediction, nn_type target_value) {
	return prediction < target_value ? -1 : 1;
}

// Calculate the mean squared error of the neural network
nn_type mean_squared_error_f(nn_type prediction, nn_type target_value) {
	nn_type diff = target_value - prediction;
	return diff * diff;
}

// Derivative of the mean squared error of the neural network
nn_type mean_squared_error_derivative(nn_type prediction, nn_type target_value) {
	return 2 * (prediction - target_value);
}

// Calculate the huber loss of the neural network: a combination of the mean absolute error and the mean squared error
nn_type huber_loss_f(nn_type prediction, nn_type target_value) {
	nn_type diff = target_value - prediction;
	return diff < 1 ? diff * diff / 2 : diff - 0.5;
}

// Derivative of the huber loss of the neural network
nn_type huber_loss_derivative(nn_type prediction, nn_type target_value) {
	nn_type diff = target_value - prediction;
	return diff < 1 ? -diff : -1;
}

// Calculate the cross entropy of the neural network / logaritmic loss / log loss / logistic loss
nn_type cross_entropy_f(nn_type prediction, nn_type target_value) {
	return -(target_value * nn_type_log(prediction)
		+ (1 - target_value) * nn_type_log(1 - prediction));
}

// Derivative of the cross entropy of the neural network
nn_type cross_entropy_derivative(nn_type prediction, nn_type target_value) {
	return (prediction - target_value) / (prediction * (1 - prediction));
}

// Calculate the relative entropy of the neural network / Kullback-Leibler divergence / KL divergence / KL distance / information divergence / information gain
nn_type relative_entropy_f(nn_type prediction, nn_type target_value) {
	return target_value * nn_type_log(target_value / prediction)
		+ (1 - target_value) * nn_type_log((1 - target_value) / (1 - prediction));
}

// Derivative of the relative entropy of the neural network
nn_type relative_entropy_derivative(nn_type prediction, nn_type target_value) {
	return (prediction - target_value) / (prediction * (1 - prediction));
}

// Calculate the squared hinge of the neural network: max(0, 1 - y * y_pred) * max(0, 1 - y * y_pred)
nn_type squared_hinge_f(nn_type prediction, nn_type target_value) {
	nn_type diff = 1 - target_value * prediction;
	return diff < 0 ? 0 : diff * diff;
}

// Derivative of the squared hinge of the neural network
nn_type squared_hinge_derivative(nn_type prediction, nn_type target_value) {
	nn_type diff = 1 - target_value * prediction;
	return diff < 0 ? 0 : -2 * diff * target_value;
}

/**
 * @brief Function to get a loss function from its name
 * 
 * @param loss_function_name	Name of the loss function
 * 
 * @return The loss function corresponding to the name
 */
nn_type (*get_loss_function(char *loss_function_name))(nn_type, nn_type) {
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

/**
 * @brief Function to get the derivative of a loss function from its name
 * 
 * @param loss_function_name	Name of the loss function
 * 
 * @return The derivative of the loss function corresponding to the name
 */
nn_type (*get_loss_function_derivative(char *loss_function_name))(nn_type, nn_type) {
	if (strcmp(loss_function_name, "MAE") == 0 || strcmp(loss_function_name, "mean_absolute_error") == 0)
		return mean_absolute_error_derivative;
	else if (strcmp(loss_function_name, "MSE") == 0 || strcmp(loss_function_name, "mean_squared_error") == 0)
		return mean_squared_error_derivative;
	else if (strcmp(loss_function_name, "huber_loss") == 0)
		return huber_loss_derivative;
	else if (strcmp(loss_function_name, "cross_entropy") == 0)
		return cross_entropy_derivative;
	else if (strcmp(loss_function_name, "relative_entropy") == 0)
		return relative_entropy_derivative;
	else if (strcmp(loss_function_name, "squared_hinge") == 0)
		return squared_hinge_derivative;
	else {
		ERROR_PRINT("Loss function not found: '%s'", loss_function_name);
		exit(EXIT_FAILURE);
	}
}


