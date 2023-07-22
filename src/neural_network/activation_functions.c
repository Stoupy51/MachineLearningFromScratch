
#include "activation_functions.h"
#include "../universal_utils.h"
#include <math.h>

/**
 * @brief Sigmoid function
 * Returns 1 / (1 + e^(-x)) : a value between 0 and 1
 * 
 * @param x		Value to apply the sigmoid function to
 * 
 * @return The value of the sigmoid function applied to x
 */
nn_type sigmoid_f(nn_type x) {
	#if NN_TYPE == 0
	return 1 / (1 + expf(-x));
	#elif NN_TYPE == 1
	return 1 / (1 + exp(-x));
	#else
	return 1 / (1 + expl(-x));
	#endif
}

/**
 * @brief Derivative of the sigmoid function
 * Returns sigmoid(x) * (1 - sigmoid(x)) : a value between 0 and 1
 * 
 * @param sigmoid_x		Value from sigmoid_f() result
 * 
 * @return The value of the derivative of the sigmoid function applied to x
 */
nn_type sigmoid_derivative_f(nn_type sigmoid_x) {
	return sigmoid_x * (1 - sigmoid_x);
}

/**
 * @brief Rectified Linear Unit function
 * Returns the max between 0 and x : a value between 0 and infinity
 * 
 * @param x		Value to apply the ReLU function to
 * 
 * @return The value of the ReLU function applied to x
 */
nn_type relu_f(nn_type x) {
	return x > 0 ? x : 0;
}

/**
 * @brief Derivative of the Rectified Linear Unit function
 * Returns 1 if x > 0, 0 otherwise
 * 
 * @param relu_x		Value from relu_f() result
 * 
 * @return The value of the derivative of the ReLU function applied to x
 */
nn_type relu_derivative_f(nn_type relu_x) {
	return relu_x > 0 ? 1 : 0;
}

/**
 * @brief Hyperbolic tangent function
 * Returns tanh(x) : a value between -1 and 1
 * 
 * @param x		Value to apply the tanh function to
 * 
 * @return The value of the tanh function applied to x
 */
nn_type tanh_f(nn_type x) {
	#if NN_TYPE == 0
	return tanhf(x);
	#elif NN_TYPE == 1
	return tanh(x);
	#else
	return tanhl(x);
	#endif
}

/**
 * @brief Derivative of the hyperbolic tangent function
 * Returns 1 - tanh(x)^2 : a value between 0 and 1
 * 
 * @param tanh_x		Value from tanh_f() result
 * 
 * @return The value of the derivative of the tanh function applied to x
 */
nn_type tanh_derivative_f(nn_type tanh_x) {
	return 1 - (tanh_x * tanh_x);
}

/**
 * @brief Identity function
 * Does nothing, returns the value of x
 * 
 * @param x		Value to apply the identity function to
 * 
 * @return The same value as x
 */
nn_type identity_f(nn_type x) {
	return x;
}

/**
 * @brief Derivative of the identity function
 * Returns 1
 * 
 * @param x		Useless parameter
 * 
 * @return 1
 */
nn_type identity_derivative_f(nn_type x) {
	(void)x;	// Avoid unused parameter warning
	return 1;
}



/**
 * @brief Get an activation function from its name
 * 
 * @param activation_function_name	Name of the activation function
 * 
 * @return The activation function corresponding to the name
 */
nn_type (*get_activation_function(char *activation_function_name))(nn_type) {
	if (activation_function_name == NULL)
		return NULL;
	else if (strcmp(activation_function_name, "sigmoid") == 0)
		return sigmoid_f;
	else if (strcmp(activation_function_name, "relu") == 0)
		return relu_f;
	else if (strcmp(activation_function_name, "tanh") == 0)
		return tanh_f;
	else if (strcmp(activation_function_name, "identity") == 0)
		return identity_f;
	else {
		ERROR_PRINT("Activation function not found: '%s'", activation_function_name);
		exit(EXIT_FAILURE);
	}
}

/**
 * @brief Get the derivative of an activation function from its name
 * 
 * @param activation_function_name	Name of the activation function
 * 
 * @return The derivative of the activation function corresponding to the name
 */
nn_type (*get_activation_function_derivative(char *activation_function_name))(nn_type) {
	if (activation_function_name == NULL)
		return NULL;
	else if (strcmp(activation_function_name, "sigmoid") == 0)
		return sigmoid_derivative_f;
	else if (strcmp(activation_function_name, "relu") == 0)
		return relu_derivative_f;
	else if (strcmp(activation_function_name, "tanh") == 0)
		return tanh_derivative_f;
	else if (strcmp(activation_function_name, "identity") == 0)
		return identity_derivative_f;
	else {
		ERROR_PRINT("Activation function not found: '%s'", activation_function_name);
		exit(EXIT_FAILURE);
	}
}

