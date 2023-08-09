
#include "activation_functions.h"
#include "../universal_utils.h"
#include <math.h>

#if NN_TYPE == 0

#define nn_type_exp expf
#define nn_type_tanh tanhf

#elif NN_TYPE == 1

#define nn_type_exp exp
#define nn_type_tanh tanh

#else

#define nn_type_exp expl
#define nn_type_tanh tanhl

#endif


/**
 * @brief Sigmoid function
 * Returns 1 / (1 + e^(-x)) : a value between 0 and 1
 * 
 * @param values	Values to apply the sigmoid function to
 * @param n			Number of values
 */
void sigmoid_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = 1.0 / (1.0 + nn_type_exp(-values[i]));
}

/**
 * @brief Derivative of the sigmoid function
 * Returns sigmoid(x) * (1 - sigmoid(x)) : a value between 0 and 1
 * 
 * @param values	Values from sigmoid_f() result
 * @param n			Number of values
 */
void sigmoid_derivative_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = values[i] * (1.0 - values[i]);
}

/**
 * @brief Rectified Linear Unit function
 * Returns the max between 0 and x : a value between 0 and infinity
 * 
 * @param values	Values to apply the ReLU function to
 * @param n			Number of values
 */
void relu_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = values[i] > 0.0 ? values[i] : 0.0;
}

/**
 * @brief Derivative of the Rectified Linear Unit function
 * Returns 1 if x > 0, 0 otherwise
 * 
 * @param values	Values from relu_f() result
 * @param n			Number of values
 */
void relu_derivative_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = values[i] > 0.0 ? 1.0 : 0.0;
}

/**
 * @brief Hyperbolic tangent function
 * Returns tanh(x) : a value between -1 and 1
 * 
 * @param values	Values to apply the tanh function to
 * @param n			Number of values
 */
void tanh_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = nn_type_tanh(values[i]);
}

/**
 * @brief Derivative of the hyperbolic tangent function
 * Returns 1 - tanh(x)^2 : a value between 0 and 1
 * 
 * @param values	Values from tanh_f() result
 * @param n			Number of values
 */
void tanh_derivative_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = 1.0 - (values[i] * values[i]);
}

/**
 * @brief Identity function
 * Does nothing, returns the value of x
 * 
 * @param values	Values to apply the identity function to
 * @param n			Number of values
 */
void identity_f(nn_type *values, int n) {
	// Avoid unused parameter warning
	(void)values;
	(void)n;
}

/**
 * @brief Derivative of the identity function
 * Returns 1
 * 
 * @param values	Useless parameter
 * @param n			Number of values
 */
void identity_derivative_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = 1.0;
}

/**
 * @brief Softmax function
 * Returns e^x / sum(e^x) : a value between 0 and 1
 * 
 * @param values	Values to apply the softmax function to
 * @param n			Number of values
 */
void softmax_f(nn_type *values, int n) {
	nn_type max_val = values[0];
	for (int i = 1; i < n; i++)
		if (values[i] > max_val)
			max_val = values[i];
	nn_type sum_exp = 0.0;
	for (int i = 0; i < n; i++) {
		values[i] = nn_type_exp(values[i] - max_val);
		sum_exp += values[i];
	}
	for (int i = 0; i < n; i++)
		values[i] /= sum_exp;
}

/**
 * @brief Derivative of the softmax function
 * Returns softmax(x) * (1 - softmax(x)) : a value between 0 and 1
 * 
 * @param values	Values from softmax_f() result
 * @param n			Number of values
 */
void softmax_derivative_f(nn_type *values, int n) {
	for (int i = 0; i < n; i++)
		values[i] = values[i] * (1.0 - values[i]);
}


/**
 * @brief Get an activation function from its name
 * 
 * @param activation_function_name	Name of the activation function
 * 
 * @return The activation function corresponding to the name
 */
void (*get_activation_function(char *activation_function_name))(nn_type*, int) {
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
	else if (strcmp(activation_function_name, "softmax") == 0)
		return softmax_f;
	else {
		ERROR_PRINT("Activation function not found: '%s'\n", activation_function_name);
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
void (*get_activation_function_derivative(char *activation_function_name))(nn_type*, int) {
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
	else if (strcmp(activation_function_name, "softmax") == 0)
		return softmax_derivative_f;
	else {
		ERROR_PRINT("Activation function not found: '%s'\n", activation_function_name);
		exit(EXIT_FAILURE);
	}
}

