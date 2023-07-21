
#ifndef __ACTIVATION_FUNCTIONS_H__
#define __ACTIVATION_FUNCTIONS_H__

#include "neural_config.h"

// Activation functions for neurons
nn_type sigmoid_f(nn_type x);
nn_type sigmoid_derivative_f(nn_type x);
nn_type relu_f(nn_type x);
nn_type relu_derivative_f(nn_type x);
nn_type tanh_f(nn_type x);
nn_type tanh_derivative_f(nn_type x);
nn_type identity_f(nn_type x);
nn_type identity_derivative_f(nn_type x);

// Function to get an activation function from its name & derivative
nn_type (*get_activation_function(char *activation_function_name))(nn_type);
nn_type (*get_activation_function_derivative(char *activation_function_name))(nn_type);


#endif

