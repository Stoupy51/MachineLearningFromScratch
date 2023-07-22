
#ifndef __LOSS_FUNCTIONS_H__
#define __LOSS_FUNCTIONS_H__

#include "neural_config.h"

/**
 * @file Loss functions for neural networks
 * @details What is a loss function?
 * The loss function is a method of evaluating how well your algorithm models your dataset.
 * If your predictions are totally off, your loss function will output a higher number.
 * If they’re pretty good, it’ll output a lower number.
 * 
 * They are used to train neural networks by measuring how far the network’s predictions are from the intended target.
 * In the training algorithm, the loss function is used to calculate the gradient, which is then used to update the weights.
**/

// Loss functions for neural networks
nn_type mean_absolute_error_f(nn_type *predictions, nn_type *excepted_values, int size);
nn_type mean_squared_error_f(nn_type *predictions, nn_type *excepted_values, int size);
nn_type huber_loss_f(nn_type *predictions, nn_type *excepted_values, int size);
nn_type cross_entropy_f(nn_type *predictions, nn_type *excepted_values, int size);
nn_type relative_entropy_f(nn_type *predictions, nn_type *excepted_values, int size);
nn_type squared_hinge_f(nn_type *predictions, nn_type *excepted_values, int size);

// Function to get a loss function from its name
nn_type (*get_loss_function(char *loss_function_name))(nn_type*, nn_type*, int);

#endif

