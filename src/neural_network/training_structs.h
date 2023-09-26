
#ifndef __NEURAL_TRAINING_STRUCTS_H__
#define __NEURAL_TRAINING_STRUCTS_H__

#include "neural_utils.h"

/**
 * @brief Training data structure.
 * 
 * @param inputs					Pointer to the inputs array
 * @param targets					Pointer to the target outputs array
 * @param nb_inputs					Number of samples in the inputs array and in the target outputs array
 * @param batch_size				Number of samples in the batch
 * @param test_inputs_percentage	Percentage of the inputs array to use as test inputs (from the end) (usually 10: 10%)
 */
typedef struct {

	nn_type **inputs;
	nn_type **targets;
	int nb_inputs;
	int batch_size;
	int test_inputs_percentage;

} TrainingData;

/**
 * @brief Training parameters structure.
 * 
 * @param nb_epochs					Number of epochs to train the neural network (optional, -1 to disable)
 * @param error_target				Target error value to stop the training (optional, 0.0 to disable)
 * At least one of the two must be specified. If both are specified, the training will stop when one of the two conditions is met.
 * 
 * @param optimizer					Optimizer to use ("SGD", "Adam", "RMSProp", ...)
 * @param loss_function_name		Loss function to use ("MSE", "MAE", "cross_entropy", ...)
 * @param learning_rate				Learning rate to use (0.1, 0.01, 0.001, ...)
 */
typedef struct {

	int nb_epochs;
	nn_type error_target;

	const char *optimizer;
	const char *loss_function_name;
	nn_type learning_rate;

} TrainingParameters;

#endif

