
#ifndef __NEURAL_UTILS_H__
#define __NEURAL_UTILS_H__

#include "neural_config.h"
#include "../universal_pthread.h"

/**
 * @file Utils for neural networks
 * @details How a Neural Network is working?
 * 
 * A neural network is composed of layers of neurons.
 * Those layers can have different sizes each (number of neurons).
 * There are 3 types of layers:
 * - Input layer: the first layer of the neural network, it is the layer that will receive the inputs of the neural network
 * (such as the pixels of an image for example)
 * - Hidden layer: the layers between the input layer and the output layer, these layers consist of neurons that will
 * process the inputs and send them to the next layer depending on their weights and biases
 * - Output layer: the last layer of the neural network, it is the layer that will send the outputs of the neural network
 * (such as the prediction of the neural network for example)
 * 
 * Each neuron of a layer is connected to each neuron of the previous layer (except for the input layer).
 * All the neurons of a layer have:
 * - A bias:				a value that will be added to the sum of the weighted inputs from the previous layer
 * - A list of weights:		a list of values that will be multiplied by the inputs from the previous layer
 * - An activation value:	the value of the neuron after the activation function has been applied to the sum of the weighted inputs
 * - A delta:				the value of the error of the neuron (used for backpropagation)
 * 
 * The activation function is a function that will be applied to the sum of the weighted inputs of a neuron.
 * It is often a sigmoid function (1 / (1 + e^(-x))) but it can be any function.
 * It is used to "activate" the neuron, to give it a value between 0 and 1 (or -1 and 1) depending on the inputs.
 * 
 * The feed forward algorithm is the algorithm that will calculate the outputs of the neural network by feeding the inputs to the neural network.
 * It requires an input array with the same size as the input layer of the neural network.
 * Feeding the inputs means: for each layer of the neural network (except the input layer), calculate the activation values of the neurons.
 * 
 * The backpropagation algorithm is the algorithm that will adjust the weights of the neural network depending on the error of the neural network.
 * It requires an expected output array with the same size as the output layer of the neural network.
 * The error of the neural network is the difference between the expected output and the actual output of the neural network using a loss function.
 * The backpropagation algorithm will calculate the deltas of the neurons of the neural network and adjust the weights of the neural network.
 * 
 * The learning rate is a value between 0 and 1 that will be used to adjust the weights of the neural network.
 * It is used to adjust the weights of the neural network by multiplying the delta of a neuron by the learning rate after the backpropagation algorithm.
 * 
 * @author redactor: Stoupy51 (COLLIGNON Alexandre)
**/


/**
 * @brief Structure representing a layer of neurons
 * 
 * @param nb_neurons						Number of neurons in the layer
 * @param nb_inputs_per_neuron				Number of inputs per neuron
 * 
 * @param activation_function_name			Name of the activation function of the neural network
 * @param activation_function				Activation function of the neural network: how the network will activate the neurons
 * @param activation_function_derivative	Derivative of the activation function of the neural network
 * 
 * @param weights							Weights of the neurons ( [nb_neurons][nb_inputs_per_neuron] )
 * @param activations_values				Outputs of the neurons when activated ( [nb_neurons] )
 * @param biases							Biases of the neurons ( [nb_neurons] )
 * 
 * @param weights_gradients					Gradients of the weights of the neurons ( [nb_neurons][nb_inputs_per_neuron] )
 * @param biases_gradients					Gradients of the biases of the neurons ( [nb_neurons] )
 */
typedef struct NeuronLayer {
	int nb_neurons;				// Arbitrary
	int nb_inputs_per_neuron;	// Depends on the previous layer
	int has_bias_neuron;		// 1 if the layer has a bias neuron set to 1.0, 0 otherwise

	// Activation function variables
	char *activation_function_name;						// Arbitrary, ex: "sigmoid", "tanh", "relu", ...
	nn_type (*activation_function)(nn_type);			// Arbitrary, ex: sigmoid, tanh, relu, ...
	nn_type (*activation_function_derivative)(nn_type);	// Arbitrary, ex: sigmoid_derivative, tanh_derivative, relu_derivative, ...

	// Feed forward variables
	nn_type *weights_flat;		// Single array of weights for better memory management
	nn_type **weights;
	nn_type *activations_values;
	nn_type *biases;

	// Backpropagation gradients variables
	nn_type *weights_gradients_flat;	// Single array of weights gradients for better memory management
	nn_type **weights_gradients;
	nn_type *biases_gradients;
} NeuronLayer;


/**
 * @brief Structure representing a neural network
 * 
 * @param nb_layers				Number of layers in the neural network
 * @param layers				Array of NeuronLayer representing the layers of the neural network
 * @param input_layer			Pointer to the input layer (For easier access and readability)
 * @param output_layer			Pointer to the output layer (For easier access and readability)
 * @param learning_rate			Learning rate of the neural network: how fast the network learns by adjusting the weights
 * 								(0.0 = no learning, 1.0 = full learning)
 * @param loss_function			Loss function of the neural network: how the network will calculate the error
 */
typedef struct NeuralNetwork {
	int nb_layers;											// Arbitrary
	NeuronLayer *layers;									// Containing the input layer, the hidden layers and the output layer
	NeuronLayer *input_layer;								// Pointer to the input layer (For easier access and readability)
	NeuronLayer *output_layer;								// Pointer to the output layer (For easier access and readability)
	nn_type learning_rate;									// Arbitrary, ex: 1.0, 0.5, 0.1, 0.01, 0.001, ...
	char *loss_function_name;								// Arbitrary, ex: "MSE", "MAE", "cross_entropy", ...
	nn_type (*loss_function)(nn_type, nn_type);				// Arbitrary, ex: mean_squared_error, mean_absolute_error, cross_entropy, ...
	nn_type (*loss_function_derivative)(nn_type, nn_type);	// Arbitrary, ex: mean_squared_error_derivative, mean_absolute_error_derivative, cross_entropy_derivative, ...

	long long memory_size;									// Memory size of the neural network (in bytes)
} NeuralNetwork;

int initNeuralNetwork(NeuralNetwork *network, int nb_layers, int nb_neurons_per_layer[], char **activation_function_names, char *loss_function_name, double learning_rate, int has_bias_neurons);
void initGradientsNeuralNetwork(NeuralNetwork *network);
void printNeuralNetwork(NeuralNetwork network);
void printActivationValues(NeuralNetwork network);
void freeNeuralNetwork(NeuralNetwork *network);
int saveNeuralNetwork(NeuralNetwork network, char *filename, int generate_human_readable_file);
int loadNeuralNetwork(NeuralNetwork *network, char *filename);
void deepCloneNeuralNetwork(NeuralNetwork *network_to_clone, NeuralNetwork *cloned_network);

#endif

