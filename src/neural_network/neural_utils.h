
#ifndef __NEURAL_UTILS_H__
#define __NEURAL_UTILS_H__

#include "../universal_utils.h"

// Generate a random double/float between min and max
double generateRandomDouble(double min, double max);
float generateRandomFloat(float min, float max);



/**
 * @brief Structure representing a layer of neurons using double as type
 * 
 * @param nb_neurons			Number of neurons in the layer
 * @param nb_inputs_per_neuron	Number of inputs per neuron
 * 
 * @param weights				Weights of the neurons ( [nb_neurons][nb_inputs_per_neuron] )
 * @param activations_values	Outputs of the neurons when activated ( [nb_neurons] )
 * @param biases				Biases of the neurons ( [nb_neurons] )
 */
typedef struct NeuronLayerD {
	int nb_neurons;				// Arbitrary
	int nb_inputs_per_neuron;	// Depends on the previous layer

	double *weights_flat;		// Single array of weights for better memory management
	double **weights;
	double *activations_values;
	double *biases;
} NeuronLayerD;


/**
 * @brief Structure representing a neural network using double as type
 * 
 * @param nb_layers				Number of layers in the neural network
 * @param layers				Array of NeuronLayerD representing the layers of the neural network
 * @param learning_rate			Learning rate of the neural network: how fast the network learns by adjusting the weights
 * 								(0.0 = no learning, 1.0 = full learning)
 * @param activation_function	Activation function of the neural network: how the network will activate the neurons
 */
typedef struct NeuralNetworkD {
	int nb_layers;							// Arbitrary
	NeuronLayerD *layers;					// [nb_layers] (containing the input layer, the hidden layers and the output layer)
	NeuronLayerD *input_layer;				// Pointer to the input layer (For easier access and readability)
	NeuronLayerD *output_layer;				// Pointer to the output layer (For easier access and readability)
	double learning_rate;					// Arbitrary
	double (*activation_function)(double);	// Arbitrary

	size_t memory_size;						// Memory size of the neural network (in bytes)
} NeuralNetworkD;

NeuralNetworkD createNeuralNetworkD(int nb_layers, int nb_neurons_per_layer[], double learning_rate, double (*activation_function)(double));
void printNeuralNetworkD(NeuralNetworkD network);
void freeNeuralNetworkD(NeuralNetworkD *network);


#endif

