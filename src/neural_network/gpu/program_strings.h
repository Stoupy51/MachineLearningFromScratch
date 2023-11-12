
#ifndef __PROGRAM_STRINGS_H__
#define __PROGRAM_STRINGS_H__

#define GC_FEED_FORWARD_FUNCTION \
"kernel void feed_forward(global "NN_STRING"* previous_layer_activation_values, global "NN_STRING"* weights, global "NN_STRING"* biases, global "NN_STRING"* activation_values, int previous_layer_size) {" \
	"int neuron = get_global_id(0);" \
	"int weight_index = neuron * previous_layer_size;" \
	""NN_STRING" sum = biases[neuron];" \
	"for (int i = 0; i < previous_layer_size; i++) {" \
		"sum += previous_layer_activation_values[i] * weights[weight_index + i];" \
	"}" \
	"activation_values[neuron] = __ACTIVATION_FUNCTION__;" \
"}\0"

// Softmax activation function launched after "feed_forward_function" assuming that exp() is already applied to the activation values
#define GC_FEED_FORWARD_SOFTMAX \
"kernel void feed_forward_softmax(global "NN_STRING"* activation_values, int current_layer_size) {" \
	""NN_STRING" sum = 0.0;" \
	"for (int i = 0; i < current_layer_size; i++) {" \
		"sum += activation_values[i];" \
	"}" \
	"for (int i = 0; i < current_layer_size; i++) {" \
		"activation_values[i] /= sum;" \
	"}" \
"}\0"

// Gradient calculations
#define GC_OUTPUT_LOSS_AND_GRADIENT \
"kernel void output_loss_and_gradient(global "NN_STRING"* prediction, global "NN_STRING"* target, global "NN_STRING"* current_activation_values, global "NN_STRING"* previous_activation_values, global "NN_STRING"* biases_gradients, global "NN_STRING"* weights_gradients_flat, int nb_inputs_per_neuron) {" \
	"int neuron = get_global_id(0);" \
	"int weight_index = neuron * nb_inputs_per_neuron;" \
	""NN_STRING" gradient = __LOSS_FUNCTION_DERIVATIVE__;" \
	"gradient *= current_activation_values[neuron];" \
	"for (int i = 0; i < nb_inputs_per_neuron; i++) {" \
		"weights_gradients_flat[weight_index + i] += gradient * previous_activation_values[i];" \
	"}" \
	"biases_gradients[neuron] += gradient;" \
"}\0"

#define GC_ACTIVATION_DERIVATIVE \
"kernel void activation_derivative(global "NN_STRING"* activation_values, int nb_neurons) {" \
	"int neuron = get_global_id(0);" \
	"activation_values[neuron] = __ACTIVATION_FUNCTION_DERIVATIVE__;" \
"}\0"

#define GC_HIDDEN_LAYER_GRADIENT \
"kernel void hidden_layer_gradient(global "NN_STRING"* next_weights, int next_layer_nb_neurons, global "NN_STRING"* current_activation_values, global "NN_STRING"* previous_activation_values, global "NN_STRING"* biases_gradients, global "NN_STRING"* weights_gradients_flat, int nb_inputs_per_neuron) {" \
	"int neuron = get_global_id(0);" \
	"int weight_index = neuron * nb_inputs_per_neuron;" \
	""NN_STRING" gradient = 0.0;" \
	"for (int i = 0; i < next_layer_nb_neurons; i++) {" \
		"gradient += next_weights[i * next_layer_nb_neurons + neuron];" \
	"}" \
	"gradient *= current_activation_values[neuron];" \
	"for (int i = 0; i < nb_inputs_per_neuron; i++) {" \
		"weights_gradients_flat[weight_index + i] += gradient * previous_activation_values[i];" \
	"}" \
	"biases_gradients[neuron] += gradient;" \
"}\0"

#define GC_CALCULATE_ERROR \
"kernel void calculate_error(global "NN_STRING"* predictions, global "NN_STRING"* target_tests, global "NN_STRING"* error, int nb_neurons) {" \
	"int sample = get_global_id(0);" \
	"int target_index = sample * nb_neurons;" \
	"for (int j = 0; j < nb_neurons; j++) {" \
		"__LOSS_FUNCTION__" \
	"}" \
	"error[sample] /= nb_neurons;" \
"}\0"



///// Update weights and biases /////
// SGD
#define GC_UPDATE_WEIGHTS_AND_BIASES_SGD \
"kernel void update_weights_and_biases(global "NN_STRING"* weights, global "NN_STRING"* biases, global "NN_STRING"* weights_gradients, global "NN_STRING"* biases_gradients, int nb_inputs_per_neuron) {" \
	"int index = get_global_id(0);" \
	"int weight_index = index * nb_inputs_per_neuron;" \
	"for (int i = 0; i < nb_inputs_per_neuron; i++) {" \
		"weights[weight_index + i] -= __LEARNING_RATE__ * weights_gradients[weight_index + i];" \
	"}" \
	"biases[index] -= __LEARNING_RATE__ * biases_gradients[index];" \
"}\0"

// Adam
#define GC_UPDATE_WEIGHTS_AND_BIASES_ADAM \
"kernel void update_weights_and_biases(global "NN_STRING"* weights, global "NN_STRING"* biases, global "NN_STRING"* weights_gradients, global "NN_STRING"* biases_gradients, global "NN_STRING"* m, global "NN_STRING"* v, global "NN_STRING"* m_hat, global "NN_STRING"* v_hat, "NN_STRING" minus_beta1_t, "NN_STRING" minus_beta2_t, int nb_inputs_per_neuron) {" \
	"int index = get_global_id(0);" \
	"int weight_index = index * nb_inputs_per_neuron;" \
	"for (int i = 0; i < nb_inputs_per_neuron; i++) {" \
		"m[index] = 0.9 * m[index] + 0.1 * weights_gradients[weight_index + i];" \
		"v[index] = 0.999 * v[index] + 0.001 * weights_gradients[weight_index + i] * weights_gradients[weight_index + i];" \
		"m_hat[index] = m[index] / minus_beta1_t;" \
		"v_hat[index] = v[index] / minus_beta2_t;" \
		"weights[weight_index + i] -= (__LEARNING_RATE__ * m_hat[index]) / (sqrt(v_hat[index]) + 1e-8);" \
	"}" \
	"m[index] = 0.9 * m[index] + 0.1 * biases_gradients[index];" \
	"v[index] = 0.999 * v[index] + 0.001 * biases_gradients[index] * biases_gradients[index];" \
	"m_hat[index] = m[index] / minus_beta1_t;" \
	"v_hat[index] = v[index] / minus_beta2_t;" \
	"biases[index] -= (__LEARNING_RATE__ * m_hat[index]) / (sqrt(v_hat[index]) + 1e-8);" \
"}\0"
#define GC_UPDATE_PARAMETERS_ADAM \
"kernel void update_parameters_adam(global "NN_STRING"* beta1_t, global "NN_STRING"* beta2_t, global "NN_STRING"* minus_beta1_t, global "NN_STRING"* minus_beta2_t) {" \
	"beta1_t[0] *= 0.9;" \
	"beta2_t[0] *= 0.999;" \
	"minus_beta1_t[0] = 1.0 - beta1_t[0];" \
	"minus_beta2_t[0] = 1.0 - beta2_t[0];" \
"}\0"

#endif

