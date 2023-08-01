
## Python equivalent of basic_plus_operation.c
# This program creates a neural network with 5 layers
# The first layer has 64 neurons, the second 48, the third 48, the fourth 48 and the fifth (output) 32
# All activation functions are sigmoid

# Import training data
training_data_str = ""
with open("bin/basic_plus_operation.data", "r") as file:
	training_data_str = file.read()
training_data = [[], []]
for line in training_data_str.split("\n"):
	input_data = []
	output_data = []
	i = 0
	for num in line.split(" "):
		if num == "":
			continue
		if i < 64:
			input_data.append(float(num))
		else:
			output_data.append(float(num))
		i += 1
	if len(input_data) > 0:
		training_data[0].append(input_data)
		training_data[1].append(output_data)

# Import the libraries
import numpy as np
from tensorflow import keras
from keras import layers

# Create the model
model = keras.Sequential(
	[
		keras.Input(shape = (64, )),
		layers.Dense(48, activation = "sigmoid"),
		layers.Dense(48, activation = "sigmoid"),
		layers.Dense(48, activation = "sigmoid"),
		layers.Dense(32, activation = "sigmoid"),
	]
)

# Print the model summary
model.summary()

# Compile the model with MSE loss, Adam optimizer and accuracy metric
model.compile(
	loss = keras.losses.MeanSquaredError(),
	optimizer = keras.optimizers.Adam(),
	metrics = ["accuracy"],
)

# Train the model using multi threading
model.fit(
	np.array(training_data[0]),
	np.array(training_data[1]),
	batch_size = 1,
	epochs = 200,
	verbose = 2,
	validation_split = 0.2,
	use_multiprocessing = True,
	workers = 8,
)

# Convert a binary double array to an integer
def convertBinaryDoubleArrayToInt(binary_double_array, start_index):
	result = 0
	for i in range(32):
		result |= int(binary_double_array[start_index + i] + 0.5) << i
	return result

# Test the model on all the training data like in the C code
test_inputs = np.array(training_data[0])
test_expected = np.array(training_data[1])
test_outputs = model.predict(test_inputs, batch_size = len(test_inputs))
nb_errors = 0
for i in range(len(test_inputs)):
	a = convertBinaryDoubleArrayToInt(test_inputs[i], 0)
	b = convertBinaryDoubleArrayToInt(test_inputs[i], 32)
	c = convertBinaryDoubleArrayToInt(test_outputs[i], 0)
	d = convertBinaryDoubleArrayToInt(test_expected[i], 0)
	if c != d:
		nb_errors += 1
		print(f"main(): Error for {a} + {b} = {c} (expected {d})")
print(f"main(): Success rate: {len(test_inputs) - nb_errors}/{len(test_inputs)} ({(len(test_inputs) - nb_errors) / len(test_inputs) * 100.0}%)")

# Save the model
model.save("bin/basic_plus_operation.py_model")

