
#include <sys/types.h>

#include "../src/universal_utils.h"
#include "../src/math/sigmoid.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_gpu.h"
#include "../src/list/image_list.h"
#include "../src/st_benchmark.h"

#define NEURAL_NETWORK_PATH "bin/image_upscaler_network.bin"
#define IMAGES_PATH "images/"

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

	// Print end of program
	INFO_PRINT("exitProgram(): End of program, press enter to exit.\n");
	getchar();
	exit(0);
}

/**
 * This program is an introduction test to Neural Networks.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'image_upscaler_training' program.\n");
	atexit(exitProgram);

	// Try to load a neural network
	NeuralNetworkD *loaded_network = loadNeuralNetworkD(NEURAL_NETWORK_PATH, sigmoid);
	NeuralNetworkD network;
	if (loaded_network == NULL) {

		// Create a neural network using double as type
		WARNING_PRINT("main(): No neural network found, creating a new one.\n");
		int nb_neurons_per_layer[] = {256*256 + 1, 4096, 4096, 4096, 256*256};
		int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
		network = createNeuralNetworkD(nb_layers, nb_neurons_per_layer, 0.1, sigmoid);
	} else {
		INFO_PRINT("main(): Neural network found, using it.\n");
		network = *loaded_network;
		free(loaded_network);
	}

	// Print the neural network information
	printNeuralNetworkD(network);

	// Create an image list
	img_list_t img_list = img_list_new();

	// List all the images in the folder using a pipe
	char command[256];
	sprintf(command, "dir /b %s", IMAGES_PATH);
	FILE* pipe = popen(command, "r");
	ERROR_HANDLE_PTR_RETURN_INT(pipe, "main(): Error listing the images in the folder '%s'\n", IMAGES_PATH);
	char file_name[256];
	while (fgets(file_name, sizeof(file_name), pipe) != NULL) {

		// Remove the \n at the end of the file name
		file_name[strlen(file_name) - 1] = '\0';

		// Get the image path
		char image_path[256];
		sprintf(image_path, "%s%s", IMAGES_PATH, file_name);

		// Load the image & add it to the list
		image_t image;
		int code = image_load(image_path, &image);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error loading the image '%s'\n", image_path);
		code = img_list_insert(&img_list, image);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error inserting the image '%s' in the list\n", image_path);
	}
	pclose(pipe);

	// Print the number of images
	INFO_PRINT("main(): %d images found in the folder '%s'\n", img_list.size, IMAGES_PATH);

	// For each image, split it into 256x256 images
	img_list_t img_list_split = img_list_new();
	int code = img_list_split_by_size(&img_list, 256, &img_list_split);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error splitting the images\n");

	// Free the original image list & shuffle the splitted images
	img_list_free(&img_list);
	img_list_shuffle(&img_list_split);

	// Print the number of splitted images
	INFO_PRINT("main(): %d images splitted into 256x256 images\n", img_list_split.size);

	// Save the neural network to a file and another human readable file
	//saveNeuralNetworkD(network, NEURAL_NETWORK_PATH, 0);

	// Free the neural network & free private GPU buffers
	freeNeuralNetworkD(&network);
	stopNeuralNetworkGpuOpenCL();
	stopNeuralNetworkGpuBuffersOpenCL();

	// Final print and return
	INFO_PRINT("main(): End of program.\n");
	return 0;
}

