
#include <sys/types.h>

#include "../src/universal_utils.h"
#include "../src/math/sigmoid.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_cpu.h"
#include "../src/neural_network/training_gpu.h"
#include "../src/list/image_list.h"
#include "../src/utils/images.h"
#include "../src/st_benchmark.h"

#define NEURAL_NETWORK_PATH "bin/image_upscaler_network.bin"
#define IMAGES_PATH "images/"

NeuralNetworkD network;
int code;

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

	// Read all the buffers & Save the neural network to a file
	WARNING_HANDLE_INT(NeuralNetworkDReadAllBuffersGPU(&network), "exitProgram(): Error reading all the buffers\n");
	WARNING_HANDLE_INT(saveNeuralNetworkD(network, NEURAL_NETWORK_PATH, 0), "exitProgram(): Error saving the neural network\n");

	// Free the neural network & free private GPU buffers
	freeNeuralNetworkD(&network);
	stopNeuralNetworkGpuOpenCL();
	stopNeuralNetworkGpuBuffersOpenCL();

	// Print end of program
	INFO_PRINT("exitProgram(): End of program, press enter to exit.\n");
	getchar();
	exit(0);
}

/**
 * This program is an introduction to image upscaling using a neural network.
 * This neural network is trained to upscale images to a factor of x8.
 * 
 * Inputs are 1 value to indicate the multiplier and 128*128 values for the image.
 * Outputs are 128*128 values for the upscaled image.
 * 
 * So the neural network has 128*128*3 + 1 inputs and 128*128*3 outputs.
 * (3 because of the RGB channels)
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'image_upscaler_training' program.\n");
	atexit(exitProgram);

	// Try to load a neural network
	NeuralNetworkD *loaded_network = loadNeuralNetworkD(NEURAL_NETWORK_PATH, sigmoid);
	if (loaded_network == NULL) {

		// Create a neural network using double as type
		WARNING_PRINT("main(): No neural network found, creating a new one.\n");
		int nb_neurons_per_layer[] = {128*128*3 + 1, 8192, 8192, 8192, 128*128*3};
		int nb_layers = sizeof(nb_neurons_per_layer) / sizeof(int);
		network = createNeuralNetworkD(nb_layers, nb_neurons_per_layer, 0.1, sigmoid);
	} else {
		INFO_PRINT("main(): Neural network found, using it.\n");
		network = *loaded_network;
		free(loaded_network);
	}

	// Print the neural network information
	printNeuralNetworkD(network);

	// List all the images in the folder using a pipe
	char command[512];
	sprintf(command, "ls %s", IMAGES_PATH);
	FILE* pipe = popen(command, "r");
	ERROR_HANDLE_PTR_RETURN_INT(pipe, "main(): Error listing the images in the folder '%s'\n", IMAGES_PATH);
	char file_name[512];
	while (fgets(file_name, sizeof(file_name), pipe) != NULL) {

		// Remove the \n at the end of the file name
		file_name[strlen(file_name) - 1] = '\0';

		// Get the image path
		char image_path[512];
		sprintf(image_path, "%s%s", IMAGES_PATH, file_name);
		INFO_PRINT("main(): Image path: '%s'\n", image_path);

		// Create an image list
		img_list_t img_list = img_list_new();

		// Load the image
		image_t image;
		int code = image_load(image_path, &image);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error loading the image '%s'\n", image_path);
		code = img_list_insert(&img_list, image);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error inserting the image '%s' in the list\n", image_path);

		// Repare the terminal colors because of image loading
		#ifdef _WIN32
			system("powershell -command \"\"");
		#endif

		// For each image, split it into 128x128 images (number of neurons in the output layer)
		img_list_t img_list_split = img_list_new();
		code = img_list_split_by_size(&img_list, 128, &img_list_split);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error splitting the images\n");

		// Free the original image list & shuffle the splitted images
		img_list_free(&img_list);
		img_list_shuffle(&img_list_split);

		// Print the number of splitted images
		INFO_PRINT("main(): %d splitted images\n", img_list_split.size);

		// For each splitted image, train the neural network
		img_list_elt_t *current_elt = img_list_split.head;
		int img_number = 0;
		while (current_elt != NULL) {

			// Print the current image
			char buffer[1024];
			char benchmark_name[512];
			sprintf(benchmark_name, "NeuralNetworkDtrain (GPU) image %d/%d (%dx%d)", ++img_number, img_list_split.size, current_elt->image.width, current_elt->image.height);
			ST_BENCHMARK_SOLO_COUNT(buffer,
				{

					// Create a list of resized images (from 128x128 to 16x16 pixel per pixel (7 images))
					img_list_t img_list_resized = img_list_new();
					for (int size = current_elt->image.width; size > 16; size -= 16) {
						
						// Resize the image & add it to the list
						image_t resized_image;
						int code = image_resize(current_elt->image, size, size, &resized_image);
						ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error resizing the image %d/%d to size %dx%d\n", img_number, img_list_split.size, size, size);
						code = img_list_insert(&img_list_resized, resized_image);
						ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error inserting the resized image %d/%d to size %dx%d in the list\n", img_number, img_list_split.size, size, size);
					}

					// Train the neural network with the resized images
					int code = NeuralNetworkDtrainFromImageListGPU(&network, img_list_resized, current_elt->image, 0);
					ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error training the neural network with the image %d/%d\n", img_number, img_list_split.size);

					// Free the resized images list & the excepted output array & go next image
					img_list_free(&img_list_resized);
					current_elt = current_elt->next;
				},
				benchmark_name, 1
			);
			PRINTER(buffer);
		}

		// Free the splitted images list
		img_list_free(&img_list_split);

		// Read all the buffers
		code = NeuralNetworkDReadAllBuffersGPU(&network);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error reading all the buffers\n");

		// Save the neural network to a file and another human readable file
		saveNeuralNetworkD(network, NEURAL_NETWORK_PATH, 0);
	}
	pclose(pipe);

	// Read all the buffers
	NeuralNetworkDReadAllBuffersGPU(&network);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error reading all the buffers\n");

	// Free the neural network & free private GPU buffers
	freeNeuralNetworkD(&network);
	stopNeuralNetworkGpuOpenCL();
	stopNeuralNetworkGpuBuffersOpenCL();

	// Final print and return
	INFO_PRINT("main(): End of program.\n");
	return 0;
}

