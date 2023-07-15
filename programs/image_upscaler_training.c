
#include <sys/types.h>

#include "../src/universal_utils.h"
#include "../src/math/sigmoid.h"
#include "../src/neural_network/neural_utils.h"
#include "../src/neural_network/training_gpu.h"
#include "../src/list/image_list.h"
#include "../src/utils/images.h"
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
	NeuralNetworkD network;
	if (loaded_network == NULL) {

		// Create a neural network using double as type
		WARNING_PRINT("main(): No neural network found, creating a new one.\n");
		int nb_neurons_per_layer[] = {128*128*3 + 1, 4096, 4096, 4096, 128*128*3};
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

		// Load the image & add it to the list
		image_t image;
		int code = image_load(image_path, &image);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error loading the image '%s'\n", image_path);
		code = img_list_insert(&img_list, image);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error inserting the image '%s' in the list\n", image_path);
	}
	pclose(pipe);

	// Repare the terminal colors because of image loading
	#ifdef _WIN32
		system("powershell -command \"\"");
	#endif

	// Print the number of images
	INFO_PRINT("main(): %d images found in the folder '%s'\n", img_list.size, IMAGES_PATH);

	// For each image, split it into 128x128 images (number of neurons in the output layer)
	img_list_t img_list_split = img_list_new();
	int code = img_list_split_by_size(&img_list, 128, &img_list_split);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error splitting the images\n");

	// Free the original image list & shuffle the splitted images
	img_list_free(&img_list);
	img_list_shuffle(&img_list_split);

	// Print the number of splitted images
	INFO_PRINT("main(): %d splitted images\n", img_list_split.size);

	// For each splitted image, train the neural network
	img_list_elt_t *current_elt = img_list_split.head;
	int i = 0;
	while (current_elt != NULL) {

		// Print the current image
		INFO_PRINT("main(): Training image %d/%d\n", ++i, img_list_split.size);

		// Get the image
		image_t *image = &current_elt->image;

		// Create a list of resized images (from 128x128 to 16x16 pixel per pixel (7 images))
		img_list_t img_list_resized = img_list_new();
		for (int size = image->width; size > 16; size -= 16) {
			
			// Resize the image & add it to the list
			image_t resized_image;
			int code = image_resize(*image, size, size, &resized_image);
			ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error resizing the image %d/%d to size %dx%d\n", i, img_list_split.size, size, size);
			code = img_list_insert(&img_list_resized, resized_image);
			ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error inserting the resized image %d/%d to size %dx%d in the list\n", i, img_list_split.size, size, size);
		}

		// Prepare the excepted output array
		double *excepted_output = (double*)malloc(network.output_layer->nb_neurons * sizeof(double));
		for (int i = 0; i < network.output_layer->nb_neurons; i++)
			excepted_output[i] = (double)image->flat_data[i] / 255.0;

		// For each resized image, train the neural network
		img_list_elt_t *current_elt_resized = img_list_resized.head;
		int j = 0;
		while (current_elt_resized != NULL) {

			// Prepare the input array
			double *input = (double*)malloc(network.input_layer->nb_neurons * sizeof(double));
			input[0] = (double)current_elt_resized->image.width / (double)image->width;	// Ratio of the resized image
			for (int i = 0; i < network.input_layer->nb_neurons - 1; i++)
				input[i + 1] = (double)current_elt_resized->image.flat_data[i] / 255.0;

			// Benchmark name
			char benchmark_name[512];
			sprintf(benchmark_name, "NeuralNetworkDtrain (GPU) - image %d/%d - resized image %d/%d", i, img_list_split.size, ++j, img_list_resized.size);

			// Benchmark the GPU training
			char buffer[1024];
			ST_BENCHMARK_SOLO_COUNT(buffer,
				NeuralNetworkDtrainGPU(&network, input, excepted_output, 0),
				benchmark_name, 1
			);
			PRINTER(buffer);

			// Free the input array & go next resized image
			free(input);
			current_elt_resized = current_elt_resized->next;
		}

		// Free the resized images list & the excepted output array & go next image
		img_list_free(&img_list_resized);
		free(excepted_output);
		current_elt = current_elt->next;
	}

	// Free the splitted images list
	img_list_free(&img_list_split);

	// Read all the buffers
	code = NeuralNetworkDReadAllBuffersGPU(&network);
	ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error reading all the buffers\n");

	// Save the neural network to a file and another human readable file
	saveNeuralNetworkD(network, NEURAL_NETWORK_PATH, 0);

	// Free the neural network & free private GPU buffers
	freeNeuralNetworkD(&network);
	stopNeuralNetworkGpuOpenCL();
	stopNeuralNetworkGpuBuffersOpenCL();

	// Final print and return
	INFO_PRINT("main(): End of program.\n");
	return 0;
}

