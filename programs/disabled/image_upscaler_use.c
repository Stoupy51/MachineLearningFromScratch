
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
#define IMAGES_INPUT_PATH "images/input/"
#define IMAGES_OUTPUT_PATH "images/output/"

NeuralNetworkD network;
int code;

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

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
	ERROR_HANDLE_PTR_RETURN_INT(loaded_network, "main(): Error loading the neural network\n");

	// If the neural network is loaded, use it
	network = *loaded_network;

	// Print the neural network information
	printNeuralNetworkD(network);

	// List all the images in the folder using a pipe
	char command[512];
	sprintf(command, "ls %s", IMAGES_INPUT_PATH);
	FILE* pipe = popen(command, "r");
	ERROR_HANDLE_PTR_RETURN_INT(pipe, "main(): Error listing the images in the folder '%s'\n", IMAGES_INPUT_PATH);
	char file_name[512];
	while (fgets(file_name, sizeof(file_name), pipe) != NULL) {

		// Remove the \n at the end of the file name
		file_name[strlen(file_name) - 1] = '\0';

		// Get the image path
		char image_path[512];
		sprintf(image_path, "%s%s", IMAGES_INPUT_PATH, file_name);
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

		// Shuffle the splitted images
		img_list_shuffle(&img_list_split);

		// Print the number of splitted images
		INFO_PRINT("main(): %d splitted images\n", img_list_split.size);

		// Prepare the new list of images
		img_list_t img_list_new_size = img_list_new();
		double multiplier = 2.0;

		// For each splitted image, active the neural network
		img_list_elt_t *current_elt = img_list_split.head;
		int i = 0;
		while (current_elt != NULL) {

			// Prepare the input array for the neural network
			double *input = image_to_double_array(current_elt->image, network.input_layer->nb_neurons, 1);
			input[0] = multiplier;

			char benchmark_buffer[1024];
			char benchmark_name[512];
			sprintf(benchmark_name, "NeuralNetworkDfeedForward : %d / %d", ++i, img_list_split.size);
			ST_BENCHMARK_SOLO_COUNT(benchmark_buffer,
			{

			// Feed forward the neural network
			code = NeuralNetworkDfeedForwardGPU(&network, input, 1);
			ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error feeding forward the neural network\n");
			// NeuralNetworkDfeedForwardCPU(&network, input);
			},
			benchmark_name, 1);
			PRINTER(benchmark_buffer);

			// Get the output of the neural network
			image_t output;
			code = image_load_empty(&output, current_elt->image.width * multiplier, current_elt->image.height * multiplier, current_elt->image.channels);
			ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error loading the output image\n");
			int total_size = output.width * output.height * output.channels;
			for (int i = 0; i < total_size; i++)
				output.flat_data[i] = (unsigned char)(network.output_layer->activations_values[i] * 255.0);

			// Add the output image to the new list
			code = img_list_insert(&img_list_new_size, output);
			ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error inserting the output image in the list\n");

			// Free the input array
			free(input);

			// Go to the next image
			current_elt = current_elt->next;
		}DEBUG_PRINT("main(): End of loop\n");

		// Merge the images
		image_t merged_image;
		merged_image.width = image.width * multiplier;
		merged_image.height = image.height * multiplier;
		code = img_list_merge(&img_list_new_size, &merged_image);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error merging the images\n");
		DEBUG_PRINT("main(): End of merge\n");

		// Save the image
		char output_path[512];
		sprintf(output_path, "%s%s", IMAGES_OUTPUT_PATH, file_name);
		code = image_save_jpg(output_path, merged_image, 100);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error saving the image '%s'\n", output_path);

		// Frees
		img_list_free(&img_list_split);
		img_list_free(&img_list_new_size);
		img_list_free(&img_list);
		image_free(&merged_image);
	}
	pclose(pipe);

	// Final print and return
	INFO_PRINT("main(): End of program.\n");
	return 0;
}

