
#include "images.h"
#include "../universal_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../libs/stb_image.h"
#include "../../libs/stb_image_write.h"

/**
 * @brief Allocate the memory for the image structure
 * and link the data to the flat_data.
 * 
 * @param image			Pointer to the image structure
 * 
 * @return int			0 if no error occurred, -1 otherwise
 */
int image_structure_allocations(image_t* image) {

	// Allocate the image data (3D array: [height][width][channels])
	image->data = malloc(image->height * sizeof(unsigned char**));
	ERROR_HANDLE_PTR_RETURN_INT(image->data, "image_structure_allocations(): Error allocating the image data\n");

	// For each row, allocate the columns
	for (int i = 0; i < image->height; i++) {

		// Allocate the columns
		image->data[i] = malloc(image->width * sizeof(unsigned char*));
		ERROR_HANDLE_PTR_RETURN_INT(image->data[i], "image_structure_allocations(): Error allocating the image data\n");

		// For each column, link the data to the flat_data
		for (int j = 0; j < image->width; j++)
			image->data[i][j] = &image->flat_data[(i * image->width + j) * image->channels];
	}

	// Return
	return 0;
}

/**
 * @brief Load an image from a file, fills the width, height and channels variables
 * and returns a pointer to the image data.
 * 
 * @param file_name		Name of the file
 * @param image			Pointer to the image structure
 * 
 * @return int			0 if no error occurred, -1 otherwise
 */
int image_load(const char* file_name, image_t* image) {

	// Initialize the image structure
	memset(image, 0, sizeof(image_t));

	// Open the file for reading in binary mode
	FILE* file = fopen(file_name, "rb");
	ERROR_HANDLE_PTR_RETURN_INT(file, "image_load(): Error opening the file '%s'\n", file_name);

	// Load the image
	image->flat_data = stbi_load_from_file(file, &image->width, &image->height, &image->channels, 0);
	ERROR_HANDLE_PTR_RETURN_INT(image->flat_data, "image_load(): Error loading the image '%s'\n", file_name);

	// Allocate the memory for the image structure
	int code = image_structure_allocations(image);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_load(): Error allocating the image structure\n");

	// Close the file & return
	fclose(file);
	return 0;
}

/**
 * @brief Free the memory allocated by the image structure.
 * 
 * @param image			Pointer to the image structure
 * 
 * @return void
 */
void image_free(image_t* image) {
	free(image->flat_data);
	for (int i = 0; i < image->height; i++)
		free(image->data[i]);
	free(image->data);
	image->width = image->height = image->channels = 0;
	image->flat_data = NULL;
	image->data = NULL;
}

/**
 * @brief Function used by stbi_write_png_to_func() & stbi_write_jpg_to_func()
 * to write data to a file.
 * 
 * @param context		Pointer to the file
 * @param data			Pointer to the data to write
 * @param size			Size of the data to write
 * 
 * @return void
 */
void img_write_func(void* context, void* data, int size) {
	fwrite(data, 1, size, (FILE*)context);
}

/**
 * @brief Save an image to a PNG file.
 * 
 * @param file_name		Name of the file
 * @param image			Image to save
 * 
 * @return int			0 if no error occurred, -1 otherwise
 */
int image_save_png(const char* file_name, image_t image) {
	
	// Open the file for writing in binary mode
	FILE* file = fopen(file_name, "wb");
	ERROR_HANDLE_PTR_RETURN_INT(file, "image_save_png(): Error opening the file '%s'\n", file_name);

	// Save the image
	int code = stbi_write_png_to_func(img_write_func, file, image.width, image.height, image.channels, image.flat_data, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_save_png(): Error saving the image\n");

	// Close the file & return
	fclose(file);
	return 0;
}

/**
 * @brief Save an image to a JPG file.
 * 
 * @param file_name		Name of the file
 * @param image			Pointer to the image data
 * @param width			Width of the image
 * @param height		Height of the image
 * @param channels		Channels of the image (1 for grayscale, 3 for RGB)
 * @param quality		Quality of the JPG image (0-100)
 * 
 * @return int			0 if no error occurred, -1 otherwise
 */
int image_save_jpg(const char* file_name, image_t image, int quality) {
	
	// Open the file for writing in binary mode
	FILE* file = fopen(file_name, "wb");
	ERROR_HANDLE_PTR_RETURN_INT(file, "image_save_jpg(): Error opening the file '%s'\n", file_name);

	// Save the image
	int code = stbi_write_jpg_to_func(img_write_func, file, image.width, image.height, image.channels, image.flat_data, quality);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_save_jpg(): Error saving the image\n");

	// Close the file & return
	fclose(file);
	return 0;
}


/**
 * @brief Split an image into multiple images of the specified size.
 * 
 * @param image			Image to split
 * @param split_size	Size of the split images (width & height)
 * @param images		Pointer to the array of images
 * @param nb_images		Pointer to the number of images
 * 
 * @return int			0 if no error occurred, -1 otherwise
 */
int image_split_by_size(image_t image, int split_size, image_t** images, int* nb_images) {
	
	// Check the parameters
	ERROR_HANDLE_PTR_RETURN_INT(image.data, "image_split_by_size(): Parameter Image data is NULL\n");
	ERROR_HANDLE_PTR_RETURN_INT(images, "image_split_by_size(): Parameter images is NULL\n");
	ERROR_HANDLE_PTR_RETURN_INT(nb_images, "image_split_by_size(): Parameter nb_images is NULL\n");

	// Calculate the number of images
	*nb_images = (image.width / split_size) * (image.height / split_size);
	*images = malloc(*nb_images * sizeof(image_t));
	ERROR_HANDLE_PTR_RETURN_INT(*images, "image_split_by_size(): Error allocating the images array\n");
	memset(*images, 0, *nb_images * sizeof(image_t));

	// Split the image by creating new images
	for (int image_index = 0; image_index < *nb_images; image_index++) {

		// Initialize the image structure
		image_t* current_image = &(*images)[image_index];
		current_image->width = split_size;
		current_image->height = split_size;
		current_image->channels = image.channels;
		current_image->flat_data = malloc(split_size * split_size * image.channels * sizeof(unsigned char));
		ERROR_HANDLE_PTR_RETURN_INT(current_image->flat_data, "image_split_by_size(): Error allocating the image data\n");
		memset(current_image->flat_data, 0, split_size * split_size * image.channels * sizeof(unsigned char));

		// Allocate the memory for the image structure
		int code = image_structure_allocations(current_image);
		ERROR_HANDLE_INT_RETURN_INT(code, "image_split_by_size(): Error allocating the image structure\n");

		// Copy the image data to the new image without going out of bounds
		int x = (image_index % (image.width / split_size)) * split_size;
		int y = (image_index / (image.width / split_size)) * split_size;
		for (int i = 0; i < split_size; i++)
			for (int j = 0; j < split_size; j++)
				for (int k = 0; k < image.channels; k++)
					current_image->data[i][j][k] = image.data[y + i][x + j][k];
	}

	// Return
	return 0;
}