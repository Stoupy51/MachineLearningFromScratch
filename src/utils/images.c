
#include "images.h"
#include "../universal_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../libs/stb_image.h"
#include "../../libs/stb_image_write.h"
#include "../../libs/stb_image_resize.h"

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

/**
 * @brief Merge multiple images into one image.
 * 
 * @param images_array		Array of images to merge
 * @param nb_images		Number of images to merge
 * @param image			Pointer to the image structure to fill the data 
 * 						(width & height must be set but not the rest)
 * 
 * @return int			0 if no error occurred, -1 otherwise
 */
int image_merge(image_t* images_array, int nb_images, image_t* image) {

	// Check the parameters
	ERROR_HANDLE_PTR_RETURN_INT(images_array, "image_merge(): Parameter images is NULL\n");
	ERROR_HANDLE_PTR_RETURN_INT(image, "image_merge(): Parameter image is NULL\n");

	// Initialize the image structure
	image->channels = images_array[0].channels;
	image->flat_data = malloc(image->width * image->height * image->channels * sizeof(unsigned char));
	ERROR_HANDLE_PTR_RETURN_INT(image->flat_data, "image_merge(): Error allocating the image data\n");
	memset(image->flat_data, 0, image->width * image->height * image->channels * sizeof(unsigned char));

	// Allocate the memory for the image structure
	int code = image_structure_allocations(image);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_merge(): Error allocating the image structure\n");

	// Merge the images
	for (int image_index = 0; image_index < nb_images; image_index++) {

		// Copy the image data to the new image without going out of bounds
		int x = (image_index % (image->width / images_array[image_index].width)) * images_array[image_index].width;
		int y = (image_index / (image->width / images_array[image_index].width)) * images_array[image_index].height;
		for (int i = 0; i < images_array[image_index].height; i++)
			for (int j = 0; j < images_array[image_index].width; j++)
				for (int k = 0; k < image->channels; k++)
					image->data[y + i][x + j][k] = images_array[image_index].data[i][j][k];
	}

	// Return
	return 0;
}

/**
 * @brief Resize an image.
 * 
 * @param image				Image to resize
 * @param new_width			New width of the image
 * @param new_height		New height of the image
 * @param resized_image		Pointer to the resized image
 * 
 * @return int				0 if no error occurred, -1 otherwise
 */
int image_resize(image_t image, int new_width, int new_height, image_t* resized_image) {
	
	// Check the parameters
	ERROR_HANDLE_PTR_RETURN_INT(image.data, "image_resize(): Parameter Image data is NULL\n");
	ERROR_HANDLE_PTR_RETURN_INT(resized_image, "image_resize(): Parameter resized_image is NULL\n");

	// Initialize the image structure
	resized_image->width = new_width;
	resized_image->height = new_height;
	resized_image->channels = image.channels;
	resized_image->flat_data = malloc(new_width * new_height * image.channels * sizeof(unsigned char));
	ERROR_HANDLE_PTR_RETURN_INT(resized_image->flat_data, "image_resize(): Error allocating the image data\n");
	memset(resized_image->flat_data, 0, new_width * new_height * image.channels * sizeof(unsigned char));

	// Allocate the memory for the image structure
	int code = image_structure_allocations(resized_image);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_resize(): Error allocating the image structure\n");

	// Resize the image
	code = stbir_resize_uint8(image.flat_data, image.width, image.height, 0, resized_image->flat_data, new_width, new_height, 0, image.channels);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_resize(): Error resizing the image\n");

	// Return
	return 0;
}

/**
 * @brief Convert an image to a double array.
 * 
 * @param image			Image to convert
 * @param size	Size of the array to allocate (-1 to allocate the exact size)
 * @param offset		Offset to apply to the array
 * 
 * @return double*		Pointer to the double array
 */
double* image_to_double_array(image_t image, int size, int offset) {
	
	// Allocate the array
	int original_size = image.width * image.height * image.channels;
	int array_size = (size < 1) ? original_size : size;
	double* array = malloc(array_size * sizeof(double));
	ERROR_HANDLE_PTR_RETURN_NULL(array, "image_to_double_array(): Error allocating the array\n");
	memset(array, 0, array_size * sizeof(double));

	// Fill the array
	int final = original_size + offset;
	for (int i = offset; i < final; i++)
		array[i] = (double)image.flat_data[i] / 255.0;

	// Return
	return array;
}

