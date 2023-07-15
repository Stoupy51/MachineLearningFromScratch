
#include "images.h"
#include "../universal_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../libs/stb_image.h"
#include "../../libs/stb_image_write.h"

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
	image->data = stbi_load_from_file(file, &image->width, &image->height, &image->channels, 0);
	ERROR_HANDLE_PTR_RETURN_INT(image->data, "image_load(): Error loading the image '%s'\n", file_name);

	// Close the file & return
	fclose(file);
	return 0;
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
	ERROR_HANDLE_PTR_RETURN_INT(file, "image_save_png(): Error opening the file '%s'.\n", file_name);

	// Save the image
	int code = stbi_write_png_to_func(img_write_func, file, image.width, image.height, image.channels, image.data, 0);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_save_png(): Error saving the image.\n");

	// Close the file
	fclose(file);

	// Return 0
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
	ERROR_HANDLE_PTR_RETURN_INT(file, "image_save_jpg(): Error opening the file '%s'.\n", file_name);

	// Save the image
	int code = stbi_write_jpg_to_func(img_write_func, file, image.width, image.height, image.channels, image.data, quality);
	ERROR_HANDLE_INT_RETURN_INT(code, "image_save_jpg(): Error saving the image.\n");

	// Close the file
	fclose(file);

	// Return 0
	return 0;
}

