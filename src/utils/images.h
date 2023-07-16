
#ifndef __IMAGES_H__
#define __IMAGES_H__

typedef struct image_t {
	int width;
	int height;
	int channels;
	unsigned char* flat_data;
	unsigned char*** data;		// 3D array: data[height][width][channels]
} image_t;

int image_structure_allocations(image_t* image);
int image_load(const char* file_name, image_t* image);
void image_free(image_t* image);
int image_save_png(const char* file_name, image_t image);
int image_save_jpg(const char* file_name, image_t image, int quality);

int image_split_by_size(image_t image, int split_size, image_t** images, int* nb_images);
int image_merge(image_t* images_array, int nb_images, image_t* image);
int image_resize(image_t image, int new_width, int new_height, image_t* resized_image);
double* image_to_double_array(image_t image);


#endif

