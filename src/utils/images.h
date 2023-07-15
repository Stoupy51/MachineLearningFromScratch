
#ifndef __IMAGES_H__
#define __IMAGES_H__

typedef struct image_t {
	int width;
	int height;
	int channels;
	unsigned char* flat_data;
	unsigned char*** data;		// 3D array: data[height][width][channels]
} image_t;

int image_load(const char* file_name, image_t* image);
int image_save_png(const char* file_name, image_t image);
int image_save_jpg(const char* file_name, image_t image, int quality);

int image_split_by_size(image_t image, int split_size, image_t** images, int* nb_images);


#endif

