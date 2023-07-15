
#ifndef __IMAGES_H__
#define __IMAGES_H__

//
typedef struct image_t {
	int width;
	int height;
	int channels;
	unsigned char* data;
} image_t;

// Functions
int image_load(const char* file_name, image_t* image);
int image_save_png(const char* file_name, image_t image);
int image_save_jpg(const char* file_name, image_t image, int quality);


#endif

