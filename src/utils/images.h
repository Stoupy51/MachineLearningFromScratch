
#ifndef __IMAGES_H__
#define __IMAGES_H__

// Functions
unsigned char* image_load(const char* file_name, int* width, int* height, int* channels);
int image_save_png(const char* file_name, unsigned char* image, int width, int height, int channels);
int image_save_jpg(const char* file_name, unsigned char* image, int width, int height, int channels, int quality);


#endif

