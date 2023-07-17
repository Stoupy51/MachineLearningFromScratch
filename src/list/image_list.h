
#ifndef __IMAGE_LIST_H__
#define __IMAGE_LIST_H__

#include "../utils/images.h"

typedef struct img_list_elt_t {
	image_t image;
	struct img_list_elt_t *next;
} img_list_elt_t;

typedef struct img_list_t {
	img_list_elt_t *head;
	int size;
} img_list_t;

img_list_t img_list_new();
int img_list_insert(img_list_t *list, image_t image);
void img_list_free(img_list_t *list);

void img_list_shuffle(img_list_t *list);
int img_list_split_by_size(img_list_t *list, int split_size, img_list_t *splitted_list);
int img_list_merge(img_list_t *list_to_merge, image_t *merged_image);

#endif

