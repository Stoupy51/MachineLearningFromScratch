
#include "image_list.h"
#include "../universal_utils.h"

/**
 * @brief Returns a new image list
 * 
 * @return img_list_t
 */
img_list_t img_list_new() {
	img_list_t list;
	list.head = NULL;
	list.size = 0;
	return list;
}

/**
 * @brief Inserts an image in an image list
 * 
 * @param list		Pointer to the image list
 * @param image		Image to insert
 * 
 * @return int		0 if success, -1 if error
 */
int img_list_insert(img_list_t *list, image_t image) {
	img_list_elt_t *elt = malloc(sizeof(img_list_elt_t));
	ERROR_HANDLE_PTR_RETURN_INT(elt, "img_list_insert(): Error allocating the image list element\n");
	elt->image = image;
	elt->next = list->head;
	list->head = elt;
	list->size++;
	return 0;
}

/**
 * @brief Frees an image list and its elements
 * 
 * @param list		Pointer to the image list
 */
void img_list_free(img_list_t *list) {
	img_list_elt_t *current_elt = list->head;
	while (current_elt != NULL) {
		img_list_elt_t *next_elt = current_elt->next;
		image_free(&current_elt->image);
		free(current_elt);
		current_elt = next_elt;
	}
	list->head = NULL;
	list->size = 0;
}


/**
 * @brief Shuffles an image list
 * 
 * @param list		Pointer t
 */
void img_list_shuffle(img_list_t *list) {
	
	// Create a new image list
	img_list_t shuffled_list = img_list_new();
	shuffled_list.size = list->size;

	// While the original list is not empty,
	while (list->size > 0) {

		// Move a random element from the original list to the new list
		int random_index = rand() % list->size;
		img_list_elt_t *current_elt = list->head;
		img_list_elt_t *previous_elt = NULL;
		for (int i = 0; i < random_index; i++) {
			previous_elt = current_elt;
			current_elt = current_elt->next;
		}
		if (previous_elt == NULL)
			list->head = current_elt->next;
		else
			previous_elt->next = current_elt->next;
		
		// Add the element to the new list
		current_elt->next = shuffled_list.head;
		shuffled_list.head = current_elt;
		list->size--;
	}

	// Copy the new list to the original list
	*list = shuffled_list;
}

/**
 * @brief Splits each image of an image list into images of a given size
 * 
 * @param list				Pointer to the image list
 * @param split_size		Size of the splitted images
 * @param splitted_list		Pointer to the splitted image list to fill
 * 
 * @return int				0 if success, -1 if error
 */
int img_list_split_by_size(img_list_t *list, int split_size, img_list_t *splitted_list) {
	
	// For each image in the list,
	img_list_elt_t *current_elt = list->head;
	while (current_elt != NULL) {

		// Split the image
		image_t *images;
		int nb_images;
		int code = image_split_by_size(current_elt->image, split_size, &images, &nb_images);
		ERROR_HANDLE_INT_RETURN_INT(code, "img_list_split_by_size(): Error splitting the image\n");

		// For each splitted image, add it to the splitted list
		for (int i = 0; i < nb_images; i++) {
			code = img_list_insert(splitted_list, images[i]);
			ERROR_HANDLE_INT_RETURN_INT(code, "img_list_split_by_size(): Error inserting the splitted image in the list\n");
		}

		// Free the splitted images array
		free(images);

		// Go to the next image
		current_elt = current_elt->next;
	}

	// Return
	return 0;
}

/**
 * @brief Merges an image list into a single image
 * 
 * @param list_to_merge		Pointer to the image list to merge
 * @param merged_image		Pointer to the merged image to fill
 * 
 * @return int				0 if success, -1 if error
 */
int img_list_merge(img_list_t *list_to_merge, image_t *merged_image) {

	// Create a single list of images
	image_t *images = malloc(list_to_merge->size * sizeof(image_t));
	ERROR_HANDLE_PTR_RETURN_INT(images, "img_list_merge(): Error allocating the images array\n");
	img_list_elt_t *current_elt = list_to_merge->head;
	for (int i = 0; i < list_to_merge->size; i++) {
		images[i] = current_elt->image;
		current_elt = current_elt->next;
	}

	// Merge the images
	int code = image_merge(images, list_to_merge->size, merged_image);
	ERROR_HANDLE_INT_RETURN_INT(code, "img_list_merge(): Error merging the images\n");

	// Free the images array
	free(images);

	// Return
	return 0;
}

