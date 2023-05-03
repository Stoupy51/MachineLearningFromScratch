
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

#include "utils.h"

struct opencl_context_t {
	cl_device_id device_id;				// ID of the GPU device
	cl_context context;					// Context of the GPU device
	cl_command_queue command_queue;		// Command queue of the GPU device
};

/**
 * @brief This function initialize OpenCL, get a GPU device,
 * create a context and a command queue.
 * 
 * @param type_of_device Type of device to use (CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_CPU)
 * 
 * @return opencl_context_t		The context of the GPU device
 */
struct opencl_context_t setupOpenCL(int type_of_device) {

	// Initialize structure
	struct opencl_context_t oc;

	// Get a GPU device
	clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &oc.device_id, NULL);

	// Create a context
	oc.context = clCreateContext(NULL, 1, &oc.device_id, NULL, NULL, NULL);

	// Create a command queue
	oc.command_queue = clCreateCommandQueue(oc.context, oc.device_id, 0, NULL);

	// Return the context
	return oc;
}


/**
 * @brief This function read a file and return its content as a string.
 * 
 * @param filename Name of the file to read
 * 
*/
char* readEntireFile(char* path) {
	
	// Open the file
	int fd = open(path, O_RDONLY);
	if (fd == -1) {
		ERROR_PRINT("readEntireFile(): Cannot open file %s\n", path);
		return NULL;
	}

	// Get the size of the file
	int size = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);

	// Allocate memory for the file content
	char* content = malloc(sizeof(char) * (size + 1));
	if (content == NULL) {
		ERROR_PRINT("readEntireFile(): Cannot allocate memory for file %s\n", path);
		return NULL;
	}

	// Read the file
	memset(content, '\0', size + 1);
	int read_size = read(fd, content, size);
	if (read_size != size) {
		ERROR_PRINT("readEntireFile(): Cannot read file %s\n", path);
		return NULL;
	}

	// Close the file
	close(fd);

	// Return the file content
	return content;
}




