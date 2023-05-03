
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <CL/cl.h>

// Defines for colors
#define RED "\033[0;31m"
#define GREEN "\033[0;32m"
#define YELLOW "\033[0;33m"
#define BLUE "\033[0;34m"
#define MAGENTA "\033[0;35m"
#define CYAN "\033[0;36m"
#define RESET "\033[0m"

// Defines for the different levels of debug output
#define INFO_LEVEL 1
#define WARNING_LEVEL 2
#define ERROR_LEVEL 4
// Change the following line to change the debug level
#define DEBUG_LEVEL (INFO_LEVEL | WARNING_LEVEL | ERROR_LEVEL)

// Utils defines to check debug level
#define IS_INFO_LEVEL (DEBUG_LEVEL & INFO_LEVEL)
#define IS_WARNING_LEVEL (DEBUG_LEVEL & WARNING_LEVEL)
#define IS_ERROR_LEVEL (DEBUG_LEVEL & ERROR_LEVEL)

// Utils defines to print debug messages
#define INFO_PRINT(...) if (IS_INFO_LEVEL) { printf(GREEN "[INFO] " RESET __VA_ARGS__); }
#define WARNING_PRINT(...) if (IS_WARNING_LEVEL) { printf(RED "[WARNING] " RESET __VA_ARGS__); }
#define ERROR_PRINT(...) if (IS_ERROR_LEVEL) { printf(RED "[ERROR] "RESET __VA_ARGS__); }


// Struct to store the OpenCL context
struct opencl_context_t {
	cl_device_id device_id;				// ID of the GPU device
	cl_context context;					// Context of the GPU device
	cl_command_queue command_queue;		// Command queue of the GPU device
};

struct opencl_context_t setupOpenCL(int type_of_device);
char* readEntireFile(char* path);




#endif

