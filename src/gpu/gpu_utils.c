
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>

#include "../utils.h"
#include "gpu_utils.h"

/**
 * @brief This function returns the error string
 * corresponding to the given OpenCL error code.
 * 
 * @param error		The OpenCL error code
 * 
 * @return char*	The error string
 */
const char* getOpenCLErrorString(cl_int error) {

	// Switch on the error code
	switch(error) {

		// Run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// Compile-Time Errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// Extension Errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
	}
}


/**
 * @brief This function prints the program build log
 * to the output depending on the mode specified.
 * 
 * @param program		The program to print the build log for
 * @param device_id		The device ID to print the build log for
 * @param mode			The mode to print the build log in (INFO_LEVEL, WARNING_LEVEL, ERROR_LEVEL)
 * @param prefix		The prefix to print before the build log
 * 
 * @return int	0 if success, -1 otherwise
 */
int printProgramBuildLog(cl_program program, cl_device_id device_id, int mode, char* prefix) {

	// Get the build log size
	size_t log_size;
	if (clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) != CL_SUCCESS) {
		ERROR_PRINT("printProgramBuildLog(): failed to get the build log size\n");
		return -1;
	}

	// Get the build log
	char *log = (char *) malloc(log_size);
	if (log == NULL) {
		ERROR_PRINT("printProgramBuildLog(): failed to allocate memory for the build log\n");
		return -1;
	}
	if (clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL) != CL_SUCCESS) {
		ERROR_PRINT("printProgramBuildLog(): failed to get the build log\n");
		return -1;
	}

	// Print the build log
	if (prefix == NULL) prefix = "";
	switch (mode) {
		case INFO_LEVEL: INFO_PRINT("%sBuild log:\n%s\n", prefix, log); break;
		case WARNING_LEVEL: WARNING_PRINT("%sBuild log:\n%s\n", prefix, log); break;
		case ERROR_LEVEL: ERROR_PRINT("%sBuild log:\n%s\n", prefix, log); break;
	}

	// Free the build log and return
	free(log);
	return 0;
}


/**
 * @brief This function initialize OpenCL, get a GPU device,
 * create a context and a command queue.
 * 
 * @param type_of_device Type of device to use (CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_CPU)
 * 
 * @return opencl_context_t		The context of the GPU device
 */
struct opencl_context_t setupOpenCL(cl_device_type type_of_device) {

	// Initialize structure
	struct opencl_context_t oc;

	// Get a platform
	cl_int code = clGetPlatformIDs(1, &oc.platform_id, NULL);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("setupOpenCL(): Cannot get a platform with code %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Get a device
	code = clGetDeviceIDs(oc.platform_id, type_of_device, 1, &oc.device_id, NULL);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("setupOpenCL(): Cannot get a device with code %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Create a context
	oc.context = clCreateContext(NULL, 1, &oc.device_id, NULL, NULL, NULL);
	if (oc.context == NULL) {
		ERROR_PRINT("setupOpenCL(): Cannot create a context\n");
		exit(EXIT_FAILURE);
	}

	// Create a command queue
	oc.command_queue = clCreateCommandQueueWithProperties(oc.context, oc.device_id, NULL, NULL);
	if (oc.command_queue == NULL) {
		ERROR_PRINT("setupOpenCL(): Cannot create a command queue\n");
		exit(EXIT_FAILURE);
	}

	// Return the context
	return oc;
}


/**
 * @brief This function prints information about the device specified such as
 * - Device name
 * - Device version
 * - Driver version
 * - OpenCL C version
 * - Max memory allocation size
 * - Global memory size
 * - Local memory size
 * - Max constant buffer size
 * 
 * @param device_id		The device to print information about
 * 
 * @return void
 */
void printDeviceInfo(cl_device_id device_id) {

	// Setup string buffer
	#define STR_BUFFER_SIZE 1024
	char buffer[STR_BUFFER_SIZE + 1] = { '\0' };
	INFO_PRINT("printDeviceInfo():\n");

	// Get all the text information
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, STR_BUFFER_SIZE, buffer, NULL);
	PRINTER("- Device name: %s\n", buffer);
	clGetDeviceInfo(device_id, CL_DEVICE_VERSION, STR_BUFFER_SIZE, buffer, NULL);
	PRINTER("- Device version: %s\n", buffer);
	clGetDeviceInfo(device_id, CL_DRIVER_VERSION, STR_BUFFER_SIZE, buffer, NULL);
	PRINTER("- Driver version: %s\n", buffer);
	clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, STR_BUFFER_SIZE, buffer, NULL);
	PRINTER("- OpenCL C version: %s\n", buffer);

	// Get all the number information
	cl_ulong buffer_ulong = 0;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &buffer_ulong, NULL);
	PRINTER("- Max memory allocation size: %llu\n", buffer_ulong);
	clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &buffer_ulong, NULL);
	PRINTER("- Global memory size: %llu\n", buffer_ulong);
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &buffer_ulong, NULL);
	PRINTER("- Local memory size: %llu\n", buffer_ulong);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &buffer_ulong, NULL);
	PRINTER("- Max constant buffer size: %llu\n", buffer_ulong);

	// Get all the number information (2)
	cl_uint buffer_uint = 0;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_uint), &buffer_uint, NULL);
	PRINTER("- Max work group size: %u\n", buffer_uint);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &buffer_uint, NULL);
	PRINTER("- Max work item dimensions: %u\n", buffer_uint);

	// Get all the number information (3)
	size_t buffer_size_t[3] = { 0 };
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, buffer_size_t, NULL);
	PRINTER("- Max work item sizes: %zu, %zu, %zu\n", buffer_size_t[0], buffer_size_t[1], buffer_size_t[2]);

	// Get all the number information (4)
	cl_bool buffer_bool = 0;
	clGetDeviceInfo(device_id, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Available: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Compiler available: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Endian little: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Error correction support: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Host unified memory: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Image support: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_LINKER_AVAILABLE, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Linker available: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Preferred interop user sync: %s\n", buffer_bool ? "true" : "false");
	clGetDeviceInfo(device_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(cl_bool), &buffer_bool, NULL);
	PRINTER("- Profiling timer resolution: %s\n", buffer_bool ? "true" : "false");

	// Print a new line at the end
	PRINTER("\n");
}


/**
 * @brief This function read a kernel program from a file and return it by
 * removing all the comments, all the new lines and all the tabs.
 * 
 * @param path			The path to the kernel program
 * 
 * @return char*		The kernel program
 */
char* readKernelProgram(char* path) {

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
	char* buffer = malloc(sizeof(char) * (size + 1));
	if (buffer == NULL) {
		ERROR_PRINT("readEntireFile(): Cannot allocate memory for file %s\n", path);
		return NULL;
	}

	// Read the file
	int read_size;
	if ((read_size = read(fd, buffer, size)) == -1) {
		ERROR_PRINT("readEntireFile(): Cannot read file %s\n", path);
		return NULL;
	}

	// Close the file
	close(fd);

	// Add a null character at the end of the buffer
	buffer[read_size] = '\0';

	// Remove all the multilines comments
	char* start;
	char* end;
	while ((start = strstr(buffer, "/*")) != NULL) {
		end = strstr(start, "*/");
		if (end == NULL) {
			ERROR_PRINT("readKernelProgram(): Cannot find the end of a comment in file %s\n", path);
			return NULL;
		}
		memset(start, ' ', end - start + 2);
	}

	// Remove all the single line comments
	while ((start = strstr(buffer, "//")) != NULL) {
		end = strstr(start, "\n");
		if (end == NULL) {
			ERROR_PRINT("readKernelProgram(): Cannot find the end of a comment in file %s\n", path);
			return NULL;
		}
		memset(start, ' ', end - start + 1);
	}

	// Remove all the tabs by rewriting the file buffer
	int i = 0;
	int j = 0;
	while (buffer[i] != '\0') {
		if (buffer[i] != '\t') {
			buffer[j] = buffer[i];
			j++;
		}
		i++;
	}
	buffer[j] = '\0';

	// Remove all the spaces after a new line
	i = 0;
	j = 0;
	while (buffer[i] != '\0') {
		if (buffer[i] == '\n') {
			buffer[j] = buffer[i];
			j++;
			i++;
			while (buffer[i] == ' ')
				i++;
		} else {
			buffer[j] = buffer[i];
			j++;
			i++;
		}
	}
	buffer[j] = '\0';

	// Remove all the new lines
	i = 0;
	j = 0;
	while (buffer[i] != '\0') {
		if (buffer[i] == '\n') {
			i++;
		} else {
			buffer[j] = buffer[i];
			j++;
			i++;
		}
	}
	buffer[j] = '\0';

	// Return the file buffer
	return buffer;
}


