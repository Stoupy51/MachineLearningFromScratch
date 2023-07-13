
#ifndef __GPU_UTILS_H__
#define __GPU_UTILS_H__

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

// Struct to store the OpenCL context
struct opencl_context_t {
	cl_device_id device_id;				// ID of the GPU device
	cl_context context;					// Context of the GPU device
	cl_command_queue command_queue;		// Command queue of the GPU device
};



// Function prototypes

const char* getOpenCLErrorString(cl_int error);
int printProgramBuildLog(cl_program program, cl_device_id device_id, int mode, char* prefix);
struct opencl_context_t setupOpenCL(cl_device_type type_of_device);
cl_device_id* getAllDevicesOfType(cl_device_type type_of_device, cl_uint* device_count);
void printPlatformInfo(cl_platform_id platform_id);
void printDeviceInfo(cl_device_id device_id);
char* readKernelProgram(char* path);
int createKernelFromSource(char* kernel_source, char* kernel_name, cl_program* program, cl_kernel* kernel, struct opencl_context_t* oc);



// One call functions

int fillRandomDoubleArrayGPU(double* array, unsigned long long size, double min, double max);



// Generic macros
#define clReleaseMemObjects(n, buffers) { \
	int ii; \
	for (ii = 0; ii < n; ii++) { \
		clReleaseMemObject(buffers[ii]); \
	} \
}

#endif

