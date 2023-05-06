
#include <stdlib.h>

#include "../src/utils.h"
#include "../src/gpu/gpu_utils.h"

#define VECTOR_SIZE 100000000

/**
 * This program is an introduction test to OpenCL programming.
 * It is a simple program that manipulates vectors of integers.
 * - The first vector is filled with random integers between 1 and 10.
 * - The second vector is filled with random integers between 1 and 100000.
 * 
 * The program use GPU to compute the first vector to the power of the second vector.
 *
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header
	mainInit("main(): Launching GPU test program.\n");
	cl_int code = 0;

	// Initialize OpenCL
	INFO_PRINT("main(): Initializing OpenCL...\n");
	struct opencl_context_t oc = setupOpenCL(CL_DEVICE_TYPE_GPU);
	INFO_PRINT("main(): OpenCL initialized.\n");

	// Print device name, version, etc. from context
	printDeviceInfo(oc.device_id);

	// Get the kernel source code
	INFO_PRINT("main(): Getting kernel source code...\n");
	char* kernel_source = readKernelProgram("src/gpu/test.cl");

	// Create the program
	INFO_PRINT("main(): Creating program...\n");
	cl_program program = clCreateProgramWithSource(oc.context, 1, (const char**)&kernel_source, NULL, &code);
	if (program == NULL) {
		ERROR_PRINT("main(): Cannot create program, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Build the program
	INFO_PRINT("main(): Building program...\n");
	code = clBuildProgram(program, 1, &oc.device_id, NULL, NULL, NULL);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("main(): Cannot build program, reason: %d / %s\n", code, getOpenCLErrorString(code));
		printProgramBuildLog(program, oc.device_id, ERROR_LEVEL, "main(): ");
		exit(EXIT_FAILURE);
	}

	// Create the kernel
	INFO_PRINT("main(): Creating kernel...\n");
	cl_kernel kernel = clCreateKernel(program, "computePowerNaiveExponentiation", &code);
	if (kernel == NULL) {
		ERROR_PRINT("main(): Cannot create kernel, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Create two vectors of random integers
	INFO_PRINT("main(): Creating two vectors of random integers...\n");
	int vec_size = VECTOR_SIZE;
	size_t vector_size_bytes = vec_size * sizeof(int);
	int* a_v = malloc(vector_size_bytes);
	int* b_v = malloc(vector_size_bytes);
	for (int i = 0; i < vec_size; i++) {
		a_v[i] = rand() % 10 + 1;
		b_v[i] = rand() % 100000 + 1;
	}

	// Create the memory buffers
	INFO_PRINT("main(): Creating memory buffers...\n");
	cl_mem a_v_buffer = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, vector_size_bytes, NULL, &code);
	if (a_v_buffer == NULL) {
		ERROR_PRINT("main(): Cannot create buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}
	cl_mem b_v_buffer = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, vector_size_bytes, NULL, &code);
	if (b_v_buffer == NULL) {
		ERROR_PRINT("main(): Cannot create buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Copy the vectors to the memory buffers
	INFO_PRINT("main(): Copying vectors to memory buffers...\n");
	code = clEnqueueWriteBuffer(oc.command_queue, a_v_buffer, CL_FALSE, 0, vector_size_bytes, a_v, 0, NULL, NULL);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("main(): Cannot write buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}
	code = clEnqueueWriteBuffer(oc.command_queue, b_v_buffer, CL_FALSE, 0, vector_size_bytes, b_v, 0, NULL, NULL);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("main(): Cannot write buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Set the arguments of the kernel
	INFO_PRINT("main(): Setting arguments of the kernel...\n");
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_v_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_v_buffer);
	clSetKernelArg(kernel, 2, sizeof(int), (void*)&vec_size);

	// Execute the kernel
	INFO_PRINT("main(): Executing the kernel...\n");
	size_t global_dimensions[] = { VECTOR_SIZE, 0, 0 };
	code = clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("main(): Cannot execute kernel, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Wait for everything to finish
	INFO_PRINT("main(): Waiting for everything to finish...\n");
	code = clFinish(oc.command_queue);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("main(): Cannot finish, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Read the result from the memory buffer
	INFO_PRINT("main(): Reading the result from the memory buffer...\n");
	code = clEnqueueReadBuffer(oc.command_queue, a_v_buffer, CL_FALSE, 0, vector_size_bytes, a_v, 0, NULL, NULL);
	if (code != CL_SUCCESS) {
		ERROR_PRINT("main(): Cannot read buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
		exit(EXIT_FAILURE);
	}

	// Clean up
	INFO_PRINT("main(): Cleaning up...\n");
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(a_v_buffer);
	clReleaseMemObject(b_v_buffer);
	clReleaseCommandQueue(oc.command_queue);
	clReleaseContext(oc.context);
	free(kernel_source);
	free(a_v);
	free(b_v);

	// Final print and return
	INFO_PRINT("main(): End of program.\n");
	return 0;
}

