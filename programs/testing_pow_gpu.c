
#include <stdlib.h>

#include "../src/utils.h"
#include "../src/gpu/gpu_utils.h"

#define VECTOR_SIZE 100000000

// Global variables
cl_int code = 0;
cl_program program = NULL;
cl_kernel kernel = NULL;
struct opencl_context_t oc;
int i;

/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

	// Release OpenCL objects
	if (kernel != NULL) clReleaseKernel(kernel);
	if (program != NULL) clReleaseProgram(program);
	if (oc.command_queue != NULL) clReleaseCommandQueue(oc.command_queue);
	if (oc.context != NULL) clReleaseContext(oc.context);
}

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

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching GPU test program.\n");
	atexit(exitProgram);

	// Initialize OpenCL and print device info
	oc = setupOpenCL(CL_DEVICE_TYPE_GPU);
	ERROR_HANDLE_PTR(oc.context, "main(): Cannot initialize OpenCL.\n");
	INFO_PRINT("main(): OpenCL initialized.\n");

	// Print device name, version, etc. from context
	printDeviceInfo(oc.device_id);

	// Create the kernel from source
	createKernelFromSource("kernels/pow.cl", "computePowerNaiveExponentiation", &program, &kernel, &oc);
	ERROR_HANDLE_PTR(kernel, "main(): Cannot create kernel from source.\n");
	INFO_PRINT("main(): Kernel 'computePowerNaiveExponentiation' created from source 'kernels/pow.cl'.\n");

	// Create two vectors of random integers
	INFO_PRINT("main(): Creating two vectors of random integers...\n");
	int vec_size = VECTOR_SIZE;
	size_t vector_size_bytes = vec_size * sizeof(int);
	int* a_v = malloc(vector_size_bytes);
	int* b_v = malloc(vector_size_bytes);
	for (i = 0; i < vec_size; i++) {
		a_v[i] = rand() % 10 + 1;
		b_v[i] = rand() % 100000 + 1;
	}
	INFO_PRINT("main(): Vectors created.\n");

	// Create the memory buffers
	cl_mem a_v_buffer = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, vector_size_bytes, NULL, &code);
	ERROR_HANDLE_INT(code, "main(): Cannot create a_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
	cl_mem b_v_buffer = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, vector_size_bytes, NULL, &code);
	ERROR_HANDLE_INT(code, "main(): Cannot create b_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
	INFO_PRINT("main(): Memory buffers created.\n");

	// Copy the vectors to the memory buffers
	INFO_PRINT("main(): Copying vectors to memory buffers...\n");
	code = clEnqueueWriteBuffer(oc.command_queue, a_v_buffer, CL_FALSE, 0, vector_size_bytes, a_v, 0, NULL, NULL);
	ERROR_HANDLE_INT(code, "main(): Cannot write a_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
	code = clEnqueueWriteBuffer(oc.command_queue, b_v_buffer, CL_FALSE, 0, vector_size_bytes, b_v, 0, NULL, NULL);
	ERROR_HANDLE_INT(code, "main(): Cannot write b_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));

	// Set the arguments of the kernel
	INFO_PRINT("main(): Setting arguments of the kernel...\n");
	code = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_v_buffer);
	ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 0, reason: %d / %s\n", code, getOpenCLErrorString(code));
	code = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_v_buffer);
	ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 1, reason: %d / %s\n", code, getOpenCLErrorString(code));
	code = clSetKernelArg(kernel, 2, sizeof(int), (void*)&vec_size);
	ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 2, reason: %d / %s\n", code, getOpenCLErrorString(code));

	// Execute the kernel
	INFO_PRINT("main(): Executing the kernel...\n");
	size_t global_dimensions[] = { VECTOR_SIZE, 0, 0 };
	code = clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
	ERROR_HANDLE_INT(code, "main(): Cannot execute kernel, reason: %d / %s\n", code, getOpenCLErrorString(code));

	// Wait for everything to finish
	INFO_PRINT("main(): Waiting for everything to finish...\n");
	code = clFinish(oc.command_queue);
	ERROR_HANDLE_INT(code, "main(): Cannot finish, reason: %d / %s\n", code, getOpenCLErrorString(code));

	// Read the result from the memory buffer
	INFO_PRINT("main(): Reading the result from the memory buffer...\n");
	code = clEnqueueReadBuffer(oc.command_queue, a_v_buffer, CL_FALSE, 0, vector_size_bytes, a_v, 0, NULL, NULL);
	ERROR_HANDLE_INT(code, "main(): Cannot read the result from the memory buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));

	// Clean up
	INFO_PRINT("main(): Cleaning up...\n");
	clReleaseMemObject(a_v_buffer);
	clReleaseMemObject(b_v_buffer);
	free(a_v);
	free(b_v);

	// Final print and return
	INFO_PRINT("main(): End of program.\n\n");
	return 0;
}

