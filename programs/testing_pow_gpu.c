
#include <stdlib.h>

#include "../src/utils.h"
#include "../src/gpu/gpu_utils.h"
#include "../src/vectors.h"
#include "../src/st_benchmark.h"

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
 * Results using GPU: NVIDIA GeForce GTX 1060 6 GB
 * [BENCHMARK] computePowerNaiveExponentiation executed 5 times in 29.724000s
 * [BENCHMARK] computePowerFastExponentiation executed 1000 times in 16.714000s
 * [BENCHMARK] computePowerBuiltInExponentiation executed 100 times in 25.098000s
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching GPU test program.\n");
	atexit(exitProgram);

	// Create two vectors of random integers
	int vec_size = VECTOR_SIZE;
	size_t vector_size_bytes = vec_size * sizeof(int);
	int* _v = malloc(vector_size_bytes * 2);
	int* a_v = _v; int* b_v = _v + vec_size;
	fill_random_vector(a_v, 1, 10, vec_size);
	fill_random_vector(b_v, 1, 100000, vec_size);
	INFO_PRINT("main(): Vectors created.\n");

	// Initialize OpenCL and print device info
	oc = setupOpenCL(CL_DEVICE_TYPE_GPU);
	ERROR_HANDLE_PTR(oc.context, "main(): Cannot initialize OpenCL.\n");

	// Create the memory buffers
	cl_mem v_buffers[2] = { NULL, NULL };
	v_buffers[0] = clCreateBuffer(oc.context, CL_MEM_READ_WRITE, vector_size_bytes, NULL, &code);
	ERROR_HANDLE_INT(code, "main(): Cannot create a_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
	v_buffers[1] = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, vector_size_bytes, NULL, &code);
	ERROR_HANDLE_INT(code, "main(): Cannot create b_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));

	// Copy the vectors to the memory buffers
	code = clEnqueueWriteBuffer(oc.command_queue, v_buffers[0], CL_FALSE, 0, vector_size_bytes, a_v, 0, NULL, NULL);
	ERROR_HANDLE_INT(code, "main(): Cannot write a_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));
	code = clEnqueueWriteBuffer(oc.command_queue, v_buffers[1], CL_FALSE, 0, vector_size_bytes, b_v, 0, NULL, NULL);
	ERROR_HANDLE_INT(code, "main(): Cannot write b_v_buffer, reason: %d / %s\n", code, getOpenCLErrorString(code));

	///// computePowerNaiveExponentiation /////
	{
		// Create the kernel
		createKernelFromSource("kernels/pow.cl", "computePowerNaiveExponentiation", &program, &kernel, &oc);
		ERROR_HANDLE_PTR(kernel, "main(): Cannot create kernel from source.\n");

		// Set the arguments of the kernel
		code = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&v_buffers[0]);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 0, reason: %d / %s\n", code, getOpenCLErrorString(code));
		code = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&v_buffers[1]);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 1, reason: %d / %s\n", code, getOpenCLErrorString(code));
		code = clSetKernelArg(kernel, 2, sizeof(int), (void*)&vec_size);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 2, reason: %d / %s\n", code, getOpenCLErrorString(code));

		// Execute the kernel
		size_t global_dimensions[] = { VECTOR_SIZE, 0, 0 };

		// Wait for everything to finish
		char buffer[2048];
		ST_BENCHMARK_SOLO(buffer,
			{
				code = clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
				code = clFinish(oc.command_queue);
			},
			"computePowerNaiveExponentiation", 5
		);
		printf("%s", buffer);
		ERROR_HANDLE_INT(code, "main(): Cannot finish, reason: %d / %s\n", code, getOpenCLErrorString(code));
	}

	///// computePowerFastExponentiation /////
	{
		// Create the kernel
		createKernelFromSource("kernels/pow.cl", "computePowerFastExponentiation", &program, &kernel, &oc);
		ERROR_HANDLE_PTR(kernel, "main(): Cannot create kernel from source.\n");

		// Set the arguments of the kernel
		code = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&v_buffers[0]);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 0, reason: %d / %s\n", code, getOpenCLErrorString(code));
		code = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&v_buffers[1]);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 1, reason: %d / %s\n", code, getOpenCLErrorString(code));
		code = clSetKernelArg(kernel, 2, sizeof(int), (void*)&vec_size);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 2, reason: %d / %s\n", code, getOpenCLErrorString(code));

		// Execute the kernel
		size_t global_dimensions[] = { VECTOR_SIZE, 0, 0 };
		code = clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT(code, "main(): Cannot execute kernel, reason: %d / %s\n", code, getOpenCLErrorString(code));

		// Wait for everything to finish
		char buffer[2048];
		ST_BENCHMARK_SOLO(buffer,
			{
				code = clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
				code = clFinish(oc.command_queue);
			},
			"computePowerFastExponentiation", 1000
		);
		printf("%s", buffer);
		ERROR_HANDLE_INT(code, "main(): Cannot finish, reason: %d / %s\n", code, getOpenCLErrorString(code));
	}

	///// computePowerBuiltInExponentiation /////
	{
		// Create the kernel
		createKernelFromSource("kernels/pow.cl", "computePowerBuiltInExponentiation", &program, &kernel, &oc);
		ERROR_HANDLE_PTR(kernel, "main(): Cannot create kernel from source.\n");

		// Set the arguments of the kernel
		code = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&v_buffers[0]);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 0, reason: %d / %s\n", code, getOpenCLErrorString(code));
		code = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&v_buffers[1]);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 1, reason: %d / %s\n", code, getOpenCLErrorString(code));
		code = clSetKernelArg(kernel, 2, sizeof(int), (void*)&vec_size);
		ERROR_HANDLE_INT(code, "main(): Cannot set kernel argument 2, reason: %d / %s\n", code, getOpenCLErrorString(code));

		// Execute the kernel
		size_t global_dimensions[] = { VECTOR_SIZE, 0, 0 };
		code = clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
		ERROR_HANDLE_INT(code, "main(): Cannot execute kernel, reason: %d / %s\n", code, getOpenCLErrorString(code));

		// Wait for everything to finish
		char buffer[2048];
		ST_BENCHMARK_SOLO(buffer,
			{
				code = clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
				code = clFinish(oc.command_queue);
			},
			"computePowerBuiltInExponentiation", 100
		);
		printf("%s", buffer);
		ERROR_HANDLE_INT(code, "main(): Cannot finish, reason: %d / %s\n", code, getOpenCLErrorString(code));
	}

	// Clean up
	INFO_PRINT("main(): Cleaning up...\n");
	clReleaseMemObjects(2, v_buffers);
	free(_v);

	// Final print and return
	INFO_PRINT("main(): End of program.\n\n");
	return 0;
}

