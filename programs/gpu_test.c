
#include "../src/gpu/utils.h"

#define VECTOR_SIZE 1000000

/**
 * This program is an introduction test to OpenCL programming.
 * It is a simple program that manipulates vectors of integers.
 * - The first vector is filled with random integers between 1 and 10.
 * - The second vector is filled with random integers between 1 and 100.
 * 
 * The program use GPU to compute the first vector to the power of the second vector.
 *
 * @author Stoupy51 (COLLIGNON Alexandre)
*/
int main(int argc, char** argv) {

	// Print program header
	printf("\n---------------------------\n");
	INFO_PRINT("main(): Launching GPU test program.\n");

	// Initialize OpenCL
	INFO_PRINT("main(): Initializing OpenCL...\n");
	struct opencl_context_t oc = setupOpenCL(CL_DEVICE_TYPE_GPU);

	// Print device name, version, etc.
	char buffer[128];
	clGetDeviceInfo(oc.device_id, CL_DEVICE_NAME, 100, buffer, NULL);
	INFO_PRINT("main(): Device name: %s\n", buffer);
	clGetDeviceInfo(oc.device_id, CL_DEVICE_VERSION, 100, buffer, NULL);
	INFO_PRINT("main(): Device version: %s\n", buffer);
	clGetDeviceInfo(oc.device_id, CL_DRIVER_VERSION, 100, buffer, NULL);
	INFO_PRINT("main(): Driver version: %s\n", buffer);
	clGetDeviceInfo(oc.device_id, CL_DEVICE_OPENCL_C_VERSION, 100, buffer, NULL);
	INFO_PRINT("main(): OpenCL C version: %s\n", buffer);

	// Get the kernel source code
	INFO_PRINT("main(): Getting kernel source code...\n");
	char* kernel_source = readEntireFile("src/gpu/test.cl");

	// Create & Build the program
	INFO_PRINT("main(): Creating & Building program...\n");
	cl_program program = clCreateProgramWithSource(oc.context, 1, (const char**)&kernel_source, NULL, NULL);
	clBuildProgram(program, 1, &oc.device_id, NULL, NULL, NULL);

	// Create the kernel
	INFO_PRINT("main(): Creating kernel...\n");
	cl_kernel kernel = clCreateKernel(program, "computePower", NULL);

	// Create two vectors of random integers
	INFO_PRINT("main(): Creating two vectors of random integers...\n");
	size_t vector_size_bytes = VECTOR_SIZE * sizeof(int);
	int* a_v = malloc(vector_size_bytes);
	int* b_v = malloc(vector_size_bytes);
	for (int i = 0; i < VECTOR_SIZE; i++) {
		a_v[i] = rand() % 10 + 1;
		b_v[i] = rand() % 100 + 1;
	}

	// Create the memory buffers
	INFO_PRINT("main(): Creating memory buffers...\n");
	cl_mem a_v_buffer = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, vector_size_bytes, NULL, NULL);
	cl_mem b_v_buffer = clCreateBuffer(oc.context, CL_MEM_READ_ONLY, vector_size_bytes, NULL, NULL);

	// Copy the vectors to the memory buffers
	INFO_PRINT("main(): Copying vectors to memory buffers...\n");
	clEnqueueWriteBuffer(oc.command_queue, a_v_buffer, CL_FALSE, 0, vector_size_bytes, a_v, 0, NULL, NULL);
	clEnqueueWriteBuffer(oc.command_queue, b_v_buffer, CL_FALSE, 0, vector_size_bytes, b_v, 0, NULL, NULL);

	// Set the arguments of the kernel
	INFO_PRINT("main(): Setting arguments of the kernel...\n");
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_v_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_v_buffer);

	// Execute the kernel
	INFO_PRINT("main(): Executing the kernel...\n");
	size_t global_dimensions[] = { VECTOR_SIZE, 0, 0 };
	clEnqueueNDRangeKernel(oc.command_queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);

	// Read the result from the memory buffer
	INFO_PRINT("main(): Reading the result from the memory buffer...\n");
	clEnqueueReadBuffer(oc.command_queue, a_v_buffer, CL_FALSE, 0, vector_size_bytes, a_v, 0, NULL, NULL);

	// Wait for everything to finish
	INFO_PRINT("main(): Waiting for everything to finish...\n");
	clFinish(oc.command_queue);

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

