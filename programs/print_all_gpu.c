
#include <stdlib.h>

#include "../src/utils.h"
#include "../src/gpu/gpu_utils.h"
#include "../src/vectors.h"
#include "../src/st_benchmark.h"

// Global variables
cl_int code = 0;
cl_uint i;


/**
 * @brief Function run at the end of the program
 * [registered with atexit()] in the main() function.
 * 
 * @return void
*/
void exitProgram() {

	// Print end of program
	INFO_PRINT("exitProgram(): End of program, press enter to exit.\n");
	getchar();
	exit(0);
}


/**
 * This program is an introduction test to OpenCL programming.
 * It is a simple program that prints all the available GPU devices.
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main() {

	// Print program header and register exitProgram() with atexit()
	mainInit("main(): Launching 'print_all_gpu' program.\n");
	atexit(exitProgram);

	///// GPU Devices /////
	{
		// Get GPU devices
		cl_uint device_count;
		cl_device_id* devices = getAllDevicesOfType(CL_DEVICE_TYPE_GPU, &device_count);
		ERROR_HANDLE_PTR(devices, "main(): Error while getting GPU devices.\n");

		// Print devices
		INFO_PRINT("main(): Found %d GPU devices.\n", device_count);
		for (i = 0; i < device_count; i++) {
			INFO_PRINT("main(): Device %d:\n", i);
			printDeviceInfo(devices[i]);
		}

		// Free memory
		for (i = 0; i < device_count; i++)
			clReleaseDevice(devices[i]);
		free(devices);
	}

	///// CPU Devices /////
	{
		// Get CPU devices
		cl_uint device_count;
		cl_device_id* devices = getAllDevicesOfType(CL_DEVICE_TYPE_CPU, &device_count);
		ERROR_HANDLE_PTR(devices, "main(): Error while getting CPU devices.\n");

		// Print devices
		INFO_PRINT("main(): Found %d CPU devices.\n", device_count);
		for (i = 0; i < device_count; i++) {
			INFO_PRINT("main(): Device %d:\n", i);
			printDeviceInfo(devices[i]);
		}

		// Free memory
		for (i = 0; i < device_count; i++)
			clReleaseDevice(devices[i]);
		free(devices);
	}

	// Final print and return
	INFO_PRINT("main(): End of program.\n\n");
	return 0;
}


