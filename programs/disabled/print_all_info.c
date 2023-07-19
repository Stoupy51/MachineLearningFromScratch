
#include <stdlib.h>

#include "../src/universal_utils.h"
#include "../src/gpu/gpu_utils.h"
#include "../src/utils/vectors.h"
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
	INFO_PRINT("exitProgram(): End of program, press enter to exit\n");
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
	mainInit("main(): Launching 'print_all_info' program\n");
	atexit(exitProgram);

	///// Print platforms /////
	{
		// Get platform
		cl_platform_id* platforms;
		cl_uint num_platforms;
		code = clGetPlatformIDs(0, NULL, &num_platforms);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while getting number of platforms with code %d / %s\n", code, getOpenCLErrorString(code));

		platforms = malloc(sizeof(cl_platform_id) * num_platforms);
		ERROR_HANDLE_PTR_RETURN_INT(platforms, "main(): Error while allocating memory for platforms\n");
		code = clGetPlatformIDs(num_platforms, platforms, NULL);
		ERROR_HANDLE_INT_RETURN_INT(code, "main(): Error while getting platforms with code %d / %s\n", code, getOpenCLErrorString(code));

		// Print platforms
		INFO_PRINT("main(): Found %d platforms\n", num_platforms);
		for (i = 0; i < num_platforms; i++) {
			INFO_PRINT("main(): Platform %d:\n", i);
			printPlatformInfo(platforms[i]);
		}

		// Free memory
		printf("\n");
		free(platforms);
	}

	///// Print all /////
	{
		// Get All devices
		cl_uint device_count;
		cl_device_id* devices = getAllDevicesOfType(CL_DEVICE_TYPE_ALL, &device_count);
		ERROR_HANDLE_PTR_RETURN_INT(devices, "main(): Error while getting ALL devices\n");

		// Print devices
		INFO_PRINT("main(): Found %d devices\n", device_count);
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
	INFO_PRINT("main(): End of program\n");
	return 0;
}


