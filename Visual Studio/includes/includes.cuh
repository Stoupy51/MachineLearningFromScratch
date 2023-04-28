
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

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
#define DEBUG_LEVEL (INFO_LEVEL || ERROR_LEVEL)

// Utils defines to check debug level
#define IS_INFO_LEVEL (DEBUG_LEVEL & INFO_LEVEL)
#define IS_WARNING_LEVEL (DEBUG_LEVEL & WARNING_LEVEL)
#define IS_ERROR_LEVEL (DEBUG_LEVEL & ERROR_LEVEL)

// Utils defines to print debug messages
#define INFO_PRINT(...) if (IS_INFO_LEVEL) { printf(GREEN "[INFO] " RESET __VA_ARGS__); }
#define WARNING_PRINT(...) if (IS_WARNING_LEVEL) { printf(RED "[WARNING] " RESET __VA_ARGS__); }
#define ERROR_PRINT(...) if (IS_ERROR_LEVEL) { printf(RED "[ERROR] "RESET __VA_ARGS__); }

