
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
	#include <windows.h>

	// Sleep function
	#define sleep(seconds) Sleep((int)seconds * 1000)
	typedef SSIZE_T ssize_t;

	// stat64
	#define stat64 _stat64
#else
	#include <unistd.h>
	#include <errno.h>
	#include <sys/stat.h>
#endif


// Utils defines
typedef unsigned char byte;

// Defines for colors
#define BLACK "\033[0;30m"
#define RED "\033[0;31m"
#define GREEN "\033[0;32m"
#define YELLOW "\033[0;33m"
#define BLUE "\033[0;34m"
#define MAGENTA "\033[0;35m"
#define CYAN "\033[0;36m"
#define WHITE "\033[0;37m"
#define RESET "\033[0m"
#define STR_BLACK_R(str) BLACK str RESET
#define STR_RED_R(str) RED str RESET
#define STR_GREEN_R(str) GREEN str RESET
#define STR_YELLOW_R(str) YELLOW str RESET
#define STR_BLUE_R(str) BLUE str RESET
#define STR_MAGENTA_R(str) MAGENTA str RESET
#define STR_CYAN_R(str) CYAN str RESET
#define STR_WHITE_R(str) WHITE str RESET
#define STR_BLACK(str) BLACK str
#define STR_RED(str) RED str
#define STR_GREEN(str) GREEN str
#define STR_YELLOW(str) YELLOW str
#define STR_BLUE(str) BLUE str
#define STR_MAGENTA(str) MAGENTA str
#define STR_CYAN(str) CYAN str
#define STR_WHITE(str) WHITE str

// Defines for the different levels of debug output
#define INFO_LEVEL 1
#define DEBUG_LEVEL 2
#define WARNING_LEVEL 4
#define ERROR_LEVEL 8
// Change the following line to change the debug states
#define DEBUG_STATES (INFO_LEVEL | DEBUG_LEVEL | WARNING_LEVEL | ERROR_LEVEL)
#define DEVELOPMENT_MODE 1

// Utils defines to check debug level
#define IS_INFO_LEVEL (DEBUG_STATES & INFO_LEVEL)
#define IS_DEBUG_LEVEL (DEBUG_STATES & DEBUG_LEVEL)
#define IS_WARNING_LEVEL (DEBUG_STATES & WARNING_LEVEL)
#define IS_ERROR_LEVEL (DEBUG_STATES & ERROR_LEVEL)

///// Utils defines to print debug messages (With errno handling)
// #define INFO_PRINT(...) : For info, print in stderr only if in development mode, else print in stdout
// #define DEBUG_PRINT(...) : For debug, print in stderr only if in development mode, else print in stdout
// #define WARNING_PRINT(...) : For warnings, always print in stderr
// #define ERROR_PRINT(...) : For errors, always print in stderr
// #define PRINTER(...) : For printing in the console, without any default color or level
void print_errno_stderr(const char* format, ...);
#define PRINT_ERRNO_STDERR(...) print_errno_stderr(__VA_ARGS__)
#if DEVELOPMENT_MODE
	#define PRINTER(...) PRINT_ERRNO_STDERR(__VA_ARGS__)
#else
	#define PRINTER(...) printf(__VA_ARGS__)
#endif
#if IS_INFO_LEVEL
	#define INFO_PRINT(...) PRINTER(GREEN "[INFO] " RESET __VA_ARGS__)
#else
	#define INFO_PRINT(...) {}
#endif
#if IS_DEBUG_LEVEL
	#define DEBUG_PRINT(...) PRINTER(CYAN "[DEBUG] " RESET __VA_ARGS__)
#else
	#define DEBUG_PRINT(...) {}
#endif
#if IS_WARNING_LEVEL
	#define WARNING_PRINT(...) PRINT_ERRNO_STDERR(YELLOW "[WARNING] " RESET __VA_ARGS__)
#else
	#define WARNING_PRINT(...) {}
#endif
#if IS_ERROR_LEVEL
	#define ERROR_PRINT(...) PRINT_ERRNO_STDERR(RED "[ERROR] " RESET __VA_ARGS__)
#else
	#define ERROR_PRINT(...) {}
#endif

// Utils for printing prefix
#define PREFIX_PRINT(print_level) { switch (print_level) { case INFO_LEVEL: INFO_PRINT(""); break; case DEBUG_LEVEL: DEBUG_PRINT(""); break; case WARNING_LEVEL: WARNING_PRINT(""); break; case ERROR_LEVEL: ERROR_PRINT(""); break; default: PRINTER(""); break; } }

// Utils for error handling
#define ENABLE_ERROR_HANDLING 1
#if ENABLE_ERROR_HANDLING == 1
	#define ERROR_HANDLE_INT_RETURN_INT(error, ...) { if (error < 0) { ERROR_PRINT(__VA_ARGS__); return error; } }
	#define ERROR_HANDLE_INT_RETURN_NULL(error, ...) { if (error < 0) { ERROR_PRINT(__VA_ARGS__); return NULL; } }
	#define ERROR_HANDLE_PTR_RETURN_INT(ptr, ...) { if (ptr == NULL) { ERROR_PRINT(__VA_ARGS__); return -1; } }
	#define ERROR_HANDLE_PTR_RETURN_NULL(ptr, ...) { if (ptr == NULL) { ERROR_PRINT(__VA_ARGS__); return NULL; } }
#else
	#define ERROR_HANDLE_INT_RETURN_INT(error, ...) {}
	#define ERROR_HANDLE_INT_RETURN_NULL(error, ...) {}
	#define ERROR_HANDLE_PTR_RETURN_INT(ptr, ...) {}
	#define ERROR_HANDLE_PTR_RETURN_NULL(ptr, ...) {}
#endif

// Utils for warning handling
#define ENABLE_WARNING_HANDLING 1
#if ENABLE_WARNING_HANDLING == 1
	#define WARNING_HANDLE_INT(error, ...) { if (error < 0) { WARNING_PRINT(__VA_ARGS__); } }
	#define WARNING_HANDLE_PTR(ptr, ...) { if (ptr == NULL) { WARNING_PRINT(__VA_ARGS__); } }
#else
	#define WARNING_HANDLE_INT(error, ...) {}
	#define WARNING_HANDLE_PTR(ptr, ...) {}
#endif

///// Structures
typedef struct simple_string_t {
	char* str;
	size_t size;
} simple_string_t;

// Function prototypes
void mainInit(char* header);
void* mallocBlocking(size_t size, const char* prefix);
void* reallocBlocking(void* ptr, size_t size, const char* prefix);
void* duplicateMemory(void* ptr, size_t size, const char* prefix);
void* try2DFlatMatrixAllocation(void ***matrix, int nb_rows, int nb_columns, size_t size, const char* prefix);
void free2DFlatMatrix(void **matrix, void *flat_matrix, int nb_rows);
void* try3DFlatMatrixAllocation(void ****matrix, int nb_rows, int nb_columns, int nb_depths, size_t size, const char* prefix);
void free3DFlatMatrix(void ***matrix, void *flat_matrix, int nb_rows, int nb_columns);
int writeEntireFile(char* path, char* content, int size, int mode);
char* readEntireFile(char* path);
int get_line_from_file(char **lineptr, int fd);
int file_accessible(char* path);
size_t get_file_size(int fd);
int hash_string(char* str);
int remove_directory(char* path);
int getNumberOfThreads();

#endif

