
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>

#ifndef _WIN32
	#include <stdarg.h>
	#include <limits.h>
#endif

#include "universal_utils.h"

/**
 * @brief Print characters in the stderr with errno handling.
 */
void print_errno_stderr(const char* format, ...) {

	// Get the arguments
	va_list args;
	va_start(args, format);

	// Print the message with errno handling
	if (errno != 0) {

		// Prepare buffer
		char buffer[16384];
		vsprintf(buffer, format, args);

		// Remove the last \n to replace it with the error message
		int err_pos = strlen(buffer);
		while (err_pos > 0 && buffer[err_pos] != '\n') err_pos--;
		buffer[err_pos] = '\0';

		// Print the message with the error message and reset errno
		fprintf(stderr, "%s: %s\n", buffer, strerror(errno));
		errno = 0;
	}
	else {

		// Print the message
		vfprintf(stderr, format, args);
	}
	va_end(args);
}


/**
 * @brief This function initializes the main program by
 * printing the header
 * 
 * @param header	The header to print
 */
void mainInit(char* header) {

	// Launch empty powershell command on Windows,
	// it's a trick to get ANSI colors in every terminal for Windows 10
	#ifdef _WIN32
		system("powershell -command \"\"");
	#endif

	// Print the header
	printf("\n---------------------------\n");
	INFO_PRINT("%s", header);
}

/**
 * @brief This function allocates memory and returns a pointer to it.
 * This function is blocking, it will wait until the memory is allocated.
 * It will try to allocate the memory every 1000ms.
 * 
 * @param size		Size of the memory to allocate
 * @param prefix	Prefix to print in the warning message, ex: "mallocBlocking()"
 * 
 * @return void*	Pointer to the allocated memory
 */
void* mallocBlocking(size_t size, const char* prefix) {
	
	// Allocate the memory
	void* ptr = malloc(size);

	// If the memory is not allocated, wait 1000ms and try again
	while (ptr == NULL) {

		// Print a warning
		WARNING_PRINT("%s: Cannot allocate memory for %zu bytes, waiting 1000ms\n", prefix == NULL ? "mallocBlocking()" : prefix, size);

		// Wait 1000ms and try again
		sleep(1);
		ptr = malloc(size);
	}

	// Return the pointer
	return ptr;
}

/**
 * @brief This function reallocates memory and returns a pointer to it.
 * This function is blocking, it will wait until the memory is allocated.
 * It will try to allocate the memory every 1000ms.
 * 
 * @param ptr		Pointer to the memory to reallocate
 * @param size		Size of the memory to reallocate
 * @param prefix	Prefix to print in the warning message, ex: "reallocBlocking()"
 * 
 * @return void*	Pointer to the reallocated memory
 */
void* reallocBlocking(void* ptr, size_t size, const char* prefix) {
	
	// Reallocate the memory
	void* new_ptr = realloc(ptr, size);

	// If the memory is not allocated, wait 1000ms and try again
	while (new_ptr == NULL) {

		// Print a warning
		WARNING_PRINT("%s: Cannot reallocate memory for %zu bytes, waiting 1000ms\n", prefix == NULL ? "reallocBlocking()" : prefix, size);

		// Wait 1000ms and try again
		sleep(1);
		new_ptr = realloc(ptr, size);
	}

	// Return the pointer
	return new_ptr;
}

/**
 * @brief This function duplicates a memory block and returns a pointer to it.
 * 
 * @param ptr		Pointer to the memory block to duplicate
 * @param size		Size of the memory block to duplicate
 * @param prefix	Prefix to print in the warning message, ex: "duplicateMemory()"
 * 
 * @return void*	Pointer to the duplicated memory block
 */
void* duplicateMemory(void* ptr, size_t size, const char* prefix) {
	if (ptr == NULL) return NULL;
	void* new_ptr = mallocBlocking(size, prefix == NULL ? "duplicateMemory()" : prefix);
	memcpy(new_ptr, ptr, size);
	return new_ptr;
}

/**
 * @brief This function allocates memory for a 2D matrix and returns a pointer to it.
 * This function is blocking, it will wait until the memory is allocated.
 * It will try to allocate the memory every 1000ms.
 * 
 * @param matrix		Pointer to the matrix to allocate
 * @param nb_rows		Number of rows of the matrix
 * @param nb_columns	Number of columns of the matrix
 * @param size			Size of each element of the matrix
 * @param prefix		Prefix to print in the warning message, ex: "try2DFlatMatrixAllocation((void***))"
 * 
 * @return void*		Pointer to the flat matrix if success, NULL otherwise (the allocation is not flat)
 */
void* try2DFlatMatrixAllocation(void ***matrix, int nb_rows, int nb_columns, size_t size, const char* prefix) {

	// Get the total size of the matrix
	long long int total_size = (long long int)nb_rows * (long long int)nb_columns * (long long int)size;

	// Allocate the pointers to the rows
	*matrix = mallocBlocking(sizeof(void*) * nb_rows, prefix == NULL ? "try2DFlatMatrixAllocation((void***))" : prefix);

	// If the total size is too big (more than INT_MAX), allocate the matrix row by row
	if (total_size > INT_MAX && sizeof(size_t) <= sizeof(int)) {

		// Allocate each row
		for (int i = 0; i < nb_rows; i++)
			(*matrix)[i] = mallocBlocking(size * nb_columns, prefix == NULL ? "try2DFlatMatrixAllocation((void***))" : prefix);

		// Return NULL (the allocation is not flat)
		return NULL;
	}

	// If the total size is not too big, allocate the matrix in one block
	else {

		// Allocate the flat matrix
		void* flat_matrix = mallocBlocking(total_size, prefix == NULL ? "try2DFlatMatrixAllocation((void***))" : prefix);

		// Fill the pointers to the rows
		for (int i = 0; i < nb_rows; i++)
			(*matrix)[i] = ((char*)flat_matrix + (i * nb_columns * size));
		
		// Return the flat matrix
		return flat_matrix;
	}
}

/**
 * @brief This function frees a 2D matrix.
 * 
 * @param matrix		Pointer to the matrix to free
 * @param flat_matrix	Pointer to the flat matrix to free (can be NULL, should be output of try2DFlatMatrixAllocation((void***)))
 * @param nb_rows		Number of rows of the matrix
 */
void free2DFlatMatrix(void **matrix, void *flat_matrix, int nb_rows) {

	// If the flat matrix is not NULL, free it
	if (flat_matrix != NULL)
		free(flat_matrix);

	// If the flat matrix is NULL, free each row
	else
		for (int i = 0; i < nb_rows; i++)
			free(matrix[i]);

	// Free the matrix
	free(matrix);
}

/**
 * @brief This function allocates memory for a 3D matrix and returns a pointer to it.
 * This function is blocking, it will wait until the memory is allocated.
 * It will try to allocate the memory every 1000ms.
 * 
 * @param matrix		Pointer to the matrix to allocate
 * @param nb_rows		Number of rows of the matrix
 * @param nb_columns	Number of columns of the matrix
 * @param nb_depths		Number of depths of the matrix
 * @param size			Size of each element of the matrix
 * @param prefix		Prefix to print in the warning message, ex: "try3DFlatMatrixAllocation()"
 * 
 * @return void*		Pointer to the flat matrix if success, NULL otherwise (the allocation is not flat)
 */
void* try3DFlatMatrixAllocation(void ****matrix, int nb_rows, int nb_columns, int nb_depths, size_t size, const char* prefix) {

	// Get the total size of the matrix
	long long int total_size = (long long int)nb_rows * (long long int)nb_columns * (long long int)nb_depths * (long long int)size;

	// Allocate the pointers to the rows
	*matrix = mallocBlocking(sizeof(void**) * nb_rows, prefix == NULL ? "try3DFlatMatrixAllocation()" : prefix);

	// If the total size is too big (more than INT_MAX), allocate the matrix row by row
	if (total_size > INT_MAX && sizeof(size_t) <= sizeof(int)) {

		// Allocate each row
		for (int i = 0; i < nb_rows; i++) {
			(*matrix)[i] = mallocBlocking(sizeof(void*) * nb_columns, prefix == NULL ? "try3DFlatMatrixAllocation()" : prefix);
			for (int j = 0; j < nb_columns; j++)
				(*matrix)[i][j] = mallocBlocking(size * nb_depths, prefix == NULL ? "try3DFlatMatrixAllocation()" : prefix);
		}

		// Return NULL (the allocation is not flat)
		return NULL;
	}

	// If the total size is not too big, allocate the matrix in one block
	else {

		// Allocate the flat matrix
		void* flat_matrix = mallocBlocking(total_size, prefix == NULL ? "try3DFlatMatrixAllocation()" : prefix);

		// Fill the pointers to the rows
		for (int i = 0; i < nb_rows; i++) {
			(*matrix)[i] = mallocBlocking(sizeof(void*) * nb_columns, prefix == NULL ? "try3DFlatMatrixAllocation()" : prefix);
			for (int j = 0; j < nb_columns; j++)
				(*matrix)[i][j] = ((char*)flat_matrix + (i * nb_columns * nb_depths * size) + (j * nb_depths * size));
		}
		
		// Return the flat matrix
		return flat_matrix;
	}
}

/**
 * @brief This function frees a 3D matrix.
 * 
 * @param matrix		Pointer to the matrix to free
 * @param flat_matrix	Pointer to the flat matrix to free (can be NULL, should be output of try3DFlatMatrixAllocation())
 * @param nb_rows		Number of rows of the matrix
 * @param nb_columns	Number of columns of the matrix
 */
void free3DFlatMatrix(void ***matrix, void *flat_matrix, int nb_rows, int nb_columns) {
	
	// If the flat matrix is not NULL, free it
	if (flat_matrix != NULL)
		free(flat_matrix);

	// If the flat matrix is NULL, free each row
	else
		for (int i = 0; i < nb_rows; i++)
			for (int j = 0; j < nb_columns; j++)
				free(matrix[i][j]);

	// Free the matrix
	for (int i = 0; i < nb_rows; i++)
		free(matrix[i]);
	free(matrix);
}

/**
 * @brief This function write a string to a file
 * depending on the mode (append or overwrite).
 * 
 * @param filename		Name of the file to write
 * @param content		Content to write
 * @param size			Size of the content
 * @param mode			Mode of writing (O_APPEND or O_TRUNC)
 * 
 * @return int	0 if success, -1 otherwise
 */
int writeEntireFile(char* path, char* content, int size, int mode) {

	// Open the file
	int fd = open(path, O_WRONLY | O_CREAT | mode, 0644);
	ERROR_HANDLE_INT_RETURN_INT(fd, "writeEntireFile(): Cannot open file '%s'\n", path);

	// Write the file
	int written_size = write(fd, content, size);
	if (written_size == -1) close(fd);
	ERROR_HANDLE_INT_RETURN_INT(written_size, "writeEntireFile(): Cannot write to file '%s'\n", path);

	// Close the file
	close(fd);

	// Return
	return 0;
}


/**
 * @brief This function read a file and return its content as a string.
 * 
 * @param filename Name of the file to read
 * 
 * @return char*	Content of the file as a string if success, NULL otherwise
 */
char* readEntireFile(char* path) {
	
	// Open the file
	int fd = open(path, O_RDONLY);
	ERROR_HANDLE_INT_RETURN_NULL(fd, "readEntireFile(): Cannot open file '%s'\n", path);

	// Get the size of the file
	size_t size = get_file_size(fd);

	// Allocate memory for the file content
	char* buffer = malloc(sizeof(char) * (size + 1));
	if (buffer == NULL) close(fd);
	ERROR_HANDLE_PTR_RETURN_NULL(buffer, "readEntireFile(): Cannot allocate memory for file '%s'\n", path);

	// Read the file
	int read_size = read(fd, buffer, size);
	if (read_size == -1) close(fd);
	ERROR_HANDLE_INT_RETURN_NULL(read_size, "readEntireFile(): Cannot read file '%s'\n", path);

	// Close the file
	close(fd);

	// Add a null character at the end of the buffer
	buffer[read_size] = '\0';

	// Return the file content
	return buffer;
}

#define GET_LINE_BUFFER_SIZE 16384

/**
 * @brief Function that reads a line from a file with a limit of 16384 characters.
 * 
 * @param lineptr	Pointer to the line read to be filled.
 * @param n			Size of the line read.
 * @param fd		File descriptor of the file to read.
 * 
 * @return int		0 if the line is read, -1 if the end of the file is reached.
 */
int get_line_from_file(char **lineptr, int fd) {

	// Variables
	char get_line_buffer[GET_LINE_BUFFER_SIZE];
	memset(get_line_buffer, 0, GET_LINE_BUFFER_SIZE);
	int i = 0;
	char c;

	// Read the file character by character
	while (read(fd, &c, 1 * sizeof(char)) > 0) {

		// If the character is a \n, break the loop
		if (c == '\n') {

			// If i == 0 and the character is a \n, it means that the line is just a \n so we continue
			if (i == 0)
				continue;

			// Break
			break;
		}

		// Add the character to the buffer and continue
		get_line_buffer[i] = c;
		i++;
		if (i == (GET_LINE_BUFFER_SIZE - 1))
			break;
	}

	// If the buffer is empty, return -1 (end of file)
	if (i == 0)
		return -1;

	// Add the \0 at the end of the buffer
	get_line_buffer[i] = '\0';

	// If the lineptr is NULL, allocate it
	if (*lineptr == NULL) {
		ERROR_HANDLE_PTR_RETURN_INT((*lineptr = malloc(i + 1)), "get_line_from_file(): Unable to allocate the lineptr\n");
	}

	// If the lineptr is too small, reallocate it
	else {
		size_t lineptr_size = strlen(*lineptr);
		size_t possible_new_size = i + 1;
		if (lineptr_size < possible_new_size)
			ERROR_HANDLE_PTR_RETURN_INT((*lineptr = realloc(*lineptr, possible_new_size)), "get_line_from_file(): Unable to reallocate the lineptr\n");
	}

	// Copy the buffer to the lineptr
	memcpy(*lineptr, get_line_buffer, i + 1);

	// Return success
	return 0;
}

/**
 * @brief Function that checks if a file is accessible.
 * 
 * @param path	Path of the file to check.
 * 
 * @return int	0 if the file is accessible, -1 otherwise.
 */
int file_accessible(char* path) {
	
	// Open the file
	int fd = open(path, O_RDONLY);

	// If the file is not accessible, return -1
	if (fd == -1)
		return -1;
	
	// If the file is not readable, return -1
	byte c;
	int code = (read(fd, &c, 1) == 1) ? 0 : -1;

	// Close the file
	close(fd);

	// Return success
	return code;
}

/**
 * @brief Function that gets the size of a file.
 * 
 * @param fd	File descriptor of the file to get the size.
 * 
 * @return size_t	Size of the file.
 */
size_t get_file_size(int fd) {
	#ifdef _WIN32
		return _filelength(fd);
	#else
		struct stat st = {0};
		int code = fstat(fd, &st);
		return (code == 0) ? st.st_size : 0;
	#endif
}

/**
 * @brief Function that returns the hash value of a string.
 * 
 * @param str	String to hash.
 * 
 * @return int	Hash value of the string.
 */
int hash_string(char* str) {
	
	// Variables
	int hash = 0;
	int pow = 1;
	int i = 0;

	// Loop through the string
	while (str[i] != '\0') {

		// Add the character to the hash
		hash += str[i] * pow;

		// Increment the power
		pow *= 31;

		// Increment the index
		i++;
	}

	// Return the hash
	return hash;
}

/**
 * @brief Function that removes a directory.
 * 
 * @param path	Path of the directory to remove.
 * 
 * @return int	0 if the directory is removed, -1 otherwise.
 */
int remove_directory(char* path) {
	char command[2048];
	#ifdef _WIN32
		sprintf(command, "rmdir /s /q \"%s\"", path);
		return system(command);
	#else
		sprintf(command, "rm -rf \"%s\"", path);
		return system(command);
	#endif
}

int local_nb_threads = 0;

/**
 * @brief Function that returns the number of threads of the CPU of the computer.
 * Efficiency: O(1) (the number of threads is computed only once)
 * 
 * @return int	Number of threads of the CPU.
 */
int getNumberOfThreads() {
	if (local_nb_threads == 0) {
		#ifdef _WIN32
			SYSTEM_INFO sysinfo;
			GetSystemInfo(&sysinfo);
			local_nb_threads = sysinfo.dwNumberOfProcessors;
		#else
			local_nb_threads = sysconf(_SC_NPROCESSORS_ONLN);
		#endif
	}
	return local_nb_threads;
}

