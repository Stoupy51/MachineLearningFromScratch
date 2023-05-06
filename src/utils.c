
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>

#include "utils.h"

/**
 * @brief This function initializes the main program by
 * printing the header and setting some parameters like:
 * - SetConsoleMode on Windows
 * 
 * @param header	The header to print
 * 
 * @return void
*/
void mainInit(char* header) {

	// Print the header
	printf("\n---------------------------\n");
	INFO_PRINT("%s", header);
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
	if (fd == -1) {
		ERROR_PRINT("writeEntireFile(): Cannot open file %s\n", path);
		return -1;
	}

	// Write the file
	int written_size = write(fd, content, size);
	if (written_size == -1) {
		ERROR_PRINT("writeEntireFile(): Cannot write file %s\n", path);
		return -1;
	}

	// Close the file
	if (close(fd) == -1) {
		ERROR_PRINT("writeEntireFile(): Cannot close file %s\n", path);
		return -1;
	}

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
	if (fd == -1) {
		ERROR_PRINT("readEntireFile(): Cannot open file %s\n", path);
		return NULL;
	}

	// Get the size of the file
	int size = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);

	// Allocate memory for the file content
	char* buffer = malloc(sizeof(char) * (size + 1));
	if (buffer == NULL) {
		ERROR_PRINT("readEntireFile(): Cannot allocate memory for file %s\n", path);
		return NULL;
	}

	// Read the file
	int read_size;
	if ((read_size = read(fd, buffer, size)) == -1) {
		ERROR_PRINT("readEntireFile(): Cannot read file %s\n", path);
		return NULL;
	}

	// Close the file
	close(fd);

	// Add a null character at the end of the buffer
	buffer[read_size] = '\0';

	// Return the file content
	return buffer;
}



