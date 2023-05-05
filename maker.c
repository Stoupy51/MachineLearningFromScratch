
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

#ifdef _WIN32
	#include <direct.h>
	#include <fileapi.h>
	#define mkdir(path, mode) _mkdir(path)
#else
	#include <sys/stat.h>
	#include <sys/types.h>
	#include <dirent.h>
#endif

#define SRC_FOLDER "src"
#define OBJ_FOLDER "obj"
#define BIN_FOLDER "bin"
#define PROGRAMS_FOLDER "programs"
#define MAKEFILE_NAME "generated_makefile"

#define OPENCL_DLL "C:/Windows/System32/OpenCL.dll"
#define OPENCL_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"

#define CC "gcc"
#define LINKER_FLAGS "-lm -lpthread -L\""OPENCL_LIB_PATH"\" -lOpenCL"
#define COMPILER_FLAGS "-Wall -Wextra -Wpedantic -Werror -O3 "LINKER_FLAGS" \""OPENCL_DLL"\""


/**
 * @brief Create the content of the Makefile
 * - For each .c file in the src folder, create a .o file in the obj folder
 * - For each .c file in the programs folder, create a program in the bin folder
 * 
 * @param content	Pointer to the content of the Makefile
 * 
 * @return int		0 if success, -1 otherwise
*/
int createMakefileContent(char *content) {

	// Written progress in the content
	size_t written = 0;

	// Write the header
	written += sprintf(content + written, "\nCC = "CC"\n");
	written += sprintf(content + written, "ALL_FLAGS = "COMPILER_FLAGS"\n");
	written += sprintf(content + written, "\nall: objects programs\n");

	///// For each .c file in the src folder, create a .o file in the obj folder
	// Write the objects rule
	written += sprintf(content + written, "\nobjects:\n");
	written += sprintf(content + written, "\t@echo \"\"\n");
	written += sprintf(content + written, "\t@echo \"Compiling the source files...\"\n");
	written += sprintf(content + written, "\t@echo \"|\"\n");

	// Write the compilation commands (On Windows)
	#ifdef _WIN32

		///// Get the list of files in the src folder recursively
		// Create the command and execute it
		char line[1024];
		char command[1024];
		sprintf(command, "dir /s /b \"%s\\*.c\"", SRC_FOLDER);
		FILE *fp = _popen(command, "r");

		// Prepare a long string to store path to every object files
		char *object_files = malloc(1024 * 1024 * sizeof(char));
		memset(object_files, '\0', 1024 * 1024 * sizeof(char));
		size_t object_files_written = 0;

		// For each line
		while (fgets(line, 1024, fp) != NULL) {

			// Remove the \n at the end of the line
			line[strlen(line) - 1] = '\0';
			
			// Get the relative path of the file (relative to the src folder)
			char *relative_path = strstr(line, SRC_FOLDER);
			relative_path += strlen(SRC_FOLDER) + 1;
			int size = strlen(relative_path) + 1;

			// Replace the \ by /
			int i;
			for (i = 0; i < size; i++)
				if (relative_path[i] == '\\')
					relative_path[i] = '/';

			// Get the object file path (.o file)
			char *object_file = malloc(size * sizeof(char));
			memset(object_file, '\0', size * sizeof(char));
			strncpy(object_file, relative_path, strrchr(relative_path, '.') - relative_path);
			strcat(object_file, ".o");

			// Create the folder if it doesn't exist
			char *folder = malloc(size * sizeof(char));
			memset(folder, '\0', size * sizeof(char));
			strcat(folder, OBJ_FOLDER);
			strcat(folder, "/");
			folder += strlen(OBJ_FOLDER) + 1;
			strncpy(folder, object_file, strrchr(object_file, '/') - object_file);
			folder -= strlen(OBJ_FOLDER) + 1;
			mkdir(folder, 0777);

			// Write the compilation command
			written += sprintf(content + written, "\t$(CC) -c \"%s/%s\" -o \"%s/%s\" $(ALL_FLAGS)\n", SRC_FOLDER, relative_path, OBJ_FOLDER, object_file);

			// Add the path to the list
			object_files_written += sprintf(object_files + object_files_written, "\"%s/%s\" ", OBJ_FOLDER, object_file);
		}

		// Close the file
		_pclose(fp);

		///// Get the list of files in the programs folder recursively

		// Write the programs rule
		written += sprintf(content + written, "\t@echo \"\"\n");
		written += sprintf(content + written, "\t@echo \"\"\n");
		written += sprintf(content + written, "\t@echo \"\"\n");
		written += sprintf(content + written, "\t@echo \"Compiling the programs...\"\n");
		written += sprintf(content + written, "\t@echo \"|\"\n");

		// Create the command and execute it
		sprintf(command, "dir /s /b \"%s\\*.c\"", PROGRAMS_FOLDER);
		fp = _popen(command, "r");

		// For each line
		while (fgets(line, 1024, fp) != NULL) {

			// Remove the \n at the end of the line
			line[strlen(line) - 1] = '\0';
			
			// Get the relative path of the file (relative to the programs folder)
			char *relative_path = strstr(line, PROGRAMS_FOLDER);
			relative_path += strlen(PROGRAMS_FOLDER) + 1;
			int size = strlen(relative_path) + 1;

			// Replace the \ by /
			int i;
			for (i = 0; i < size; i++)
				if (relative_path[i] == '\\')
					relative_path[i] = '/';
			
			// Get the exe file path (.exe file)
			char *exe_file = malloc(size * sizeof(char) + sizeof(".exe"));
			memset(exe_file, '\0', size * sizeof(char));
			strncpy(exe_file, relative_path, strrchr(relative_path, '.') - relative_path);
			strcat(exe_file, ".exe");

			// Write the compilation command
			written += sprintf(content + written, "\t$(CC) -o \"%s/%s\" \"%s/%s\" %s $(ALL_FLAGS)\n", BIN_FOLDER, exe_file, PROGRAMS_FOLDER, relative_path, object_files);
		}

		// Close the file
		_pclose(fp);

		// Free the memory
		free(object_files);

		// Write the last line of programs rule
		written += sprintf(content + written, "\t@echo \"|\"\n");
		written += sprintf(content + written, "\t@echo \"Compilation done\"\n");
		written += sprintf(content + written, "\t@echo \"\"\n");

	// TODO: Write the compilation commands (On Linux)
	#else

	#endif

	// Write the clean rule
	written += sprintf(content + written, "\nclean:\n");
	written += sprintf(content + written, "\t@echo \"Cleaning the project...\"\n");
	written += sprintf(content + written, "\t@rm -rf %s\n", OBJ_FOLDER);
	written += sprintf(content + written, "\t@rm -rf %s\n", BIN_FOLDER);
	written += sprintf(content + written, "\t@echo \"Clean done\"\n\n");
	
	// Return
	return 0;
}


/**
 * @brief Program that creates the content
 * of the Makefile for the project
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
*/
int main() {

	// Create folders
	mkdir(SRC_FOLDER, 0777);
	mkdir(OBJ_FOLDER, 0777);
	mkdir(BIN_FOLDER, 0777);
	mkdir(PROGRAMS_FOLDER, 0777);

	// Create the content variable
	char *content = (char*)malloc(1024 * 1024 * sizeof(char));

	// Create the content of the Makefile
	if (createMakefileContent(content) == -1) {
		perror("Error while creating the content of the Makefile");
		return 1;
	}

	// Create the Makefile
	int fd = open(MAKEFILE_NAME, O_CREAT | O_WRONLY | O_TRUNC, 0644);
	if (fd == -1) {
		perror("Error while creating the Makefile");
		return 1;
	}

	// Write the content of the Makefile
	content[1024*1024-1] = '\0';
	int written_bytes = write(fd, content, strlen(content) * sizeof(char));
	if (written_bytes == -1) {
		perror("Error while writing the Makefile");
		return 1;
	}

	// Close the Makefile
	if (close(fd) == -1) {
		perror("Error while closing the Makefile");
		return 1;
	}

	// Free the content
	free(content);

	// Launch the make command with the Makefile created and the --no-print-directory option
	system("make -f " MAKEFILE_NAME " --no-print-directory");

	// Return
	return 0;
}

