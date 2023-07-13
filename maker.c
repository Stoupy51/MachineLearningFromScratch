
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef _WIN32
	#include <unistd.h>
#endif

#if 1 == 1

typedef struct file_path_t {
	int size;				// Size of the string
	char* str;				// String
	long long timestamp;	// Last modification time of the file and all the files included
} file_path_t;

typedef struct strlinked_list_element_t {
	file_path_t path;
	struct strlinked_list_element_t* next;
} str_linked_list_element_t;

typedef struct str_linked_list_t {
	int size;
	str_linked_list_element_t* head;
} str_linked_list_t;

int str_linked_list_init(str_linked_list_t* list) {
	list->size = 0;
	list->head = NULL;
	return 0;
}

str_linked_list_element_t* str_linked_list_insert(str_linked_list_t* list, file_path_t path) {
	str_linked_list_element_t* new_element = malloc(sizeof(str_linked_list_element_t));
	if (new_element == NULL) {
		perror("Error while allocating memory for a new element of the linked list\n");
		return NULL;
	}
	new_element->path = path;
	new_element->next = list->head;
	list->head = new_element;
	list->size++;
	return new_element;
}

str_linked_list_element_t* str_linked_list_search(str_linked_list_t list, file_path_t path) {
	str_linked_list_element_t* current_element = list.head;
	while (current_element != NULL) {
		if (strcmp(current_element->path.str, path.str) == 0)
			return current_element;
		current_element = current_element->next;
	}
	return NULL;
}

void str_linked_list_free(str_linked_list_t* list) {
	if (list->size == 0)
		return;
	str_linked_list_element_t* current_element = list->head;
	while (current_element != NULL) {
		str_linked_list_element_t* next_element = current_element->next;
		free(current_element);
		current_element = next_element;
	}
	list->size = 0;
	list->head = NULL;
}

#endif

#ifdef _WIN32
	#include <direct.h>
	#define mkdir(path, mode) _mkdir(path)
	#define WINDOWS_FLAGS " -lws2_32"
#else
	#define WINDOWS_FLAGS ""
#endif

#define SRC_FOLDER "src"
#define OBJ_FOLDER "obj"
#define BIN_FOLDER "bin"
#define PROGRAMS_FOLDER "programs"
#define FILES_TIMESTAMPS ".files_timestamps"

#define CC "gcc"
#define LINKER_FLAGS "-lm -lpthread" WINDOWS_FLAGS
#define COMPILER_FLAGS "-Wall -Wextra -Wpedantic -Werror -O3"
#define ALL_FLAGS COMPILER_FLAGS " " LINKER_FLAGS

// Global variables
char* additional_flags = NULL;
char* linking_flags = NULL;
char* obj_files = NULL;
int obj_files_size = 0;
str_linked_list_t files_timestamps;

/**
 * @brief Clean the project by deleting all the .o and .exe files in their respective folders
 * 
 * @return int		0 if success, -1 otherwise
 */
int clean_project() {
	printf("Cleaning the project...\n");

	// Delete all the .o files in the obj folder
	int code = system("rm -rf "OBJ_FOLDER"/**/*.o");
	if (code != 0) {
		perror("Error while deleting the .o files\n");
		return -1;
	}

	// Delete all the .exe files in the bin folder
	code = system("rm -f "BIN_FOLDER"/*.exe");
	if (code != 0) {
		perror("Error while deleting the .exe files\n");
		return -1;
	}

	// Delete the .files_timestamps file
	code = system("rm -f "FILES_TIMESTAMPS);

	// Delete this file
	code = system("rm -f maker.exe");

	// Return
	printf("Project cleaned!\n\n");
	return 0;
}

/**
 * @brief Load the last modification time of each file in the .files_timestamps file
 * 
 * @return int		0 if success, -1 otherwise
 */
int load_files_timestamps(str_linked_list_t* files_timestamps) {

	// Initialize the list
	int code = str_linked_list_init(files_timestamps);
	if (code != 0) {
		perror("Error while initializing the list of files timestamps\n");
		return -1;
	}

	// Open the file and create it if it doesn't exist
	FILE *file = fopen(FILES_TIMESTAMPS, "rb");
	if (file == NULL) {
		file = fopen(FILES_TIMESTAMPS, "w");
		fclose(file);
		return 0;
	}

	// While there are bytes to read,
	while (1) {

		// Read the size of the string
		int size;
		code = fread(&size, sizeof(int), 1, file);
		if (code != 1)
			break;

		// Read the string
		char* str = malloc(size + 1);
		if (str == NULL) {
			perror("Error while allocating memory for a string in the .files_timestamps file\n");
			return -1;
		}
		code = fread(str, size, 1, file);
		if (code != 1) {
			free(str);
			break;
		}
		str[size] = '\0';

		// Read the timestamp
		long long timestamp;
		code = fread(&timestamp, sizeof(long long), 1, file);
		if (code != 1) {
			free(str);
			break;
		}

		// Insert the string in the list
		file_path_t path;
		path.size = size;
		path.str = str;
		path.timestamp = timestamp;
		str_linked_list_insert(files_timestamps, path);
	}

	// Close the file and return
	fclose(file);
	return 0;
}

/**
 * @brief Save the last modification time of each file in the .files_timestamps file
 * 
 * @return int		0 if success, -1 otherwise
 */
int save_files_timestamps(str_linked_list_t files_timestamps) {

	// Open the file in write mode
	int fd = open(FILES_TIMESTAMPS, O_CREAT | O_WRONLY | O_TRUNC, 0777);
	if (fd == -1) {
		perror("Error while opening the .files_timestamps file\n");
		return -1;
	}

	// For each element in the list,
	str_linked_list_element_t* current_element = files_timestamps.head;
	while (current_element != NULL) {

		// Write the size of the string
		int size = current_element->path.size;
		write(fd, &size, sizeof(int));

		// Write the string
		write(fd, current_element->path.str, size);

		// Write the timestamp
		long long timestamp = current_element->path.timestamp;
		write(fd, &timestamp, sizeof(long long));

		// Next element
		current_element = current_element->next;
	}

	// Close the file
	close(fd);

	// Return
	return 0;
}

/**
 * @brief Get a line from a file
 * 
 * @param lineptr		Pointer to the line
 * @param n				Pointer to the size of the line
 * @param stream		Pointer to the file
 * 
 * @return int			Number of characters read, -1 if error or end of file
 */
int custom_getline(char **lineptr, size_t *n, FILE *stream) {
	size_t capacity = *n;
	size_t pos = 0;
	int c;
	if (*lineptr == NULL)
		if ((*lineptr = malloc(capacity)) == NULL)
			return -1;
	while ((c = fgetc(stream)) != EOF) {
		(*lineptr)[pos++] = c;
		if (pos >= capacity) {
			capacity *= 2;
			char *newptr = realloc(*lineptr, capacity);
			if (newptr == NULL)
				return -1;
			*lineptr = newptr;
		}
		if (c == '\n')
			break;
	}
	if (pos == 0)
		return -1;
	(*lineptr)[pos] = '\0';
	*n = capacity;
	return pos;
}

/**
 * @brief Get the last modification time of a file
 * and all the files included in it
 * 
 * @param filepath			Path of the file
 * @param past_filepath		Path of the file that included this file (To avoid infinite loops)
 * 
 * @return long long
 */
long long getTimestampRecursive(const char* filepath, const char* past_filepath) {

	// Open the file
	FILE* file = fopen(filepath, "r");
	if (file == NULL)	// Ignore files that don't exist
		return -1;

	// Get the last modification time of the file
	struct stat file_stat;
	stat(filepath, &file_stat);
	long long timestamp = file_stat.st_mtime;

	// For each line in the file,
	char* line = NULL;
	size_t len = 128;
	int read;
	while ((read = custom_getline(&line, &len, file)) != -1) {

		// If the line is an include,
		if (strncmp(line, "#include", 8) == 0) {

			// Get the path of the included file
			char* included_file = line + 9;
			while (*included_file == ' ' || *included_file == '\t')
				included_file++;
			if (*included_file == '"' || *included_file == '<') {
				included_file++;
				char* end = included_file;
				while (*end != '"' && *end != '>')
					end++;
				*end = '\0';
			}
			else
				continue;
			
			// Get the path of the included file depending on the path of the current file
			char* included_file_path = malloc(strlen(filepath) + strlen(included_file) + 1);
			if (included_file_path == NULL) {
				perror("Error while allocating memory for a file path\n");
				return -1;
			}
			strcpy(included_file_path, filepath);
			char* last_slash = strrchr(included_file_path, '/');
			if (last_slash != NULL)
				*(last_slash + 1) = '\0';
			strcat(included_file_path, included_file);

			// If the included file is the same as the past file, ignore it
			if (past_filepath != NULL && strcmp(included_file_path, past_filepath) == 0) {
				free(included_file_path);
				continue;
			}

			// Get the last modification time of the included file & update the timestamp if needed
			long long included_file_timestamp = getTimestampRecursive(included_file_path, filepath);
			if (included_file_timestamp > timestamp)
				timestamp = included_file_timestamp;
		}
	}

	// Close the file
	fclose(file);

	// If the filepath finish with .h, do the same thing with the .c file
	int filepath_len = strlen(filepath);
	if (filepath[filepath_len - 1] == 'h' && filepath[filepath_len - 2] == '.') {
		char* c_filepath = strdup(filepath);
		c_filepath[filepath_len - 1] = 'c';
		long long c_timestamp = getTimestampRecursive(c_filepath, filepath);
		if (c_timestamp > timestamp)
			timestamp = c_timestamp;
		free(c_filepath);
	}

	// Free the line and return
	if (line != NULL)
		free(line);
	return timestamp;
}

/**
 * @brief Create all the folders in a path
 * 
 * @param filepath		Path of the file
 */
void create_folders_from_path(const char* filepath) {

	// Get path of the folder
	char* folder = strdup(filepath);
	int size = strlen(folder);
	for (int i = size - 1; i >= 0; i--) {
		if (folder[i] == '/') {
			folder[i] = '\0';
			break;
		}
	}

	// Create the folder if it doesn't exist
	if (folder[0] != '\0')
		mkdir(folder, 0777);
	free(folder);
	return;
}


/**
 * @brief Find all the .c files in the src folder recursively,
 * compare their timestamp with the one in the .files_timestamps file
 * and compile them if needed
 * 
 * @param files_timestamps		Pointer to the list of files timestamps
 * 
 * @return int		0 if success, -1 otherwise
 */
int findCFiles(str_linked_list_t *files_timestamps) {

	// Create the command
	#ifdef _WIN32
		char command[256] = "dir /s /b "SRC_FOLDER"\\*.c";
	#else
		char command[256] = "find "SRC_FOLDER" -name \"*.c\"";
	#endif

	// Open the PIPE
	FILE* pipe = popen(command, "r");
	if (pipe == NULL) {
		perror("Error while opening the PIPE\n");
		return -1;
	}

	// Initialize the counter
	int compileCount = 0;

	// For each line in the PIPE,
	char* line = NULL;
	size_t len = 128;
	int read;
	while ((read = custom_getline(&line, &len, pipe)) != -1) {

		// Remove the \n at the end of the line
		line[read - 1] = '\0';

		// On windows, replace the \ by /
		#ifdef _WIN32
			for (int i = 0; i < read; i++)
				if (line[i] == '\\')
					line[i] = '/';
		#endif

		// Get the relative path (ignoring everything before the src folder)
		char* relative_path = strdup(strstr(line, SRC_FOLDER));

		// Get the timestamp of the file
		long long timestamp = getTimestampRecursive(line, NULL);

		// Get the saved timestamp of the file
		file_path_t path;
		path.size = strlen(relative_path);
		path.str = relative_path;
		path.timestamp = timestamp;
		str_linked_list_element_t* element = str_linked_list_search(*files_timestamps, path);

		// Get the path of the .o file
		char* obj_path = strdup(relative_path);
		obj_path[strlen(obj_path) - 1] = 'o';	// Replace the .c by .o
		int i;
		for (i = 0; i < sizeof(SRC_FOLDER) - 1; i++)
			obj_path[i] = OBJ_FOLDER[i];
		
		// Add the .o file to the list of object files
		if (obj_files == NULL) {
			obj_files_size = 1024 * 1024 * sizeof(char);
			obj_files = malloc(obj_files_size);
		}
		else {
			int needed_size = strlen(obj_path) + 4;
			if (obj_files_size < needed_size) {
				obj_files_size = needed_size * 2;
				char* newptr = realloc(obj_files, obj_files_size);
				if (newptr == NULL) {
					perror("Error while reallocating memory for the list of object files\n");
					return -1;
				}
				obj_files = newptr;
			}
		}
		strcat(obj_files, "\"");
		strcat(obj_files, obj_path);
		strcat(obj_files, "\" ");

		// If the file is not in the list or if the timestamp is different,
		if (element == NULL || element->path.timestamp != timestamp) {

			// Manage the counter
			if (compileCount++ == 0)
				printf("Compiling the source files...\n");

			// If the folder of the .o file doesn't exist, create it
			create_folders_from_path(obj_path);

			// Compile the file
			char command[32768];
			sprintf(
				command,
				CC" -c \"%s\" -o \"%s\" %s",

				relative_path,
				obj_path,
				additional_flags == NULL ? "" : additional_flags
			);
			printf("- %s\n", command);
			if (system(command) != 0) {
				save_files_timestamps(*files_timestamps);
				exit(-1);
			}

			// If the file is not in the list, add it
			if (element == NULL)
				str_linked_list_insert(files_timestamps, path);
			// Else, update the timestamp
			else
				element->path.timestamp = timestamp;
		}
	}

	// If there is no file to compile, print a message
	if (compileCount == 0)
		printf("No source file to compile...\n\n");
	else
		printf("\nCompilation of %d source files done!\n\n", compileCount);

	// Free the line
	if (line != NULL)
		free(line);
	
	// Close the PIPE
	pclose(pipe);

	// Return
	return 0;
}

/**
 * @brief Compile all the .c files in the src folder recursively
 * For each file found, check if there is a change in the .c file or in the .h files included
 * If there is a change, compile the file and remember the last modification time in the .files_timestamps file
 * Else, do nothing
 * 
 * @return int		0 if success, -1 otherwise
 */
int compile_sources() {

	// Get the last modification time of each file in the .files_timestamps file
	if (load_files_timestamps(&files_timestamps) != 0) {
		perror("Error while loading the files timestamps\n");
		return -1;
	}

	// For each .c file in the src folder,
	findCFiles(&files_timestamps);

	// Save the last modification time of each file in the .files_timestamps file
	if (save_files_timestamps(files_timestamps) != 0) {
		perror("Error while saving the files timestamps\n");
		return -1;
	}

	// Return
	return 0;
}

/**
 * @brief Compile all the .c files in the programs folder (not recursively)
 * 
 * @return int		0 if success, -1 otherwise
 */
int compile_programs() {
	
	// Create the command
	#ifdef _WIN32
		char command[256] = "dir /b "PROGRAMS_FOLDER"\\*.c";
	#else
		// Ignore "programs/" in the path
		char command[256] = "ls "PROGRAMS_FOLDER"/*.c | sed 's/"PROGRAMS_FOLDER"\\///'";
	#endif

	// Open the PIPE
	FILE* pipe = popen(command, "r");
	if (pipe == NULL) {
		perror("Error while opening the PIPE\n");
		return -1;
	}

	// Initialize the counter
	int compileCount = 0;

	// For each line in the PIPE,
	char* line = NULL;
	size_t len = 128;
	int read;
	while ((read = custom_getline(&line, &len, pipe)) != -1) {

		// Remove the \n at the end of the line & Copy the path of the file
		line[read - 1] = '\0';

		// Get the path of the file
		char* filename = malloc(read + sizeof(PROGRAMS_FOLDER) + 1);
		if (filename == NULL) {
			perror("Error while allocating memory for a file path\n");
			return -1;
		}
		strcpy(filename, PROGRAMS_FOLDER"/");
		strcat(filename, line);

		// Check timestamp
		file_path_t path;
		path.size = strlen(filename);
		path.str = filename;
		path.timestamp = getTimestampRecursive(filename, NULL);
		str_linked_list_element_t* element = str_linked_list_search(files_timestamps, path);
		if (element == NULL || element->path.timestamp != path.timestamp) {

			// Add "exe" at the end of the line
			line[read - 2] = '\0';	// Remove the 'c' in ".c"
			strcat(line, "exe");

			// Manage the counter
			if (compileCount++ == 0)
				printf("Compiling programs...\n");

			// Compile the file
			char command[32768];
			sprintf(
				command,
				CC" \"%s\" -o \""BIN_FOLDER"/%s\" %s %s %s",

				filename,
				line,
				obj_files == NULL ? "" : obj_files,
				additional_flags == NULL ? "" : additional_flags,
				linking_flags == NULL ? "" : linking_flags
			);
			char command_reduced[96];
			memcpy(command_reduced, command, 95);
			command_reduced[95] = '\0';
			printf("- %s...\n", command_reduced);
			if (system(command) != 0)
				continue;
			
			// If the file is not in the list, add it
			if (element == NULL)
				str_linked_list_insert(&files_timestamps, path);
			else
				element->path.timestamp = path.timestamp;
		}
	}

	// If there is no file to compile, print a message
	if (compileCount == 0)
		printf("No program to compile...\n\n");
	else
		printf("\nCompilation of %d programs done!\n\n", compileCount);

	// Free the line
	if (line != NULL)
		free(line);
	
	// Close the PIPE
	pclose(pipe);

	// Return
	return 0;
}


/**
 * @brief Program that compiles the entire project
 * 
 * @param argc Number of arguments
 * @param argv Arguments, accept only: "clean"
 * 
 * @author Stoupy51 (COLLIGNON Alexandre)
 */
int main(int argc, char **argv) {

	// Print the header
	system("clear");

	// Check if the user wants to clean the project
	if (argc == 2 && strcmp(argv[1], "clean") == 0)
		return clean_project();
	else if (argc == 3) {
		additional_flags = argv[1];
		linking_flags = argv[2];
	}

	// Create folders if they don't exist
	mkdir(SRC_FOLDER, 0777);
	mkdir(OBJ_FOLDER, 0777);
	mkdir(BIN_FOLDER, 0777);
	mkdir(PROGRAMS_FOLDER, 0777);

	// Compile all the .c files in the src folder recursively
	if (compile_sources() != 0) {
		perror("Error while compiling the sources\n");
		return -1;
	}

	// For each .c file in the programs folder,
	if (compile_programs() != 0) {
		perror("Error while compiling the programs\n");
		return -1;
	}

	// Save & Free the list of files timestamps
	save_files_timestamps(files_timestamps);
	str_linked_list_free(&files_timestamps);

	// Free the list of object files
	if (obj_files != NULL)
		free(obj_files);

	// Return
	return 0;
}

