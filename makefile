
####################################################################################################
# Makefile for the project "MachineLearningFromScratch".
# Made for Windows 10, but should work on Linux and MacOS
####################################################################################################
#
# This makefile handles the compilation of the project by doing the following instructions:
# - Prepare every dependency for the compilation for all the files in the
# 	"src" folder recursively so that subfolders are also included
#
# - Compile every file in the src folder recursively and put the object files in the "obj" folder
#
# - And for each executable file in the "programs" folder:
#	- Prepare dependencies for the compilation
#	- Compile the file and put the executable in the "bin" folder
#
# This makefile also handles the cleaning of the project using "make clean"
# This will remove all the object files and the executable files
#
####################################################################################################
# Author: 	Stoupy51 (COLLIGNON Alexandre)
####################################################################################################

# Variables
SRC_FOLDER = src
OBJ_FOLDER = obj
BIN_FOLDER = bin
PROGRAMS_FOLDER = programs
DEPENDANCIES_FILE = dependencies.txt
OPENCL_DLL = "C:/Windows/System32/OpenCL.dll"
OPENCL_LIB_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"

COMPILER = gcc
LINKER_FLAGS = -lm -lpthread -L$(OPENCL_LIB_PATH) -lOpenCL
COMPILER_FLAGS = -Wall -Wextra -O3 $(LINKER_FLAGS) $(OPENCL_DLL)

# Get all the source files recursively in the src folder
# If on linux :
#SRC_FILES = $(shell find $(SRC_FOLDER) -type f -name '*.c' -o -name '*.h')
# Else if on Windows:
# Get all relative paths of the source files recursively
SRC_FILES = $(shell powershell -Command 'Get-ChildItem -Path $(SRC_FOLDER) -Filter *.c -Recurse | ForEach-Object {Resolve-Path (Join-Path $$_.DirectoryName $$_.Name) -Relative} | ForEach-Object { $$_.Replace("\", "/") } ')
QUOTED_SRC_FILES = $(foreach file, $(SRC_FILES),"$(file)")

# Get all the object files recursively (replace the .c extension by .o from the source files to get the object files)
# (Removing the src folder from the path by substringing the path)
SRC_FILES_WITHOUT_FOLDER = $(subst $(SRC_FOLDER)/,,$(SRC_FILES))
OBJ_FILES = $(foreach file, $(SRC_FILES_WITHOUT_FOLDER),"$(OBJ_FOLDER)/$(file:.c=.o)")

# Get all the executable files recursively
# If on linux :
#PROGRAMS_FILES = $(shell find $(PROGRAMS_FOLDER) -type f -name '*.c' -o -name '*.h')
# Else if on Windows:
PROGRAMS_FILES = $(shell powershell -Command 'Get-ChildItem -Path $(PROGRAMS_FOLDER) -Filter *.c -Recurse | ForEach-Object {Resolve-Path (Join-Path $$_.DirectoryName $$_.Name) -Relative} | ForEach-Object { $$_.Replace("\", "/") } ')
QUOTED_PROGRAMS_FILES = $(foreach file, $(PROGRAMS_FILES), "$(file)")
PROGRAMS_FILES_WITHOUT_FOLDER = $(subst $(PROGRAMS_FOLDER)/,,$(PROGRAMS_FILES))

# Compile the whole project
all:
	@echo "Preparing the project for compilation..."
	@rm -rf $(BIN_FOLDER)/*
	@mkdir -p $(OBJ_FOLDER)
	@mkdir -p $(BIN_FOLDER)
	@$(foreach file, $(SRC_FILES_WITHOUT_FOLDER), mkdir -p $(OBJ_FOLDER)/$(dir $(file));)

	@echo "Compiling the source files..."
	@echo $(COMPILER) $(COMPILER_FLAGS) -c $(SRC_FILES) -o $(OBJ_FILES)
	@$(COMPILER) $(COMPILER_FLAGS) -c $(SRC_FILES) -o $(OBJ_FILES)

	@echo "Compiling the programs..."
	@echo $(foreach file, $(PROGRAMS_FILES_WITHOUT_FOLDER), $(COMPILER) $(COMPILER_FLAGS) $(OBJ_FILES) $(PROGRAMS_FOLDER)/$(file) -o $(BIN_FOLDER)/$(file:.c=);)
	@$(foreach file, $(PROGRAMS_FILES_WITHOUT_FOLDER), $(COMPILER) $(COMPILER_FLAGS) $(OBJ_FILES) $(PROGRAMS_FOLDER)/$(file) -o $(BIN_FOLDER)/$(file:.c=);)

	@echo "Compilation done"

# Make dependencies
depend:
	@echo "Preparing dependencies for the compilation..."

	@echo "Dependencies for the source files:"
	@$(COMPILER) -MM $(QUOTED_SRC_FILES) > $(DEPENDANCIES_FILE)

	@echo "Dependencies for the programs:"
	@$(COMPILER) -MM $(QUOTED_PROGRAMS_FILES) >> $(DEPENDANCIES_FILE)

	@echo "Dependencies are ready for the compilation"


# Clean the project
clean:
	@echo "Cleaning the project..."
	@rm -rf $(OBJ_FOLDER)
	@rm -rf $(BIN_FOLDER)
	@rm -f $(DEPENDANCIES_FILE)
	@echo "Project cleaned"


# Clean the project and recompile it
restart: clean all

# Include the dependencies file
-include $(DEPENDANCIES_FILE)


