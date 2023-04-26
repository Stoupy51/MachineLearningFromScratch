
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

COMPILER = gcc
COMPILER_FLAGS = -Wall -Wextra -Werror
LINKER_FLAGS = -lm

# Get all the source files recursively in the src folder
# If on linux :
#SRC_FILES = $(shell find $(SRC_FOLDER) -type f -name '*.c' -o -name '*.h')
# Else if on Windows:
# Get all relative paths of the source files recursively
QUOTE = \"
SRC_FILES = $(shell powershell -Command 'Get-ChildItem -Path .\src\ -Filter *.c -Recurse | ForEach-Object {Resolve-Path (Join-Path $$_.DirectoryName $$_.Name) -Relative} | ForEach-Object { $$_ }')

# Get all the object files recursively (replace the .c extension by .o from the source files to get the object files)
#OBJ_FILES = $(SRC_FILES:.c=.o)

# Get all the executable files recursively
# If on linux :
#PROGRAMS_FILES = $(shell find $(PROGRAMS_FOLDER) -type f -name '*.c' -o -name '*.h')
# Else if on Windows:
PROGRAMS_FILES = 

# Make dependencies
depend:
	@echo "Preparing dependencies for the compilation..."

	@echo "Dependencies for the source files:"
	$(COMPILER) -MM $(SRC_FILES) > $(DEPENDANCIES_FILE)

	@echo "Dependencies for the programs:"
	$(COMPILER) -MM $(PROGRAMS_FILES) >> $(DEPENDANCIES_FILE)

	@echo "Dependencies are ready for the compilation"

