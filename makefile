
####################################################################################################
# Universal Makefile for C projects
# Made for Windows, Linux and MacOS
####################################################################################################
#
# This makefile handles the compilation of the project by doing the following instructions:
# - Recursively search for all .c files in src/ and subdirectories
# - Compile all .c files into .o files if they have been modified or
# their dependencies have been modified by checking timestamps
#
# - Recursively search for all .c files in programs/
# - Compile all .c files into executables if they have been modified or
# their dependencies have been modified by checking timestamps
#
# WARNING: .c files in src/ should have the same name as their header file (.h)
# Example: src/my_file.c and src/my_file.h
#
####################################################################################################
# Author: 	Stoupy51 (COLLIGNON Alexandre)
####################################################################################################

# Variables (Linking flags are only used for 'programs/*.c' files, because it only matters when cooking executables)
ADDITIONAL_FLAGS = -Wall -Wextra -Wpedantic -Werror -O3
LINKING_FLAGS = -Llibs -Ilibs/CL -lOpenCL libs/OpenCL.dll -lm

# Parallel compilation: create a thread for each compilation command (0 = no, 1 = yes)
PARALLEL_COMPILATION = 1

all:
	@./maker.exe "$(ADDITIONAL_FLAGS)" "$(LINKING_FLAGS)" "$(PARALLEL_COMPILATION)"

init:
	gcc maker.c -o maker.exe
	@./maker.exe "$(ADDITIONAL_FLAGS)" "$(LINKING_FLAGS)" "$(PARALLEL_COMPILATION)"

clean:
	@./maker.exe clean

restart:
	@if [ -f "maker.exe" ]; then make clean --no-print-directory; fi
	@make init --no-print-directory

