
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
####################################################################################################
# Author: 	Stoupy51 (COLLIGNON Alexandre)
####################################################################################################

# Variables (Linking flags are only used for 'programs/*.c' files, because it only matters when cooking executables)
ADDITIONAL_FLAGS = -Wall -Wextra -Wpedantic -Werror -O3 -lm -lpthread -lws2_32
LINKING_FLAGS = -L\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64\" -I\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include\" -lOpenCL C:/Windows/System32/OpenCL.dll

all:
	@./maker.exe "$(ADDITIONAL_FLAGS)" "$(LINKING_FLAGS)"

init:
	gcc maker.c -o maker.exe
	@./maker.exe "$(ADDITIONAL_FLAGS)" "$(LINKING_FLAGS)"

clean:
	@./maker.exe clean

restart:
	@if [ -f "maker.exe" ]; then make clean --no-print-directory; fi
	@make init --no-print-directory

