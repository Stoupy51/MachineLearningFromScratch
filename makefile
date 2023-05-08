
####################################################################################################
# Makefile for the project "MachineLearningFromScratch".
# Made for Windows 10, but should work on Linux and MacOS
####################################################################################################
#
# This makefile handles the compilation of the project by doing the following instructions:
# - Generate a makefile that handles the compilation of all the project
# - Launch the generated makefile
#
# The generated makefile will do the following instructions:
# - Compile every file in the src folder recursively and put the object files in the "obj" folder
# - Compile every file in the programs folder recursively and put the executable files in the "bin" folder
#
####################################################################################################
# Author: 	Stoupy51 (COLLIGNON Alexandre)
####################################################################################################

all:
	@./maker.exe

init:
	gcc maker.c -o maker.exe
	@./maker.exe

clean:
	@make -f generated_makefile clean --no-print-directory
	@rm -f maker.exe
	@rm -f generated_makefile

restart:
	@clear
	@make clean --no-print-directory
	@make init --no-print-directory

