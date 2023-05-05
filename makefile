
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
	@if [ ! -f "maker.exe" ]; then gcc maker.c -o maker.exe; fi
	@clear
	@./maker.exe

runTest:
	@if [ ! -f "maker.exe" ]; then gcc maker.c -o maker.exe; fi
	@clear
	@./maker.exe
	@./bin/gpu_test.exe

clean:
	@make -f generated_makefile clean
	@rm -f maker.exe
	@rm -f generated_makefile

