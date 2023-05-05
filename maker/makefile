
# Make file for the project
all:
	@gcc maker.c -o maker.exe -Wall -Wextra -O3 -lm

	@./maker.exe

clean:
	@make -f generated_makefile clean
	@rm -f maker.exe
	@rm -f generated_makefile

