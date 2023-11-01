
#include "../universal_utils.h"
#include "plots.h"

/**
 * @brief Generate a plot with lines from a data file using gnuplot.
 * 
 * @param output_image_path		Path to the output image
 * @param data_filepath			Path to the data file
 * @param title					Title of the plot
 * @param xlabel				Label of the x axis
 * @param ylabel				Label of the y axis
 * 
 * @return int					0 if success, -1 otherwise
 */
int generate2DLinesPlot(char *output_image_path, char *data_filepath, char *title, char *xlabel, char *ylabel) {

	// Read the data file to get the datasets names
	char *file_string = readEntireFile(data_filepath);
	int file_string_size = strlen(file_string);
	int nb_datasets = 0;
	for (int i = 0; i < file_string_size; i++)
		if (file_string[i] == '"')
			nb_datasets++;
	nb_datasets /= 2;
	free(file_string);

	// Open a pipe to gnuplot
	FILE *gnuplot_pipe = popen("gnuplot -persistent", "w");
	ERROR_HANDLE_PTR_RETURN_INT(gnuplot_pipe, "generate2DLinesPlot(): Error while opening gnuplot pipe\n");

	// Set the output image path
	fprintf(gnuplot_pipe, "set terminal jpeg\n");
	fprintf(gnuplot_pipe, "set output \"%s\"\n", output_image_path);

	// Set the title, xlabel and ylabel
	fprintf(gnuplot_pipe, "set title \"%s\"\n", title);
	fprintf(gnuplot_pipe, "set xlabel \"%s\"\n", xlabel);
	fprintf(gnuplot_pipe, "set ylabel \"%s\"\n", ylabel);

	// Plot the data
	fprintf(gnuplot_pipe, "plot ");
	for (int i = 0; i < nb_datasets; i++) {
		fprintf(gnuplot_pipe, "\"%s\" index %d using 1:2 title columnheader(1) with lines", data_filepath, i);
		if (i < nb_datasets - 1)
			fprintf(gnuplot_pipe, ", ");
	}

	// Close the pipe and return
	ERROR_HANDLE_INT_RETURN_INT(pclose(gnuplot_pipe), "generate2DLinesPlot(): Error while closing gnuplot pipe\n");
	return 0;
}

