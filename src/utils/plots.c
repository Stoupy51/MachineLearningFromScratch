
#include "../universal_utils.h"
#include "plots.h"

/**
 * @brief Generate a plot with lines from a data file.
 * 
 * @param output_image_path		Path to the output image
 * @param data_filepath			Path to the data file
 * @param title					Title of the plot
 * @param xlabel				Label of the x axis
 * @param ylabel				Label of the y axis
 * 
 * @return int					0 if success, -1 otherwise
 */
int generateLinesPlot(char *output_image_path, char *data_filepath, char *title, char *xlabel, char *ylabel) {

	// Open a pipe to gnuplot
	FILE *gnuplot_pipe = popen("gnuplot -persistent", "w");
	ERROR_HANDLE_PTR_RETURN_INT(gnuplot_pipe, "generateLinesPlot(): Error while opening gnuplot pipe\n");

	// Set the output image path
	fprintf(gnuplot_pipe, "set term jpeg\n");
	fprintf(gnuplot_pipe, "set output '%s'\n", output_image_path);

	// Set the title, xlabel and ylabel
	fprintf(gnuplot_pipe, "set title '%s'\n", title);
	fprintf(gnuplot_pipe, "set xlabel '%s'\n", xlabel);
	fprintf(gnuplot_pipe, "set ylabel '%s'\n", ylabel);

	// Plot the data
	fprintf(gnuplot_pipe, "plot '%s' with lines title '%s'\n", data_filepath, title);

	// Close the pipe and return
	ERROR_HANDLE_INT_RETURN_INT(pclose(gnuplot_pipe), "generateLinesPlot(): Error while closing gnuplot pipe\n");
	return 0;
}

