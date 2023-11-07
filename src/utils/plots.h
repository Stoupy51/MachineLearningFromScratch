
#ifndef __PLOTS_H__
#define __PLOTS_H__

#include "../neural_network/neural_config.h"

int generate2DLinesPlot(char *output_image_path, char *data_filepath, char *title, char *xlabel, char *ylabel);
int generate2DLinesPlotFromFloatArray(char *output_image_path, char *data_filepath, nn_type **data, char **names, int nb_lines, int nb_points_per_line, char *title, char *xlabel, char *ylabel);

#endif

