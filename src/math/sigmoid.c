
#include "sigmoid.h"
#include <math.h>

/**
 * @brief Calculate the sigmoid of a number (double)
 * 
 * @param value		Number to calculate the sigmoid of
 * 
 * @return double	Sigmoid of the number
 */
double sigmoid(double value) {
	return (1 / (1 + pow(EULER_NUMBER, -value)));
}

/**
 * @brief Calculate the sigmoid of a number (float)
 * 
 * @param value		Number to calculate the sigmoid of
 * 
 * @return float	Sigmoid of the number
 */
float sigmoidf(float value) {
    return (1 / (1 + powf(EULER_NUMBER, -value)));
}

/**
 * @brief Calculate the sigmoid of a number (long double)
 * 
 * @param value		Number to calculate the sigmoid of
 * 
 * @return long double	Sigmoid of the number
 */
long double sigmoidl(long double value) {
    return (1 / (1 + powl(EULER_NUMBER, -value)));
}

