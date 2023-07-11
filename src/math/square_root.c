
#include "square_root.h"

/**
 * @brief Calculate the square root of a double value
 *
 * @param value		Value to calculate the square root of
 *
 * @return double	Square root of the value
 */
double squareRootDouble(double value) {

	// If the value is 0 or 1, return the value itself
	if (value == 0.0 || value == 1.0)
		return value;

	// Initialize the variables for Newton's method
	double previousResult = value;
	double currentResult = value / 2.0;

	// Iterate until the results converge
	while (previousResult - currentResult > 0.000001 || currentResult - previousResult > 0.000001) {
		previousResult = currentResult;
		currentResult = 0.5 * (previousResult + value / previousResult);
	}

	// Return the result
	return currentResult;
}

/**
 * @brief Calculate the square root of a float value
 *
 * @param value		Value to calculate the square root of
 *
 * @return float	Square root of the value
 */
float squareRootFloat(float value) {
	
	// If the value is 0 or 1, return the value itself
	if (value == 0.0f || value == 1.0f)
		return value;

	// Initialize the variables for Newton's method
	float previousResult = value;
	float currentResult = value / 2.0f;

	// Iterate until the results converge
	while (previousResult - currentResult > 0.000001f || currentResult - previousResult > 0.000001f) {
		previousResult = currentResult;
		currentResult = 0.5f * (previousResult + value / previousResult);
	}

	// Return the result
	return currentResult;
}

/**
 * @brief Calculate the square root of a long double value
 *
 * @param value		Value to calculate the square root of
 *
 * @return long double	Square root of the value
 */
long double squareRootLongDouble(long double value) {
	
	// If the value is 0 or 1, return the value itself
	if (value == 0.0l || value == 1.0l)
		return value;

	// Initialize the variables for Newton's method
	long double previousResult = value;
	long double currentResult = value / 2.0l;

	// Iterate until the results converge
	while (previousResult - currentResult > 0.000001l || currentResult - previousResult > 0.000001l) {
		previousResult = currentResult;
		currentResult = 0.5l * (previousResult + value / previousResult);
	}

	// Return the result
	return currentResult;
}

