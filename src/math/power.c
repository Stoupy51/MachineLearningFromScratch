
#include "power.h"

/**
 * @brief Use the fast method of exponentiation
 * to calculate the power of an integer
 * 
 * @param value		Value to calculate the power of
 * @param power		Power to calculate
 * 
 * @return int		Result of the power
 */
int powerIntFastExp(int value, int power) {

	// If the power is 0, return 1
	if (power == 0)
		return 1;

	// If the power is negative, return 0
	if (power < 0)
		return 0;

	// While the power is not 0
	int result = 1;
	while (power > 0) {

		// If the power is odd, multiply the result by the value
		if (power & 1)
			result *= value;

		// In every case, square the value and divide the power by 2
		value *= value;
		power >>= 1;
	}

	// Return the result
	return result;
}

/**
 * @brief Use the fast method of exponentiation
 * to calculate the power of a double
 *
 * @param value		Value to calculate the power of
 * @param power		Power to calculate
 *
 * @return double	Result of the power
 */
double powerDoubleFastExp(double value, double power) {

	// If the power is 0, return 1
	if (power == 0.0)
		return 1.0;

	// If the power is negative, return 1 / value ^ -power
	if (power < 0)
		return 1.0 / powerDoubleFastExp(value, -power);
	
	// Store the power as an integer
	int powerInt = (int)power;

	// While the power is not 0
	double result = 1.0;
	while (powerInt > 0) {

		// If the power is odd, multiply the result by the value
		if (powerInt & 1)
			result *= value;

		// In every case, square the value and divide the power by 2
		value *= value;
		powerInt /= 2;
	}

	// At this point, the remaining power is between 0 and 1
	double remainingPower = power - (int)power;

	// If the remaining power is 0, return the result
	if (remainingPower == 0.0)
		return result;
}

/**
 * @brief Use the fast method of exponentiation
 * to calculate the power of a float
 *
 * @param value		Value to calculate the power of
 * @param power		Power to calculate
 *
 * @return float	Result of the power
 */
float powerFloatFastExp(float value, float power);

long double powerLongDoubleFastExp(long double value, long double power);

