
#include "power.h"
#include "square_root.h"

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
	
	// Store the power as an integer and the initial value
	int powerInt = (int)power;
	int initialValue = value;

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

	// Calculate the remaining power between 0 and 1 using logarithmic approach
	double remainingPower = power - (int)power;
	double remainingValue = 1.0;

	// Use logarithmic approach to calculate the remaining value
	while (remainingPower > 0.0) {
		initialValue = squareRootDouble(initialValue);
		if (remainingPower >= 0.5) {
			remainingValue *= initialValue;
			remainingPower -= 0.5;
		}
		remainingPower *= 2.0;
	}

	// Return the result
	return result * remainingResult;
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
float powerFloatFastExp(float value, float power) {

	// If the power is 0, return 1
	if (power == 0.0f)
		return 1.0f;

	// If the power is negative, return 1 / value ^ -power
	if (power < 0)
		return 1.0f / powerFloatFastExp(value, -power);

	// Store the power as an integer and the initial value
	int powerInt = (int)power;
	int initialValue = value;

	// While the power is not 0
	float result = 1.0f;
	while (powerInt > 0) {

		// If the power is odd, multiply the result by the value
		if (powerInt & 1)
			result *= value;

		// In every case, square the value and divide the power by 2
		value *= value;
		powerInt /= 2;
	}

	// Calculate the remaining power between 0 and 1 using logarithmic approach
	float remainingPower = power - (int)power;
	float remainingValue = 1.0f;

	// Use logarithmic approach to calculate the remaining value
	while (remainingPower > 0.0f) {
		initialValue = squareRootFloat(initialValue);
		if (remainingPower >= 0.5f) {
			remainingValue *= initialValue;
			remainingPower -= 0.5f;
		}
		remainingPower *= 2.0f;
	}

	// Return the result
	return result * remainingResult;
}

/**
 * @brief Use the fast method of exponentiation
 * to calculate the power of a long double
 *
 * @param value		Value to calculate the power of
 * @param power		Power to calculate
 *
 * @return long double	Result of the power
 */
long double powerLongDoubleFastExp(long double value, long double power) {

	// If the power is 0, return 1
	if (power == 0.0l)
		return 1.0l;

	// If the power is negative, return 1 / value ^ -power
	if (power < 0)
		return 1.0l / powerLongDoubleFastExp(value, -power);

	// Store the power as an integer and the initial value
	int powerInt = (int)power;
	int initialValue = value;

	// While the power is not 0
	long double result = 1.0l;
	while (powerInt > 0) {

		// If the power is odd, multiply the result by the value
		if (powerInt & 1)
			result *= value;

		// In every case, square the value and divide the power by 2
		value *= value;
		powerInt /= 2;
	}

	// Calculate the remaining power between 0 and 1 using logarithmic approach
	long double remainingPower = power - (int)power;
	long double remainingValue = 1.0l;

	// Use logarithmic approach to calculate the remaining value
	while (remainingPower > 0.0l) {
		initialValue = squareRootLongDouble(initialValue);
		if (remainingPower >= 0.5l) {
			remainingValue *= initialValue;
			remainingPower -= 0.5l;
		}
		remainingPower *= 2.0l;
	}

	// Return the result
	return result * remainingResult;
}

