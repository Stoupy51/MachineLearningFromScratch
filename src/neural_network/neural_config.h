
#ifndef __NEURAL_CONFIG_H__
#define __NEURAL_CONFIG_H__

// Type of values used in the neural network
#define NN_TYPE 2	// 0: float, 1: double, 2: long double
#if NN_TYPE == 0

typedef float nn_type;
#define NN_FORMAT "f"

#elif NN_TYPE == 1

typedef double nn_type;
#define NN_FORMAT "lf"

#else

typedef long double nn_type;
#define NN_FORMAT "Lf"

#endif


#endif

