
#ifndef __NEURAL_CONFIG_H__
#define __NEURAL_CONFIG_H__

// Type of values used in the neural network
#define NN_TYPE 1	// 0: float, 1: double, 2: long double
#if NN_TYPE == 0

typedef float nn_type;
#define NN_FORMAT "f"
#define NN_STRING "float"
#define NN_EPSILON_STR "1e-6f"

#elif NN_TYPE == 1

typedef double nn_type;
#define NN_FORMAT "lf"
#define NN_STRING "double"
#define NN_EPSILON_STR "1e-15"

#else

typedef long double nn_type;
#define NN_FORMAT "Lf"
#define NN_STRING "long double"
#define NN_EPSILON_STR "1e-15l"

#endif


#endif

