
#ifndef __NEURAL_CONFIG_H__
#define __NEURAL_CONFIG_H__

// Type of values used in the neural network
#define NN_TYPE 1
#if NN_TYPE == 0
typedef float nn_type;
#elif NN_TYPE == 1
typedef double nn_type;
#else
typedef long double nn_type;
#endif


#endif

