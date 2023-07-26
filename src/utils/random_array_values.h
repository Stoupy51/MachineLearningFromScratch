
#ifndef __RANDOM_ARRAY_VALUES_H__
#define __RANDOM_ARRAY_VALUES_H__

#include "../neural_network/neural_config.h"

// Private functions

int fillRandomFloatArrayCPUThreads(nn_type* array, unsigned long long size, nn_type min, nn_type max);



// One call functions

void fillRandomFloatArray(nn_type* array, unsigned long long size, nn_type min, nn_type max);

#endif

