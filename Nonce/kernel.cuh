#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include "sha1.cuh"

// if i don't add "extern" then it will tell me that it is defined in multiple object files. (because it is defined ina  header file and included in multiple compilation units)
extern __constant__ char d_const_input[256];

__global__ void find_nonce(uint32_t* result, int* found, int zeros_to_find);

#endif // KERNEL_CUH
