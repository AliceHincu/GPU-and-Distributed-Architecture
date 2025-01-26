#include <cuda_runtime.h>

__device__ int cudaIntToStr(uint32_t num, char* str);
__device__ int cudaStrlen(const char* str);