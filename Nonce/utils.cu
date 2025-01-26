#include <cuda_runtime.h>
#include "utils.cuh"

/**
 * @brief Converts an integer into a string representation.
 *
 * This function takes an integer and converts it into a string,
 * storing the result in the provided buffer.
 *
 * @param num The integer to convert.
 * @param str Buffer where the string representation of the number will be stored.
 * @return The length of the generated string.
 */
__device__ int cudaIntToStr(uint32_t num, char* str) {
    int i = 0;
    if (num == 0) {
        str[i++] = '0';
        str[i] = '\0';
        return i;
    }

    size_t n = num;
    int len = 0;
    while (n > 0) {
        len++;
        n /= 10;
    }

    for (int j = len - 1; j >= 0; j--) {
        str[j] = '0' + (num % 10);
        num /= 10;
    }

    str[len] = '\0';
    return len;
}

/**
 * @brief Computes the length of a null-terminated string in device code.
 *
 * This function iterates through the string until it finds the null terminator (`\0`),
 * returning the total length. It is designed to be used within CUDA device functions
 * since standard `strlen()` is not available in CUDA device code.
 *
 * @param str Pointer to the null-terminated string.
 * @return The length of the string (excluding the null terminator).
 */
__device__  int cudaStrlen(const char* str) {
	int length = 0;
	while (str[length] != '\0') {
		length++;
	}
	return length;
}
