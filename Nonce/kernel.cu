#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include "utils.cuh"
#include "sha1.cuh"

#include <stdio.h>
#include <iostream>
#include <cstring>

__constant__ char d_const_input[256];

__device__ void kernel_sha1_hash(CUDA_SHA1_CTX ctx, const BYTE* input, size_t input_len, BYTE* output_hash) {
	cuda_sha1_init(&ctx);
	cuda_sha1_update(&ctx, input, input_len);
	cuda_sha1_final(&ctx, output_hash);
}


/**
 * @brief CUDA kernel to find a nonce that satisfies the hash condition.
 *
 * Each thread attempts different nonce values and checks if the resulting SHA-1 hash contains a specified number of trailing zeros.
 *
 * @param result Pointer to the variable where the first valid nonce will be stored.
 * @param found Pointer to an atomic flag indicating if a nonce has been found.
 * @param input The base input string used for hashing.
 * @param zeros_to_find The number of trailing zeros required in the hash.
 */
__global__ void find_nonce(uint32_t* result, int* found, int zeros_to_find) {
    uint32_t nonce = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_SHA1_CTX ctx;
    unsigned char buffer[256];
    unsigned char hash[20];

    int input_len = cudaStrlen(d_const_input);
    memcpy(buffer, d_const_input, input_len);

    while (!atomicOr(found, 0)) {
        // Convert nonce to string
        char nonce_str[12];
        int nonce_len = cudaIntToStr(nonce, nonce_str);

        // Copy nonce to buffer
        memcpy(buffer + input_len, nonce_str, nonce_len);
        int total_len = input_len + nonce_len;

        // compute sha1 hash
        kernel_sha1_hash(ctx, buffer, total_len, hash);

        // check if it's ok
        bool matches = true;
        for (int i = 0; i < zeros_to_find; i++) {
            if (hash[19 - i] != 0) {
                matches = false;
                break;
            }
        }

        // use atomicCAS so that the first thread that found the nonce is taken into consideration
        if (matches && atomicCAS(found, 0, 1) == 0) {
            *result = nonce;
        }

        nonce += blockDim.x * gridDim.x;
    }
}



