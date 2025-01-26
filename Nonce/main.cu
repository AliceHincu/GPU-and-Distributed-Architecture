#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cstring>

#include "kernel.cuh" 

#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define ZEROS_TO_FIND 2

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 1. modify path: $env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64;" + $env:PATH
// 2. command to compile:  nvcc -rdc=true main.cu sha1.cu utils.cu kernel.cu  (https://stackoverflow.com/questions/27590166/how-to-compile-multiple-files-in-cuda)
// 3. command to start: ./a
// 4. verify with this link: http://www.sha1-online.com/
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    //printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    //printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    const int THREADS_PER_BLOCK = min(256, prop.maxThreadsPerBlock); // 256 good default because it balances parallelism and register usage
    const int MAX_BLOCKS = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / THREADS_PER_BLOCK);
    printf("Using MAX_BLOCKS: %d, THREADS_PER_BLOCK: %d\n", MAX_BLOCKS, THREADS_PER_BLOCK);

    const char* h_string = "abc";
    int h_found = 0;
    uint32_t h_nonce = 0;
    int h_string_length = strlen(h_string) + 1;

    uint32_t* d_nonce;
    int* d_found;

    // alocate memory on gpu
    CUDA_CHECK(cudaMalloc(&d_nonce, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
    // copy input to constant memory
    cudaMemcpyToSymbol(d_const_input, h_string, h_string_length);

    // copy data on gpu
    CUDA_CHECK(cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice));

    // start looking for nonce
    find_nonce << <MAX_BLOCKS, THREADS_PER_BLOCK >> > (d_nonce, d_found, ZEROS_TO_FIND);
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy resuly on cpu
    CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_nonce, d_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (h_found) {
        printf("Nonce found: %I32u\n", h_nonce);
    } else {
        printf("Nonce not found.\n");
    }

    cudaFree(d_nonce);
    cudaFree(d_found);

    return 0;
}

