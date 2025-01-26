#ifndef SHA1_CUH
#define SHA1_CUH

#include <cuda_runtime.h>

typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef unsigned long long LONG;

typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[5];
	WORD k[4];
} CUDA_SHA1_CTX;


#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define USE_MD2 1
#define USE_MD5 1
#define USE_SHA1 1
#define USE_SHA256 1

#define CUDA_HASH 1
#define OCL_HASH 0

#define ROTATE_LEFT(x, n) ((x << n) | (x >> (32 - n)))

#define SHA1_BLOCK_SIZE 20

__device__ void cuda_sha1_transform(CUDA_SHA1_CTX* ctx, const BYTE data[]);
__device__ void cuda_sha1_init(CUDA_SHA1_CTX* ctx);
__device__ void cuda_sha1_update(CUDA_SHA1_CTX* ctx, const BYTE data[], size_t len);
__device__ void cuda_sha1_final(CUDA_SHA1_CTX* ctx, BYTE hash[]);

#endif // SHA1_CUH