#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "SyncedMemory.h"

#define BLOCK_SIZE 512
#define LSB(i) ((i) & -(i))

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct is_char{
    __host__ __device__
    bool operator()(char a, char b)
    {
        return !(b == '\n');
    }
};
struct is_space{
    __host__ __device__
    int operator()(char a)
    {
        return (a == '\n') ? 0 : 1;
    }
};

__device__ int BIT_range(int *nums, int i, int j) {
    int sum = 0;
    
    while (j > i) {
        //sum += nums[j-1];
        sum = max(sum, nums[j-1]);
		j -= LSB(j);
    }
    
    while (i > j) {
		//sum -= nums[i-1];
        sum = max(sum, nums[j-1]);
		i -= LSB(i);
	}
    
    return sum;
}

__global__ void maxScan(const char *g_idata, int *g_odata, int text_size, int *auxArray)
{
    __shared__ int nums[BLOCK_SIZE];  // allocated on invocation
    __shared__ int flags[BLOCK_SIZE];
    // __shared__ int trees[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    // initial value
    if (i < text_size) {
        //nums[tid] = (g_idata[i] == '\n') ? 0 : 1;
        if (i == 0) {
            flags[tid] = 1;
        }
        else {
            flags[tid] = (g_idata[i - 1] == '\n' || g_idata[i] == '\n') ? 1 : 0;
        }
    }
    else {
        //nums[tid] = 0;
        flags[tid] = 1;
    }
    
    if (flags[tid] == 1) {
        nums[tid] = i;
    }
    else {
        nums[tid] = 0;
    }
    
    int offset = 1;
    for (int d = BLOCK_SIZE>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            //nums[bi] += nums[ai];
            nums[bi] = max(nums[bi], nums[ai]);
        }
        offset *= 2;
    }
    
    __syncthreads();
    
    if (i < text_size) {
        g_odata[i] = BIT_range(nums, 0, tid + 1);
    }
    
    if (tid == 0) {
        auxArray[blockIdx.x] = nums[BLOCK_SIZE-1];
    }
}

__global__ void subtract(const char *g_idata, int *g_odata, int text_size, int *auxArray)
{
    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < text_size) {
        int auxValue = 0;
        if (blockIdx.x != 0) {
            auxValue = auxArray[blockIdx.x-1];
        }
        int startIdx = max(auxValue, g_odata[i]);
        g_odata[i] = (g_idata[i] == '\n') ? 0 : i-startIdx+1;
    }
}

__global__ void prescan(int *g_idata, int *g_odata, const int text_size)
{
    __shared__ int nums[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < text_size) {
        nums[tid] = g_idata[i];
    }
    
    int offset = 1;
    for (int d = BLOCK_SIZE>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            nums[bi] = max(nums[bi], nums[ai]);
        }
        offset *= 2;
    }
    
    __syncthreads();
    
    if (i < text_size) {
        g_odata[i] = BIT_range(nums, 0, tid + 1);
    }
    
}

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::device_ptr<const char> keys = thrust::device_pointer_cast(text);
    thrust::device_ptr<int> val = thrust::device_pointer_cast(pos);
    thrust::transform(keys, keys + text_size, val, is_space());
    thrust::inclusive_scan_by_key(keys, keys + text_size, val, val, is_char());
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    const unsigned int numBlocks = (text_size-1)/BLOCK_SIZE+1;
    MemoryBuffer<int> auxBuf(numBlocks);
    auto mb = auxBuf.CreateSync(numBlocks);
    int *auxArray = mb.get_gpu_wo();
    cudaMemset(auxArray, 0, sizeof(int)*numBlocks);
    printf("numBlocks: %d\n", numBlocks);
    printf("text size: %d\n", text_size);
    
    maxScan<<<numBlocks, BLOCK_SIZE >>>(text, pos, text_size, auxArray);
    
    //int auxBlockSize = 511/BLOCK_SIZE + 1;
    //prescan<<<(numBlocks-1)/auxBlockSize+1, BLOCK_SIZE >>>(auxArray, auxArray, numBlocks);
    
    subtract<<<numBlocks, BLOCK_SIZE >>>(text, pos, text_size, auxArray);
    
    /*thrust::device_ptr<int> aux = thrust::device_pointer_cast(auxArray);
    for (unsigned i=0; i<numBlocks; ++i) {
        std::cout<<aux[i]<<" ";
    }
    std::cout<<std::endl;
    thrust::device_ptr<int> val = thrust::device_pointer_cast(pos);
    for (unsigned i=0; i<text_size; ++i) {
        std::cout<<val[i]<<" ";
    }
    std::cout<<std::endl;*/
    
    
    
    
}
