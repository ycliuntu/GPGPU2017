#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct is_char{
    __host__ __device__
    bool operator()(char a, char b)
    {
        return !(b == '\n' || b == ' ');
    }
};
struct is_space{
    __host__ __device__
    int operator()(char a)
    {
        return (a == ' ' || a == '\n') ? 0 : 1;
    }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::device_ptr<const char> keys = thrust::device_pointer_cast(text);
    thrust::device_ptr<int> val = thrust::device_pointer_cast(pos);
    is_space opint;
    thrust::transform(keys, keys + text_size, val, opint);
    is_char op;
    thrust::inclusive_scan_by_key(keys, keys + text_size, val, val, op);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
}
