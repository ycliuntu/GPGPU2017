#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__constant__ int directions[4][2] = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };

__device__ __host__ bool white(float val) { return val > 127.0f; }
__device__ __host__ bool interior(int x, int w, int h, int channel = 1) { return x >= 0 && x < w * h * channel ;}
__device__ __host__ int clipvalue(int x, int lb, int ub) {return min(ub-1, max(x, lb));}

//__device__ __host__

__global__ void SimpleClone(
        const float *background,
        const float *target,
        const float *mask,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        )
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt*yt+xt;
    if (yt < ht and xt < wt and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = target[curt*3+0];
            output[curb*3+1] = target[curt*3+1];
            output[curb*3+2] = target[curt*3+2];
        }
    }
}

__global__ void CalculateFixed(const float *background, const float *target, const float *mask, float *fixed, int wb, int hb, int wt, int ht, int oy, int ox, int stride = 1) {
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int syt = stride*yt;
    const int sxt = stride*xt;
    const int curt = wt*syt+sxt;
    if (syt < ht and sxt < wt and white(mask[curt])) {
        float tmpfixedr = 0.0f;
        float tmpfixedg = 0.0f;
        float tmpfixedb = 0.0f;
        for (int i=0; i<4; ++i) {
            int neighbor_y = syt + stride*directions[i][0];
            int neighbor_x = sxt + stride*directions[i][1];
            const int tneighbor = wt*neighbor_y + neighbor_x;
            if (0 <= neighbor_y and neighbor_y < ht and 0 <= neighbor_x and neighbor_x < wt) {
                tmpfixedr += target[curt*3 + 0] - target[tneighbor*3 + 0];
                tmpfixedg += target[curt*3 + 1] - target[tneighbor*3 + 1];
                tmpfixedb += target[curt*3 + 2] - target[tneighbor*3 + 2];
            }
            const int bgneighbor = wb*(clipvalue(oy + stride*(yt + directions[i][0]), 0, hb)) + clipvalue(ox + stride*(xt + directions[i][1]), 0, wb);
            if (!(0 <= neighbor_y and neighbor_y < ht and 0 <= neighbor_x and neighbor_x < wt) or !white(mask[tneighbor])) {
                tmpfixedr += background[bgneighbor*3 + 0];
                tmpfixedg += background[bgneighbor*3 + 1];
                tmpfixedb += background[bgneighbor*3 + 2];
            }
        }
        fixed[curt*3 + 0] = tmpfixedr;
        fixed[curt*3 + 1] = tmpfixedg;
        fixed[curt*3 + 2] = tmpfixedb;
    }
}

__global__ void PoissonImageCloningIteration(const float *fixed, const float *mask, const float *buf1, float *buf2, int wt, int ht, int stride = 1, float omega = 1) {
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int syt = stride*yt;
    const int sxt = stride*xt;
    const int curt = wt*syt+sxt;
    if (syt < ht and sxt < wt and white(mask[curt])) {
        float newr = fixed[curt*3 + 0];
        float newg = fixed[curt*3 + 1];
        float newb = fixed[curt*3 + 2];

        for (int i=0; i<4; ++i) {
            int neighbor_y = syt + stride*directions[i][0];
            int neighbor_x = sxt + stride*directions[i][1];
            const int tneighbor = wt*neighbor_y + neighbor_x;
            if (0 <= neighbor_y and neighbor_y < ht and 0 <= neighbor_x and neighbor_x < wt and white(mask[tneighbor])) {
                newr += buf1[tneighbor*3 + 0];
                newg += buf1[tneighbor*3 + 1];
                newb += buf1[tneighbor*3 + 2];
            }
        }
        buf2[curt*3 + 0] = newr * 0.25f;
        buf2[curt*3 + 1] = newg * 0.25f;
        buf2[curt*3 + 2] = newb * 0.25f;
    }
}

__global__ void scaleUp(
        const float *mask,
        float *output,
        const int wb, const int hb, 
        const int wt, const int ht,
        const int oy, const int ox,
        int stride
        )
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt*yt+xt;
    if (yt < ht and xt < wt and white(mask[curt])) {
        int basey = (yt/stride) * stride;
        int basex = (xt/stride) * stride;
        int baset = basey*wt + basex;
        if (0 <= basey and basey < ht and 0 <= basex and basex < wt) {
            output[curt*3 + 0] = output[baset*3 + 0];
            output[curt*3 + 1] = output[baset*3 + 1];
            output[curt*3 + 2] = output[baset*3 + 2];
        }
    }
}

void PoissonImageCloning(
        const float *background,
        const float *target,
        const float *mask,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        )
{
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    //SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
    //    background, target, mask, output,
    //    wb, hb, wt, ht, oy, ox
    //);
    
    // set up
    float *fixed, *buf1, *buf2;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

    // initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    
    //CalculateFixed<<<gdim, bdim>>>( background, target, mask, fixed, wb, hb, wt, ht, oy, ox);
    cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    
    // iterate
    //CalculateFixed<<<gdim, bdim>>>( output, target, mask, fixed, wb, hb, wt, ht, oy, ox, 1);
    //for (int i = 0; i < 10000; ++i) {
    //    PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf1, buf2, wt, ht, 1, 1);
    //    PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf2, buf1, wt, ht, 1, 1);
    //}
    // scale up
    for (int scale=16; scale>0; scale>>=1) {
        CalculateFixed<<<gdim, bdim>>>( output, target, mask, fixed, wb, hb, wt, ht, oy, ox, scale);
        for (int iter = 0; iter < 100; ++iter) {
            PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf1, buf2, wt, ht, scale, 1);
            PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf2, buf1, wt, ht, scale, 1);
        }
        if (scale == 1) break;
        //SimpleClone<<<gdim, bdim>>>(background, buf1 , mask, output, wb, hb, wt, ht, oy, ox);
        scaleUp<<<gdim, bdim>>>(mask, buf1, wb, hb, wt, ht, oy, ox, scale);
    }
    
    
    // copy the image back
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    SimpleClone<<<gdim, bdim>>>(background, buf1 , mask, output, wb, hb, wt, ht, oy, ox);
    
    // clean up
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
    
}
