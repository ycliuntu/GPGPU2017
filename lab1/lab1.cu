#include "lab1.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;


__global__ void DrawY(uint8_t *frame, const float iteration) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < H and x < W) {
        uint8_t c = 0;
        float sFrame = float(NFRAME)/2.0;
        //float overHalf = (iteration > sFrame) ? 1.0 : 0.0;
        float xi = 0.0;
        float yi = 0.0;
        //float midx = float(W)*(0.5 - 0.0717*iteration*max(iteration - sFrame, 0.0)/(sFrame));
        float midx = float(W)*(0.5);
        float midy = float(H)*(0.5);
        float zoom = 1 * pow(1.005, max(iteration - sFrame, 0.0) * iteration / 12.0);
        float scalex = float(W/2)*zoom;
        float scaley = float(H/2)*zoom;
        float x0 = 1.5*(float(x) - midx)/scalex;
        float y0 = (float(y) - midy)/scaley;
        float cx = -0.675511;
        float cy = -0.097732;
        //float xtemp;
        float lengthsq = 0.0;
        float power = 1.0 * (1 + min(iteration, sFrame)/(sFrame)*4.0);
        float max_iter = 100000;
        while(c < (uint8_t)max_iter and lengthsq < 2.0*2.0) {
            float modulus = sqrt(xi*xi + yi*yi);
            float argument = atan2(yi, xi);
            modulus = pow(modulus, power);
            argument *= power;
            /*xtemp = xi*xi - yi*yi + x0;
            yi = 2*xi*yi + y0;
            xi = xtemp;*/
            xi = modulus*cos(argument) + x0 + cx;
            yi = modulus*sin(argument) + y0 + cy;
            lengthsq = xi*xi + yi*yi;
            c++;
        }
        float modulus = sqrt(lengthsq);
        //float continuous_index = (float)c;
        /*if ( c < (uint8_t)max_iter) {
            float log_zn = logf(xi*xi + yi*yi) / 2;
            continuous_index = logf( log_zn / logf(2) ) / logf(power);
            continuous_index = (float)c + 1 - continuous_index;
        }*/
        float continuous_index = (float)c + 1.0 - logf(logf(modulus)/logf(2.0)) / logf(power);
        uint8_t r = (uint8_t)(sin(0.017 * continuous_index + 8) * 200 + 55);
        uint8_t g = (uint8_t)(sin(0.011 * continuous_index + 4) * 55 + 200);
        uint8_t b = (uint8_t)(sin(0.005 * continuous_index + 1) * 55 + 200);
        /*uint8_t r = (uint8_t)((sin(0.016 * 1 * continuous_index + 1) + 1) / 2 * 255.0);
        uint8_t g = (uint8_t)((sin(0.013 * 1 * continuous_index + 2) + 1) / 2 * 255.0);
        uint8_t b = (uint8_t)((sin(0.010 * 1 * continuous_index + 3) + 1) / 2 * 255.0);*/
        
        /*float t = (float)continuous_index/max_iter;
        uint8_t r = (uint8_t)(9*(1-t)*t*t*t*255);
        uint8_t g = (uint8_t)(15*(1-t)*(1-t)*t*t*255);
        uint8_t b =  (uint8_t)(8.5*(1-t)*(1-t)*(1-t)*t*255);*/
        
        frame[y*W+x] = 0.299 * r + 0.587 * g + 0.114 * b;
        if (x%2 == 0 and y%2 == 0) {
            frame[W*H + y*W/4 + x/2]        = -0.169 * r - 0.331 * g + 0.500 * b + 128;
            frame[W*H + W*H/4 + y*W/4 + x/2] = 0.500 * r - 0.419 * g - 0.081 * b + 128;
        }
    }
}

__global__ void DrawUV(uint8_t *frame, int t) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < H/2 and x < W/2) {
        uint8_t c = 0;
        float xi = 0.0;
        float yi = 0.0;
        float x0 = (float(x*2) - 320.0)/160.0;
        float y0 = (float(y*2) - 240.0)/120.0;
        float temp, lengthsq = 0.0;
        while(c < t and lengthsq < 4.0) {
            temp = xi*xi - yi*yi + x0;
            yi = 2*xi*yi + y0;
            xi = temp;
            lengthsq = xi*xi + yi*yi;
            c++;
        }
        c = -0.169 * 0 - 0.331 * 0 + 0.500 * (c%256) + 128;
        //c = (uint8_t(0.5*c)*255/NFRAME)%256+128;
        //c = 255 - c;
        //c = (c == 0) ? 0 : 255;
        //frame[y*W/2+x] = 0.5*c + 128;
        frame[y*W/2+x] = c;
    }
    else if (y < H and x < W/2) {
        uint8_t c = 0;
        float xi = 0.0;
        float yi = 0.0;
        float x0 = (float(x*2) - 320.0)/160.0;
        float y0 = (float(y*2 - H) - 240.0)/120.0;
        float temp, lengthsq = 0.0;
        while(c < t and lengthsq < 4.0) {
            temp = xi*xi - yi*yi + x0;
            yi = 2*xi*yi + y0;
            xi = temp;
            lengthsq = xi*xi + yi*yi;
            c++;
        }
        c = 0.500 * 0 - 0.419 * 0 - 0.081 * (c%256) + 128;
        //c = (uint8_t(-0.081*c)*255/NFRAME)%256+128;
        //c = 255 - c;
        //c = (c == 0) ? 0 : 255;
        //frame[y*W/2+x] = -0.081*c + 128;
        frame[y*W/2+x] = c;
    }
}


struct Lab1VideoGenerator::Impl {
    int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
    info.w = W;
    info.h = H;
    info.n_frame = NFRAME;
    // fps = 24/1 = 24
    info.fps_n = 24;
    info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
    DrawY<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(yuv, (float)impl->t);
    //DrawUV<<<dim3((W/2-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(yuv+W*H, impl->t);
    //cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
    //cudaMemset(yuv+W*H, 128, W*H/2);
    ++(impl->t);
}

