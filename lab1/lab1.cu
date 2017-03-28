#include "lab1.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;


__global__ void DrawY(uint8_t *frame, const int iteration) {
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H and x < W) {
		uint8_t c = 0;
		double xi = 0.0;
		double yi = 0.0;
		double x0 = (double(x) - 320.0)/160.0;
		double y0 = (double(y) - 240.0)/120.0;
		double temp, lengthsq = 0.0;
		while(c < iteration/4 and lengthsq < 4.0) {
		    temp = xi*xi - yi*yi + x0;
		    yi = 2*xi*yi + y0;
		    xi = temp;
		    lengthsq = xi*xi + yi*yi;
		    c++;
		}
		double t = (double)c/(double)iteration*4;
		uint8_t r = (uint8_t)(9*(1-t)*t*t*t*255);
        uint8_t g = (uint8_t)(15*(1-t)*(1-t)*t*t*255);
        uint8_t b =  (uint8_t)(8.5*(1-t)*(1-t)*(1-t)*t*255);
		//c = t*255/NFRAME;
		//c = 0.299 * 127 + 0.587 * 127 + 0.114 * (c%256);
		//c = (uint8_t(0.114*c)*255/NFRAME)%256;
		//c = 255 - c;
		//uint8_t b = c%256;
		frame[y*W+x] = 0.299 * r + 0.587 * g + 0.114 * b;
		if (x%2 == 0 and y%2 == 0) {
		    frame[W*H + y*W/4 + x/2]            = -0.169 * r - 0.331 * g + 0.500 * b + 128;
		    frame[W*H + W*H/4 + y*W/4 + x/2]    = 0.500 * r - 0.419 * g - 0.081 * b + 128;
		}
	}
}

__global__ void DrawUV(uint8_t *frame, int t) {
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H/2 and x < W/2) {
		uint8_t c = 0;
		double xi = 0.0;
		double yi = 0.0;
		double x0 = (double(x*2) - 320.0)/160.0;
		double y0 = (double(y*2) - 240.0)/120.0;
		double temp, lengthsq = 0.0;
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
		double xi = 0.0;
		double yi = 0.0;
		double x0 = (double(x*2) - 320.0)/160.0;
		double y0 = (double(y*2 - H) - 240.0)/120.0;
		double temp, lengthsq = 0.0;
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
	DrawY<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(yuv, impl->t);
	//DrawUV<<<dim3((W/2-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(yuv+W*H, impl->t);
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	//cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}

