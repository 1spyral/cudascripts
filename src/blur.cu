#include <math.h>
#include <cassert>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>

#include "images.h"

#define BLOCK_SIDE_LENGTH 32

__device__ inline bool isValid(int x, int y, int width, int height);

__global__ void unweightedBlurKernel(u_int8_t* d_out, u_int8_t* d_in, int width, int height, int radius) {
    extern __shared__ int pixels[];

    const size_t side_length = BLOCK_SIDE_LENGTH + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        __syncthreads();

        return;
    }

    int startX = threadIdx.x == 0 ? 0 : threadIdx.x + radius;
    int endX = threadIdx.x == blockDim.x - 1 ? side_length: threadIdx.x + radius + 1;
    int startY = threadIdx.y == 0 ? 0 : threadIdx.y + radius;
    int endY = threadIdx.y == blockDim.y - 1 ? side_length: threadIdx.y + radius + 1;

    for (int i = startX; i < endX; i++) {
        for (int j = startY; j < endY; j++) {
            int imageX = i + blockIdx.x * blockDim.x - radius;
            int imageY = j + blockIdx.y * blockDim.y - radius;
            if (isValid(imageX, imageY, width, height)) {
                pixels[(i + j * side_length) * 3] = d_in[(imageX + imageY * width) * 3];
                pixels[(i + j * side_length) * 3 + 1] = d_in[(imageX + imageY * width) * 3 + 1];
                pixels[(i + j * side_length) * 3 + 2] = d_in[(imageX + imageY * width) * 3 + 2];
            } else {
                pixels[(i + j * side_length) * 3] = -1;
                pixels[(i + j * side_length) * 3 + 1] = -1;
                pixels[(i + j * side_length) * 3 + 2] = -1;
            }
        }
    }
    __syncthreads();

    int sumB = 0;
    int sumG = 0;
    int sumR = 0;
    int count = 0;

    for (int i = threadIdx.x; i < threadIdx.x + 2 * radius; i++) {
        for (int j = threadIdx.y; j < threadIdx.y + 2 * radius; j++) {
            if (pixels[(i + j * side_length) * 3] == -1) {
                continue;
            }
            sumB += pixels[(i + j * side_length) * 3];
            sumG += pixels[(i + j * side_length) * 3 + 1];
            sumR += pixels[(i + j * side_length) * 3 + 2];
            count++;
        }
    }
    
    d_out[(x + y * width) * 3] = sumB / count;
    d_out[(x + y * width) * 3 + 1] = sumG / count;
    d_out[(x + y * width) * 3 + 2] = sumR / count;
}

__global__ void gaussianBlurKernel(u_int8_t* d_out, u_int8_t* d_in, size_t width, size_t height, int radius) {
    extern __shared__ int pixels[];

    size_t side_length = BLOCK_SIDE_LENGTH + 2 * radius;
    float sigma = 0.3 * (radius - 1) + 0.8;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        __syncthreads();

        return;
    }

    int startX = threadIdx.x == 0 ? 0 : threadIdx.x + radius;
    int endX = threadIdx.x == blockDim.x - 1 ? side_length: threadIdx.x + radius + 1;
    int startY = threadIdx.y == 0 ? 0 : threadIdx.y + radius;
    int endY = threadIdx.y == blockDim.y - 1 ? side_length: threadIdx.y + radius + 1;

    for (int i = startX; i < endX; i++) {
        for (int j = startY; j < endY; j++) {
            int imageX = i + blockIdx.x * blockDim.x - radius;
            int imageY = j + blockIdx.y * blockDim.y - radius;
            if (isValid(imageX, imageY, width, height)) {
                pixels[(i + j * side_length) * 3] = d_in[(imageX + imageY * width) * 3];
                pixels[(i + j * side_length) * 3 + 1] = d_in[(imageX + imageY * width) * 3 + 1];
                pixels[(i + j * side_length) * 3 + 2] = d_in[(imageX + imageY * width) * 3 + 2];
            } else {
                pixels[(i + j * side_length) * 3] = -1;
                pixels[(i + j * side_length) * 3 + 1] = -1;
                pixels[(i + j * side_length) * 3 + 2] = -1;
            }
        }
    }
    __syncthreads();

    float sumB = 0;
    float sumG = 0;
    float sumR = 0;
    float weight = 0;

    for (int i = threadIdx.x; i < threadIdx.x + 2 * radius; i++) {
        for (int j = threadIdx.y; j < threadIdx.y + 2 * radius; j++) {
            if (pixels[(i + j * side_length) * 3] == -1) {
                continue;
            }
            float x_offset = threadIdx.x + radius - i;
            float y_offset = threadIdx.y + radius - j;
            float gaussian_weight = 1.0 / (2 * M_PI * sigma * sigma) * exp(-(x_offset * x_offset + y_offset * y_offset) / (2 * sigma * sigma)); 

            sumB += pixels[(i + j * side_length) * 3] * gaussian_weight;
            sumG += pixels[(i + j * side_length) * 3 + 1] * gaussian_weight;
            sumR += pixels[(i + j * side_length) * 3 + 2] * gaussian_weight;
            weight += gaussian_weight;
        }
    }
    d_out[(x + y * width) * 3] = sumB / weight;
    d_out[(x + y * width) * 3 + 1] = sumG / weight;
    d_out[(x + y * width) * 3 + 2] = sumR / weight;
}

cudaError_t unweightedBlurParallel(cv::Mat &out, cv::Mat &in, int radius = 5) {
    cudaError_t cudaStatus;

    uint8_t* h_in;
    size_t size = flattenColor(h_in, in);
    uint8_t* h_out = new uint8_t[size];

    size_t width = in.cols;
    size_t height = in.rows;

    uint8_t* d_in;
    uint8_t* d_out;

    cudaStatus = cudaMalloc((void**) &d_in, size * sizeof(uint8_t));
    cudaStatus = cudaMalloc((void**) &d_out, size * sizeof(uint8_t));

    cudaStatus = cudaMemcpy(d_in, h_in, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIDE_LENGTH, BLOCK_SIDE_LENGTH);
    dim3 gridDim((width + BLOCK_SIDE_LENGTH - 1) / BLOCK_SIDE_LENGTH, (height + BLOCK_SIDE_LENGTH - 1) / BLOCK_SIDE_LENGTH);
    int sharedMem((BLOCK_SIDE_LENGTH + 2 * radius) * (BLOCK_SIDE_LENGTH + 2 * radius) * sizeof(int) * 3);

    unweightedBlurKernel<<<gridDim, blockDim, sharedMem>>>(d_out, d_in, width, height, radius);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "unweightedBlurKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        fprintf(stderr, "NOTE: with the current arguments, %d bytes of shared block memory are being used\n", sharedMem);
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching unweightedBlurKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_out, d_out, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for h_out!");
        goto Error;
    }

    out = cv::Mat(height, width, CV_8UC3, h_out);

Error:
    cudaFree(d_in);
    cudaFree(d_out);
    return cudaStatus;
}

cudaError_t gaussianBlurParallel(cv::Mat &out, cv::Mat &in, int radius = 5) {
    cudaError_t cudaStatus;

    uint8_t* h_in;
    size_t size = flattenColor(h_in, in);
    uint8_t* h_out = new uint8_t[size];

    size_t width = in.cols;
    size_t height = in.rows;

    uint8_t* d_in;
    uint8_t* d_out;

    cudaStatus = cudaMalloc((void**) &d_in, size * sizeof(uint8_t));
    cudaStatus = cudaMalloc((void**) &d_out, size * sizeof(uint8_t));

    cudaStatus = cudaMemcpy(d_in, h_in, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIDE_LENGTH, BLOCK_SIDE_LENGTH);
    dim3 gridDim((width + BLOCK_SIDE_LENGTH - 1) / BLOCK_SIDE_LENGTH, (height + BLOCK_SIDE_LENGTH - 1) / BLOCK_SIDE_LENGTH);
    int sharedMem((BLOCK_SIDE_LENGTH + 2 * radius) * (BLOCK_SIDE_LENGTH + 2 * radius) * sizeof(int) * 3);

    gaussianBlurKernel<<<gridDim, blockDim, sharedMem>>>(d_out, d_in, width, height, radius);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "gaussianBlurKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        fprintf(stderr, "NOTE: with the current arguments, %d bytes of shared block memory are being used\n", sharedMem);
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching gaussianBlurKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_out, d_out, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for h_out!");
        goto Error;
    }

    out = cv::Mat(height, width, CV_8UC3, h_out);

Error:
    cudaFree(d_in);
    cudaFree(d_out);
    return cudaStatus;
}

__device__ inline bool isValid(int x, int y, int width, int height) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

/*
int main() {
	std::string PATH = "images/starrynight.jpg";
	cv::Mat img = getImage(PATH);
	previewImage(img, 500, 500);
	unweightedBlurParallel(img, img, 16);
	previewImage(img, 500, 500);
}
*/

/*
int main() {
	std::string PATH = "images/starrynight.jpg";
	cv::Mat img = getImage(PATH);
	previewImage(img, 500, 500);
    while (true) {
        gaussianBlurParallel(img, img, 5);
	    previewImage(img, 500, 500);
    }
}
*/