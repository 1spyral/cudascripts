// Conversion reference:
// https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

#include <stdio.h>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>

#include "images.h"

__device__ float linearize(float gamma);
__device__ float convertToGamma(float linear);
__device__ inline float scale1(uint8_t x);
__device__ inline uint8_t scale255(float x);

__global__ void grayscaleKernel(uint8_t* d_out, uint8_t* d_in, size_t size) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) {
        return;
    }
    float b = scale1(d_in[idx * 3]);
    float g = scale1(d_in[idx * 3 + 1]);
    float r = scale1(d_in[idx * 3 + 2]);

    float linearR = linearize(r);
    float linearG = linearize(g);
    float linearB = linearize(b);

    uint8_t intensity = scale255(convertToGamma(linearR * 0.2126 + linearG * 0.7152 + 0.0722 * linearB));

    d_out[idx] = intensity;
}

cudaError_t grayscaleParallel(cv::Mat &h_out, cv::Mat &h_in) {
    cudaError_t cudaStatus;

    size_t size = sizeInPixels(h_in);

    uint8_t* h_arr_in = flattenColor(h_in);
    uint8_t* h_arr_out = new uint8_t[size];

    uint8_t* d_arr_in;
    uint8_t* d_arr_out;

    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cudaStatus = cudaMalloc((void**)&d_arr_in, size * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_arr_in!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_arr_out, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_arr_out!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_arr_in, h_arr_in, size * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_arr_in!");
        goto Error;
    }

    grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr_out, d_arr_in, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "grayscaleKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching grayscaleKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_arr_out, d_arr_out, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for h_arr_out!");
        goto Error;
    }

    // Copy the output array back to the cv::Mat
    h_out = cv::Mat(h_in.rows, h_in.cols, CV_8UC1, h_arr_out);

Error:
    cudaFree(d_arr_in);
    cudaFree(d_arr_out);
    return cudaStatus;
}

__device__ float linearize(float gamma) {
    if (gamma <= 0.04045) {
        return gamma / 12.92;
    }
    return pow((gamma + 0.055) / 1.055, 2.4);
}

__device__ float convertToGamma(float linear) {
    if (linear <= 0.0031308) {
        return 12.92 * linear;
    }
    return 1.055 * pow(linear, 1 / 2.4) - 0.055;
}

__device__ inline float scale1(uint8_t x) {
    return x / 255.0;
}

__device__ inline uint8_t scale255(float x) {
    return (uint8_t)(x * 255);
}

/*
int main() {
	std::string PATH = "images/starrynight.jpg";
	cv::Mat img = getImage(PATH);
	previewImage(img, 500, 500);
	grayscaleParallel(img, img);
	previewImage(img, 500, 500);
}
*/