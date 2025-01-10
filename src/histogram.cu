#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_PER_BLOCK 1024

inline __device__ int findBin(int min, int max, size_t bin_count, int val);

__global__ void histogramAtomicKernel(int* d_out, int* d_in, int min, int max, size_t bin_count, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&d_out[findBin(min, max, bin_count, d_in[idx])], 1);
    }
}

cudaError_t histogramAtomicParallel(int* out, int* in, int min, int max, size_t bin_count, size_t size) {
    cudaError_t cudaStatus;

    int* d_out;
    int* d_in;

    cudaStatus = cudaMalloc(&d_out, bin_count * sizeof(int));
    cudaStatus = cudaMalloc(&d_in, size * sizeof(int));

    cudaStatus = cudaMemset(d_out, 0, bin_count * sizeof(int));
    cudaStatus = cudaMemcpy(d_in, in, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridDim((size + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK);
    dim3 blockDim(gridDim.x > 1 ? MAX_PER_BLOCK : size);

    histogramAtomicKernel<<<gridDim, blockDim>>>(d_out, d_in, min, max, bin_count, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramAtomicKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Exit;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Exit;
    }

    cudaStatus = cudaMemcpy(out, d_out, bin_count * sizeof(int), cudaMemcpyDeviceToHost);

Exit:
    cudaFree(d_out);
    cudaFree(d_in);
    return cudaStatus;
}

inline __device__ int findBin(int min, int max, size_t bin_count, int val) {
    return ((float)val - min) / ((float)max - min) * bin_count;
}

/*
int main() {
	int min = 5;
	int max = 15;
	int bin_count = 5;
	int size = 10;

	int in[] = { 6, 7, 8, 8, 9, 14, 10, 8, 5, 6 };
	int out[bin_count];
	printArray(in, size);
	histogramAtomicParallel(out, in, min, max, bin_count, size);
	printArray(out, bin_count);
}
*/