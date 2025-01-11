#include <stdio.h>
#include <math.h>

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

__global__ void histogramReductionKernel(int* d_out, int* d_in, int min, int max, size_t bin_count, size_t size, size_t load) {
    extern __shared__ int shared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t threads = (size + load - 1) / load;
    int blocks = (threads + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK;
    int block_size = MAX_PER_BLOCK;
    if (blockIdx.x == blocks - 1) {
        block_size = threads - (gridDim.x - 1) * block_size;
    }
    
    if (idx < threads) {
        for (int i = idx * load; i < (idx + 1) * load && i < size; i++) {
            shared[findBin(min, max, bin_count, d_in[i]) + bin_count * threadIdx.x] += 1;
        }
    }

    size_t steps = ceil(log2f(block_size));
    for (int i = 0; i < bin_count; i++) {
        if (steps == 0 && idx < threads) {
            atomicAdd(&d_out[i], shared[i + bin_count * threadIdx.x]);
            continue;
        }
        int step = 2;
        int half_step = 1;

        __syncthreads();
        for (int j = 1; j < steps; j++) {
            if ((threadIdx.x & (step - 1)) == 0 && threadIdx.x < threads) {
                shared[i + bin_count * threadIdx.x] += shared[i + bin_count * (threadIdx.x + half_step)];
            }
            step <<= 1;
            half_step <<= 1;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(&d_out[i], shared[i + bin_count * threadIdx.x] + shared[i + bin_count * (threadIdx.x + half_step)]);
        }
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

cudaError_t histogramReductionParallel(int* out, int* in, int min, int max, size_t bin_count, size_t size, size_t load = 10) {
    cudaError_t cudaStatus;

    int* d_out;
    int* d_in;

    size_t threads = (size + load - 1) / load;

    dim3 gridDim((threads + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK);
    dim3 blockDim(gridDim.x > 1 ? MAX_PER_BLOCK: threads);
    size_t sharedMem(bin_count * threads * sizeof(int));

    cudaStatus = cudaMalloc(&d_out, bin_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Exit;
    }
    cudaStatus = cudaMalloc(&d_in, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Exit;
    }

    cudaStatus = cudaMemset(d_out, 0, bin_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Exit;
    }
    cudaStatus = cudaMemcpy(d_in, in, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Exit;
    }

    histogramReductionKernel<<<gridDim, blockDim, sharedMem>>>(d_out, d_in, min, max, bin_count, size, load);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramReductionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

/*
int main() {
	int min = 5;
	int max = 15;
	int bin_count = 5;
	int size = 10;

	int in[] = { 6, 7, 8, 8, 9, 14, 10, 8, 5, 6 };
	int out[bin_count];
	printArray(in, size);
	histogramReductionParallel(out, in, min, max, bin_count, size, 2);
	printArray(out, bin_count);
}
*/