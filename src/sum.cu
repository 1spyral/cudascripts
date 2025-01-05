#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helpers.h"

#define MAX_PER_BLOCK 2048

cudaError_t runKernel(int &out, int* in, size_t size);

__global__ void sumKernel(int* d_out, int* d_in, size_t size) {
    extern __shared__ int sums[];

    int idx = threadIdx.x;

    if (size == 0) {
        *d_out = 0;
        return;
    }
    
    size_t steps = ceil(log2f(size));

    if (steps == 0) {
        *d_out = d_in[0];
        return;
    }
    
    if (steps == 1) {
        *d_out = d_in[0] + d_in[1];
        return;
    }
    if (idx * 2 + 1 >= size) {
        sums[idx] = d_in[idx * 2];
    } else {
        sums[idx] = d_in[idx * 2] + d_in[idx * 2 + 1];
    }
    __syncthreads();

    for (size_t i = 1; i < steps - 1; i++) {
        int step = 1 << i;
        int half_step = step >> 1;
        if (idx < size << 1 << i && idx * step + half_step < (size + 1) >> 1) {
            sums[idx * step] += sums[idx * step + half_step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *d_out = sums[0] + sums[1 << steps >> 2];
    }
}

cudaError_t sumParallel(int &out, int* in, size_t size) {
    cudaError_t cudaStatus;

    if (size == 0) {
        out = 0;
        goto Error;
    }

    if (size <= MAX_PER_BLOCK) {
        cudaStatus = runKernel(out, in, size);
    } else {
        size_t subprocesses = size / MAX_PER_BLOCK;
        if (size % MAX_PER_BLOCK > 0) {
            subprocesses++;
        }
        int sums[subprocesses];
        size_t i;
        for (i = 0; i < subprocesses - 1; i++) {
            cudaStatus = runKernel(sums[i], &in[i * MAX_PER_BLOCK], MAX_PER_BLOCK);
        }
        size_t remaining = size % MAX_PER_BLOCK;
        cudaStatus = runKernel(sums[i], &in[i], remaining == 0 ? MAX_PER_BLOCK: remaining);
        cudaStatus = sumParallel(out, sums, subprocesses);
    }
Error:
    return cudaStatus;
}

cudaError_t runKernel(int &out, int* in, size_t size) {
    cudaError_t cudaStatus;

    int* d_in;
    int* d_out;

    cudaStatus = cudaMalloc((void**) &d_in, size * sizeof(int));
    cudaStatus = cudaMalloc((void**) &d_out, sizeof(int));

    cudaStatus = cudaMemcpy(d_in, in, size * sizeof(int), cudaMemcpyHostToDevice);

    sumKernel<<<1, (size + 1) >> 1, ((size + 1) >> 1) * sizeof(int)>>>(d_out, d_in, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

Error:
    cudaFree(d_in);
    cudaFree(d_out);
    return cudaStatus;
}

/*
int main() {
	int sum;
	int size = 4096;
	int arr[size];
	for (int i = 0; i < size; i++) {
		arr[i] = 1;
	}
	sumParallel(sum, arr, size);
	std::cout << sum << " " << size << std::endl;
	// for (int size = 0; size <= INT_MAX; size++) {
	// 	int sum;
	// 	int arr[size];
	// 	for (int i = 0; i < size; i++) {
	// 		arr[i] = 1;
	// 	}
	// 	sumParallel(sum, arr, size);
	// 	assert(sum == size);
	// 	std::cout << sum << " " << size << std::endl;
	// }
}
*/