#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_PER_BLOCK 3

__global__ void psaHSKernel(int *d_out, int *d_in, size_t size, bool inclusive) {
    extern __shared__ int psa[];

    size_t idx = threadIdx.x;

    size_t steps = ceil(log2f(size));

    if (steps == 0) {
        if (inclusive) {
            d_out[0] = d_in[0];
        } else {
            d_out[0] = 0;
        }
        return;
    }
    else if (steps == 1) {
        int first = d_in[0];
        if (inclusive) {
            d_out[0] = first;
            d_out[1] = first + d_in[1];
        } else {
            d_out[0] = 0;
            d_out[1] = first;
        }
        return;
    }

    size_t i = 1;
    size_t step = 1;

    if (idx == 0) {
        psa[0] = d_in[0];
    } else {
        psa[idx] = d_in[idx - i] + d_in[idx];
    }
    __syncthreads();

    for (i = 2; i < steps; i++) {
        step <<= 1;
        int addend = idx < step ? 0 : psa[idx - step];
        __syncthreads();
        psa[idx] += addend;
        __syncthreads();
    }

    step <<= 1;
    if (inclusive) {
        d_out[idx] = psa[idx] + (idx < step ? 0 : psa[idx - step]);
        __syncthreads();
    } else {
        if (idx == 0) {
            d_out[idx] = 0;
        } else if (idx - 1 < step) {
            d_out[idx] = psa[idx - 1];
        } else {
            d_out[idx] = psa[idx - 1] + psa[idx - step - 1];
        }
    }
}

__global__ void psaBKernel(int *d_out, int *d_in, size_t size, bool inclusive) {
    extern __shared__ int psa[];

}

__global__ void addKernel(int *d_arr, int num) {
    int idx = threadIdx.x;

    d_arr[idx] += num;
}

cudaError_t psaParallelHS(int *out, int *in, size_t size, bool inclusive = false) {
    cudaError_t cudaStatus;

    if (size == 0) {
        out = 0;
        goto Exit;
    }

    int* d_out;
    int* d_in;

    cudaStatus = cudaMalloc((void**) &d_out, size * sizeof(int));
    cudaStatus = cudaMalloc((void**) &d_in, size * sizeof(int));

    cudaStatus = cudaMemcpy(d_in, in, size * sizeof(int), cudaMemcpyHostToDevice);
    
    psaHSKernel<<<1, size, size * sizeof(int)>>>(d_out, d_in, size, inclusive);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "psaParallelHS launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Exit;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Exit;
    }

    cudaStatus = cudaMemcpy(out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

Exit:
    cudaFree(d_out);
    cudaFree(d_in);
    return cudaStatus;
}

cudaError_t psaParallelB(int *out, int *in, size_t size, bool inclusive = false) {
    cudaError_t cudaStatus;

    if (size == 0) {
        out = 0;
        goto Exit;
    }

Exit:

    return cudaStatus;
}

/*
int main() {
	int size = 10;
	int in[size];
	int out[size];
	for (int i = 1; i <= size; i++) {
		in[i - 1] = i;
	}
	printArray(in, size);
	psaParallelHS(out, in, size, false);
	printArray(out, size);
}
*/