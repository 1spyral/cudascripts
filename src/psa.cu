#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_PER_BLOCK 1024

__global__ void psaHSKernel(int* d_out, int* d_in, int* d_offset, size_t size, bool inclusive) {
    __shared__ int psa[MAX_PER_BLOCK];

    size_t idx = threadIdx.x;
    size_t global_idx = blockIdx.x * MAX_PER_BLOCK + idx;

    size_t block_size = MAX_PER_BLOCK;
    if (blockIdx.x == (size + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK - 1) {
        block_size = size % MAX_PER_BLOCK;
        if (block_size == 0) {
            block_size = MAX_PER_BLOCK;
        }
    }

    size_t steps = ceil(log2f(block_size));
    size_t i = 1;
    size_t step = 1;

    if (steps == 0) {
        if (inclusive) {
            int val = d_in[global_idx];
            d_out[global_idx] = val;
            if (blockIdx.x != (size + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK - 1) {
                d_offset[global_idx] = val;
            }
        } else {
            d_out[global_idx] = 0;
        }
        return;
    }
    else if (steps == 1) {
        int output = 0;
        if (idx < 2) {
            if (inclusive) {
                output += d_in[blockIdx.x * MAX_PER_BLOCK];
                if (idx == 1) {
                    output += d_in[global_idx];
                    if (blockIdx.x != (size + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK - 1) {
                        d_offset[blockIdx.x] = output;
                    }
                }
            } else {
                if (idx == 1) {
                    output += d_in[blockIdx.x * MAX_PER_BLOCK];
                    if (blockIdx.x != (size + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK - 1) {
                        d_offset[blockIdx.x] = output + d_in[global_idx];
                    }
                }
            }
        }
        __syncthreads();
        if (idx < 2) {
            d_out[global_idx] = output;
        }
        return;
    }

    if (idx < block_size) {
        if (idx == 0) {
            psa[0] = d_in[global_idx];
        } else {
            psa[idx] = d_in[global_idx - i] + d_in[global_idx];
        }
    }
    __syncthreads();

    for (i = 2; i < steps; i++) {
        int addend;
        if (idx < block_size) {
            step <<= 1;
            addend = idx < step ? 0 : psa[idx - step];
        }
        __syncthreads();
        if (idx < block_size) {
            psa[idx] += addend;
        }
        __syncthreads();
    }

    if (idx >= block_size) {
        return;
    }

    int output;

    step <<= 1;
    if (inclusive) {
        output = psa[idx] + (idx < step ? 0 : psa[idx - step]);
    } else {
        if (idx == 0) {
            output = 0;
        } else if (idx - 1 < step) {
            output = psa[idx - 1];
        } else {
            output = psa[idx - 1] + psa[idx - step - 1];
        }
    }

    d_out[global_idx] = output;
    if (idx == block_size - 1 && blockIdx.x != (size + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK - 1) {
        if (inclusive) {
            d_offset[blockIdx.x] = output;
        } else {
            d_offset[blockIdx.x] = output + psa[idx];
        }
    }
}

__global__ void psaBKernel(int* d_out, int* d_in, int* d_offset, size_t size, bool inclusive) {
    extern __shared__ int psa[];

}

__global__ void addKernel(int* d_arr, int* d_in, int* d_offset, bool inclusive) {
    __shared__ int addend;

    int idx = threadIdx.x;
    int global_idx = (blockIdx.x + 1) * MAX_PER_BLOCK + idx;

    if (idx == 0) {
        addend = d_offset[blockIdx.x];
    }
    __syncthreads();

    d_arr[global_idx] += addend;
}

cudaError_t psaParallelHS(int* out, int* in, size_t size, bool inclusive = false, bool recursive = false) {
    cudaError_t cudaStatus;

    int blocks = (size + MAX_PER_BLOCK - 1) / MAX_PER_BLOCK;

    if (blocks == 0) {
        out = 0;
        goto Exit;
    }

    int* d_out;
    int* d_in;
    int* d_offset;

    if (recursive) {
        d_out = out;
        d_in = in;
    } else {
        cudaStatus = cudaMalloc((void**) &d_out, size * sizeof(int));
        cudaStatus = cudaMalloc((void**) &d_in, size * sizeof(int));
        
        cudaStatus = cudaMemcpy(d_in, in, size * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaStatus = cudaMalloc((void**) &d_offset, (blocks - 1) * sizeof(int));
    
    psaHSKernel<<<blocks, MAX_PER_BLOCK>>>(d_out, d_in, d_offset, size, inclusive);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "psaParallelHSKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Exit;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Exit;
    }

    if (blocks > 2) {
        cudaStatus = psaParallelHS(d_offset, d_offset, blocks - 1, true, true);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Recursive psaParallelHS failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Exit;
        }
    }

    if (blocks > 1) {
        addKernel<<<blocks - 1, MAX_PER_BLOCK>>>(d_out, d_in, d_offset, inclusive);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Exit;
        }
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Exit;
    }

    if (!recursive) {
        cudaStatus = cudaMemcpy(out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);
    }

Exit:
    if (!recursive) {
        cudaFree(d_out);
        cudaFree(d_in);
    }
    cudaFree(d_offset);
    return cudaStatus;
}

cudaError_t psaParallelB(int* out, int* in, size_t size, bool inclusive = false) {
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