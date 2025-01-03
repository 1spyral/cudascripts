#include <cassert>
#include <chrono>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helpers.h"

#define BLOCK_SIZE 1000

int testValues(size_t num_threads, size_t num_elements, bool atomic);

__global__ void incrementKernel(size_t num_threads, size_t num_elements, int* d_arr) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    d_arr[idx * num_elements / num_threads] = d_arr[idx * num_elements / num_threads] + 1;
}

__global__ void incrementAtomicKernel(size_t num_threads, size_t num_elements, int* d_arr) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&d_arr[idx * num_elements / num_threads], 1);
}

void incrementTest(size_t* num_threads_arr, size_t* num_elements_arr, size_t count, int* times_regular, int* times_atomic) {    
    for (size_t i = 0; i < count; i++) {
        std::cout << "Test " << i + 1 << ":" << std::endl;
        std::cout << "# of threads: " << num_threads_arr[i] << "\t# of elements: " << num_elements_arr[i] << std::endl;
        // Test non-atomic increment
        std::cout << "Regular increment: ";
        times_regular[i] = testValues(num_threads_arr[i], num_elements_arr[i], false);
        std::cout << "Time: " << times_regular[i] << "µs" << std::endl;
        // Test atomic increment
        std::cout << "Atomic increment: ";
        times_atomic[i] = testValues(num_threads_arr[i], num_elements_arr[i], true);
        std::cout << "Time: " << times_atomic[i] << "µs" << std::endl;
        std::cout << std::endl;
    }
}

int testValues(size_t num_threads, size_t num_elements, bool atomic) {
    assert(num_elements <= num_threads);

    cudaError_t cudaStatus;

    int* d_arr;

    cudaStatus = cudaMalloc((void**)&d_arr, num_elements * sizeof(int));
    cudaStatus = cudaMemset(d_arr, 0, num_elements * sizeof(int));

    int grid_size = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_size = num_threads < BLOCK_SIZE ? num_threads: BLOCK_SIZE;

    auto start = std::chrono::high_resolution_clock::now();

    if (atomic) {
        incrementAtomicKernel<<<grid_size, block_size>>>(num_threads, num_elements, d_arr);
    } else {
        incrementKernel<<<grid_size, block_size>>>(num_threads, num_elements, d_arr);
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_arr);
        return -1;
    }

    auto stop = std::chrono::high_resolution_clock::now();

    cudaStatus = cudaGetLastError();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_arr);
        return -1;
    }

    int h_arr[num_elements];

    cudaStatus = cudaMemcpy(h_arr, d_arr, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    printArray(h_arr, num_elements);

    cudaStatus = cudaFree(d_arr);

    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();;
}

/*
int main() {
	size_t num_threads_arr[] {10, 10000000, 10};
	size_t num_elements_arr[] {10, 10, 1};
	size_t count = 3;
	int times_regular[count];
	int times_atomic[count];

	incrementTest(num_threads_arr, num_elements_arr, count, times_regular, times_atomic);

	std::cout << "Times: ";
	std::cout << "Regular: ";
	printArray(times_regular, count);
	std::cout << "Atomic: ";
	printArray(times_atomic, count);
}
*/