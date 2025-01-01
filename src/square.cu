#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


cudaError_t squareParallel(int* h_out, const int* h_in, size_t size); 

__global__ void squareKernel(int* d_out, int* d_in) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = d_in[idx];
	d_out[idx] = i * i;
}

cudaError_t squareParallel(int* h_out, const int* h_in, size_t size) {
	cudaError_t cudaStatus;

	int* d_out;
	int* d_in;

	cudaStatus = cudaMalloc((void**)&d_out, size * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_in, size * sizeof(int));

	cudaStatus = cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

	squareKernel<<<100, 450>>>(d_out, d_in);

	cudaStatus = cudaGetLastError();

	cudaStatus = cudaMemcpy(h_out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);

	return cudaStatus;
}

/*
int main() {
	const size_t SIZE = 45000;
	int h_in[SIZE];
	for (size_t i = 0; i < SIZE; i++) {
		h_in[i] = i + 1;
	}
	int h_out[SIZE];

	squareParallel(h_out, h_in, SIZE);

	for (size_t i = 0; i < SIZE; i++) {
		printf("%d", h_out[i]);
		printf((i % 8) == 7 ? "\n" : "\t");
	}
	return 0;

}
*/