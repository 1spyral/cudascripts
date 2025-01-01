#pragma once

#include <cuda_runtime.h>

/**
 * @brief Adds two arrays using CUDA.
 *
 * This function adds the elements of two arrays `a` and `b` of size `size` and stores the result in array `c`.
 * The addition is performed using CUDA to leverage GPU acceleration.
 *
 * @param c Pointer to the output array where the result will be stored.
 * @param a Pointer to the first input array.
 * @param b Pointer to the second input array.
 * @param size The number of elements in the input arrays.
 * @return cudaError_t Returns CUDA error code indicating the success or failure of the operation.
 */
cudaError_t addParallel(int *c, const int *a, const int *b, size_t size);
