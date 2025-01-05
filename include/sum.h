#pragma once

#include <cuda_runtime.h>

/**
 * @brief Computes the sum of an array in parallel using CUDA reduction.
 *
 * This function performs a parallel reduction to calculate the sum of the elements
 * in the input array. The result is stored in the output parameter.
 *
 * @param[out] out Reference to an integer where the result will be stored.
 * @param[in] in Pointer to the input array of integers.
 * @param[in] size The number of elements in the input array.
 * @return cudaError_t CUDA error code indicating the success or failure of the operation.
 */
cudaError_t sumParallel(int &out, int* in, size_t size);