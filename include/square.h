#pragma once

#include <cuda_runtime.h>

/**
 * @brief Squares each integer in an array using CUDA parallel processing.
 * 
 * This function takes an array of `size` elements `h_in` and writes their squared values into an array of `size` elements `h_out`. The function utilizes CUDA for parallel processing to achieve efficient computation.
 * 
 * @param h_out The array to store the squared values.
 * @param h_in The input array containing the original values.
 * @param size The number of elements in the input and output arrays.
 * @return cudaError_t Returns a CUDA error code indicating success or failure.
 */
cudaError_t squareParallel(int* h_out, const int* h_in, size_t size);