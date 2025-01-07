#include <cuda_runtime.h>

/**
 * @brief Performs a parallel prefix sum (scan) using the Hillis-Steele algorithm.
 * 
 * This function computes the prefix sum array using the Hillis-Steele parallel scan algorithm.
 * The prefix sum array is computed either inclusively or exclusively based on the `inclusive` parameter.
 * 
 * @param out Pointer to the output array of integers that will contain the prefix sum.
 * @param in Pointer to the input array of integers.
 * @param size The size of the input array.
 * @param inclusive Boolean flag indicating whether the scan should be inclusive or exclusive. Default is false (exclusive).
 * @return cudaError_t CUDA error code indicating the success or failure of the operation.
 */
cudaError_t psaParallelHS(int* out, int* in, size_t size, bool inclusive = false);

/**
 * @brief Performs a parallel prefix sum (scan) using the Blelloch algorithm.
 * 
 * This function computes the prefix sum array using the Blelloch parallel scan algorithm.
 * The prefix sum array is computed either inclusively or exclusively based on the `inclusive` parameter.
 * 
 * @param out Pointer to the output array of integers that will contain the prefix sum.
 * @param in Pointer to the input array of integers.
 * @param size The size of the input array.
 * @param inclusive Boolean flag indicating whether the scan should be inclusive or exclusive. Default is false (exclusive).
 * @return cudaError_t CUDA error code indicating the success or failure of the operation.
 */
cudaError_t psaParallelB(int* out, int* in, size_t size, bool inclusive = false);