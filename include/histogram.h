#include <cuda_runtime.h>

/**
 * @brief Computes a histogram using atomic operations in parallel on the GPU.
 *
 * This function calculates the histogram of the input data array using atomic operations
 * to ensure thread safety when updating the histogram bins. It leverages the parallel
 * processing capabilities of CUDA to speed up the computation.
 *
 * @param[out] out Pointer to the output array where the histogram will be stored. The array
 *                 should have a size equal to the number of bins (bin_count).
 * @param[in] in Pointer to the input data array containing the values to be histogrammed.
 * @param[in] min The minimum value in the input data range.
 * @param[in] max The maximum value in the input data range.
 * @param[in] bin_count The number of bins to be used for the histogram.
 * @param[in] size The number of elements in the input data array.
 * @return cudaError_t Returns CUDA error code indicating the success or failure of the kernel execution.
 */
cudaError_t histogramAtomicParallel(int* out, int* in, int min, int max, size_t bin_count, size_t size);