#pragma once

#include <cuda_runtime.h>

#include <opencv2/core.hpp>

/**
 * @brief Converts an input image to greyscale using CUDA for parallel processing.
 *
 * @param h_out Reference to the output cv::Mat object where the greyscale image will be stored.
 * @param h_in Reference to the input cv::Mat object containing the original image.
 * @return cudaError_t CUDA error code indicating the success or failure of the operation.
 */
cudaError_t greyscaleParallel(cv::Mat &h_out, cv::Mat &h_in);
