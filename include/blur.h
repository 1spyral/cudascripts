#include <opencv2/core.hpp>

/**
 * @brief Applies an unweighted blur to an image using CUDA.
 * 
 * This function applies an unweighted blur to the input image using CUDA parallel processing.
 * The blur is applied using a square kernel of the specified radius.
 * 
 * @param out Reference to the cv::Mat object that will contain the blurred image.
 * @param in Reference to the cv::Mat object containing the input image.
 * @param radius The radius of the blur kernel. Default is 5.
 * @return cudaError_t CUDA error code indicating the success or failure of the operation.
 */
cudaError_t unweightedBlurParallel(cv::Mat &out, cv::Mat &in, int radius = 5);

/**
 * @brief Applies a Gaussian blur to an image using CUDA.
 * 
 * This function applies a Gaussian blur to the input image using CUDA parallel processing.
 * The blur is applied using a Gaussian kernel of the specified radius.
 * 
 * @param out Reference to the cv::Mat object that will contain the blurred image.
 * @param in Reference to the cv::Mat object containing the input image.
 * @param radius The radius of the Gaussian kernel. Default is 5.
 * @return cudaError_t CUDA error code indicating the success or failure of the operation.
 */
cudaError_t gaussianBlurParallel(cv::Mat &out, cv::Mat &in, int radius = 5);
