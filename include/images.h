#pragma once

#include <opencv2/core.hpp>

/**
 * @brief Loads an image from the specified file path.
 * 
 * @param path The path to the image file.
 * @return cv::Mat The loaded image.
 */
cv::Mat getImage(std::string path);

/**
 * @brief Writes the given image to the specified file path.
 * 
 * @param path The file path where the image will be saved.
 * @param img The image to be written to the file.
 * @return true if the image is successfully written, false otherwise.
 */
bool writeImage(std::string path, cv::Mat &img);

/**
 * @brief Displays a preview of the given image with specified dimensions.
 * 
 * This function takes an OpenCV matrix (cv::Mat) representing an image and 
 * displays a preview of it with the specified width and height.
 * 
 * @param img Reference to the cv::Mat object containing the image to be previewed.
 * @param width The desired width of the preview window.
 * @param height The desired height of the preview window.
 */
void previewImage(cv::Mat &img, int width, int height);

/**
 * @brief Displays a preview of the given image.
 * 
 * This function takes an OpenCV matrix (cv::Mat) representing an image and 
 * displays a preview of it with default dimensions.
 * 
 * @param img Reference to the cv::Mat object containing the image to be previewed.
 */
void previewImage(cv::Mat &img);

/**
 * @brief Calculates the size of the given image in pixels.
 * 
 * This function takes an OpenCV matrix (cv::Mat) representing an image and 
 * returns the total number of pixels in the image.
 * 
 * @param img Reference to the cv::Mat object containing the image.
 * @return size_t The total number of pixels in the image.
 */
extern inline size_t sizeInPixels(cv::Mat &img) {
    return img.rows * img.cols;
}

/**
 * @brief Calculates the size of the given image in pixels.
 * 
 * This function takes an OpenCV matrix (cv::Mat) representing an image and 
 * returns the total number of pixels in the image.
 * 
 * @param img Reference to the cv::Mat object containing the image.
 * @return size_t The total number of pixels in the image.
 */
size_t sizeInPixels(cv::Mat &img);

/**
 * @brief Calculates the size of the given image in pixels.
 * 
 * This function takes a pointer to an OpenCV matrix (cv::Mat) representing an image and 
 * returns the total number of pixels in the image.
 * 
 * @param img Pointer to the cv::Mat object containing the image.
 * @return size_t The total number of pixels in the image.
 */
extern inline size_t sizeInPixels(cv::Mat* img) {
    return img->rows * img->cols;
}

/**
 * @brief Calculates the size of the given image in pixels.
 * 
 * This function takes a pointer to an OpenCV matrix (cv::Mat) representing an image and 
 * returns the total number of pixels in the image.
 * 
 * @param img Pointer to the cv::Mat object containing the image.
 * @return size_t The total number of pixels in the image.
 */
size_t sizeInPixels(cv::Mat* img);
