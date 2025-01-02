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
bool writeImage(std::string path, cv::Mat &img)

/**
 * @brief Displays the given image in a window.
 * 
 * @param img The image to display.
 */
void previewImage(cv::Mat &img);
