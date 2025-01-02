#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

cv::Mat getImage(std::string path) {
    return cv::imread(path, cv::IMREAD_COLOR);
}

bool writeImage(std::string path, cv::Mat &img) {
    return cv::imwrite(path, img);
}

void previewImage(cv::Mat &img, int width, int height) {
    cv::namedWindow("Display window", cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("Display window", width, height);
    cv::imshow("Display window", img);
    cv::waitKey();
}

void previewImage(cv::Mat &img) {
    previewImage(img, img.cols, img.rows);
}

size_t sizeInPixels(cv::Mat &img) {
    return img.rows * img.cols;
}

size_t sizeInPixels(cv::Mat *img) {
    return sizeInPixels(*img);
}
