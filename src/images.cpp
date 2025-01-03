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

uint8_t* flattenColor(cv::Mat &p_img) {
    uint8_t* out = new uint8_t[sizeInPixels(p_img) * 3];
    for (int row = 0; row < p_img.rows; row++) {
        for (int col = 0; col < p_img.cols; col++) {
            int idx = col + row * p_img.cols;
            cv::Vec3b pixel = p_img.at<cv::Vec3b>(row, col);
            out[3 * idx] = pixel[0];
            out[3 * idx + 1] = pixel[1];
            out[3 * idx + 2] = pixel[2];
        }
    }
    return out;
}
