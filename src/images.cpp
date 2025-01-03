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

size_t flattenColor(uint8_t* &img_out, cv::Mat &img_in) {
    size_t size = sizeInPixels(img_in) * 3;
    img_out = new uint8_t[size];
    for (int row = 0; row < img_in.rows; row++) {
        for (int col = 0; col < img_in.cols; col++) {
            int idx = col + row * img_in.cols;
            cv::Vec3b pixel = img_in.at<cv::Vec3b>(row, col);
            img_out[3 * idx] = pixel[0];
            img_out[3 * idx + 1] = pixel[1];
            img_out[3 * idx + 2] = pixel[2];
        }
    }
    return size;
}
