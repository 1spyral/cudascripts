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

void previewImage(cv::Mat &img) {
   cv::imshow("Display window", img);
   cv::waitKey();
}
