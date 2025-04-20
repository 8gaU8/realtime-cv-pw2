#ifndef ANAGLYPHMETHODS_HPP
#define ANAGLYPHMETHODS_HPP

#include <opencv2/opencv.hpp>


void trueAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result);
void grayAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result);
void colorAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result);
void halfColorAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result);
void optimizedAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result);

void processImageToAnaglyph(
    const cv::Mat_<cv::Vec3b> &source,
    cv::Mat_<cv::Vec3b> &destination,
    const AnaglyphFunctionType anaglyphFunction);

AnaglyphFunctionType selectAnaglyphFunction(const char *anaglyphType);

#endif // ANAGLYPHMETHODS_HPP