#include <opencv2/opencv.hpp>

void trueAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result)
{
    // https://3dtv.at/Knowhow/AnaglyphComparison_en.aspx True Anaglyph Method
    // red channel
    result[2] = 0.299 * left[2] + 0.587 * left[1] + 0.114 * left[0];
    // green channel
    result[1] = 0.0;
    // blue channel
    result[0] = 0.299 * right[2] + 0.587 * right[1] + 0.114 * right[0];
}

void grayAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result)
{
    // red channel
    result[2] = 0.299 * left[2] + 0.587 * left[1] + 0.114 * left[0];
    // green channel
    result[1] = 0.299 * right[2] + 0.587 * right[1] + 0.114 * right[0];
    // blue channel
    result[0] = 0.299 * right[2] + 0.587 * right[1] + 0.114 * right[0];
}

void colorAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result)
{
    // red channel
    result[2] = left[2];
    // green channel
    result[1] = right[1];
    // blue channel
    result[0] = right[0];
}

void halfColorAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result)
{
    // red channel
    result[2] = 0.299 * left[2] + 0.587 * left[1] + 0.114 * left[0];
    // green channel
    result[1] = right[1];
    // blue channel
    result[0] = right[2];
}

void optimizedAnaglyph(const cv::Vec3b left, const cv::Vec3b right, cv::Vec3b &result)
{
    // red channel
    result[2] = 0.7 * left[1] + 0.3 * left[0];
    // green channel
    result[1] = right[1];
    // blue channel
    result[0] = right[0];
}