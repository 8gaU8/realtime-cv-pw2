#ifndef ANAGLYPHMETHODS_HPP
#define ANAGLYPHMETHODS_HPP

#include <opencv2/opencv.hpp>

using AnaglyphFunctionType = void (*)(const cv::Vec3b, const cv::Vec3b, cv::Vec3b &);

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

// select the anaglyph function based on the type
AnaglyphFunctionType selectAnaglyphFunction(const char *anaglyphType)
{
    if (strcmp(anaglyphType, "true") == 0)
        return &trueAnaglyph;
    else if (strcmp(anaglyphType, "gray") == 0)
        return &grayAnaglyph;
    else if (strcmp(anaglyphType, "color") == 0)
        return &colorAnaglyph;
    else if (strcmp(anaglyphType, "halfColor") == 0)
        return &halfColorAnaglyph;
    else if (strcmp(anaglyphType, "optimized") == 0)
        return &optimizedAnaglyph;
    else
        return nullptr;
}

// process input image to anaglyph image
void processImageToAnaglyph(
    const cv::Mat_<cv::Vec3b> &source,
    cv::Mat_<cv::Vec3b> &destination,
    const AnaglyphFunctionType anaglyphFunction)
{
    // Process the image
#pragma omp parallel for
    for (int row = 0; row < source.rows; row++)
    {
        cv::Vec3b result;
        for (int col = 0; col < source.cols / 2; col++)
        {

            int leftCol = col;
            int rightCol = col + source.cols / 2;
            anaglyphFunction(source(row, leftCol), source(row, rightCol), result);
            destination(row, col) = result;
        }
    }
}

#endif // ANAGLYPHMETHODS_HPP