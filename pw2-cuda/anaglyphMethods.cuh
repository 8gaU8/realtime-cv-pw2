#ifdef ANAGLYPHMETHODS_CUH
#define ANAGLYPHMETHODS_CUH

#include <iostream>
#include <chrono> // for high_resolution_clock
#include <cfloat>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

using AnaglyphFuncion = void (*)(const cv::cuda::PtrStep<uchar3>, cv::cuda::PtrStep<uchar3>, int, int);

// ======== ANAGLYPH FUNCTIONS ========
__global__ void trueAnaglyph(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int left_x = dst_x;
    const int right_x = dst_x + cols;

    if (dst_x < cols && dst_y < rows)
    {
        uchar3 left = src(dst_y, left_x);
        uchar3 right = src(dst_y, right_x);

        dst(dst_y, dst_x).x = 0.299 * left.x + 0.587 * left.y + 0.114 * left.z;
        dst(dst_y, dst_x).y = 0.0;
        dst(dst_y, dst_x).z = 0.299 * right.x + 0.587 * right.y + 0.114 * right.z;
    }
}

__global__ void grayAnaglyph(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int left_x = dst_x;
    const int right_x = dst_x + cols;

    if (dst_x < cols && dst_y < rows)
    {
        uchar3 left = src(dst_y, left_x);
        uchar3 right = src(dst_y, right_x);

        uchar gray_left = 0.299 * left.x + 0.587 * left.y + 0.114 * left.z;
        uchar gray_right = 0.299 * right.x + 0.587 * right.y + 0.114 * right.z;

        dst(dst_y, dst_x).x = gray_left;
        dst(dst_y, dst_x).y = gray_left;
        dst(dst_y, dst_x).z = gray_right;
    }
}

__global__ void colorAnaglyph(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int left_x = dst_x;
    const int right_x = dst_x + cols;

    if (dst_x < cols && dst_y < rows)
    {
        uchar3 left = src(dst_y, left_x);
        uchar3 right = src(dst_y, right_x);

        dst(dst_y, dst_x).x = left.x;
        dst(dst_y, dst_x).y = left.y;
        dst(dst_y, dst_x).z = right.z;
    }
}

__global__ void halfColorAnaglyph(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int left_x = dst_x;
    const int right_x = dst_x + cols;

    if (dst_x < cols && dst_y < rows)
    {
        uchar3 left = src(dst_y, left_x);
        uchar3 right = src(dst_y, right_x);

        dst(dst_y, dst_x).x = 0.299 * left.x + 0.587 * left.y + 0.114 * left.z;
        dst(dst_y, dst_x).y = left.y;
        dst(dst_y, dst_x).z = right.z;
    }
}

__global__ void optimizedAnaglyph(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int left_x = dst_x;
    const int right_x = dst_x + cols;

    if (dst_x < cols && dst_y < rows)
    {
        uchar3 left = src(dst_y, left_x);
        uchar3 right = src(dst_y, right_x);

        dst(dst_y, dst_x).x = 0.7 * left.x + 0.3 * left.y;
        dst(dst_y, dst_x).y = left.y;
        dst(dst_y, dst_x).z = right.z;
    }
}
// anaglyph functions end

// parse anaglyph type
AnaglyphFuncion selectAnaglyphFunction(const char *anaglyphType)
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

#endif // ANAGLYPHMETHODS_CUH