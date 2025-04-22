#include <iostream>
#include <chrono> // for high_resolution_clock
#include <cfloat>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

using namespace std;

// Anaglyph declarations
// =========================================================
using AnaglyphFunction = void (*)(const cv::cuda::PtrStep<uchar3>, cv::cuda::PtrStep<uchar3>, int, int);

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

        // z,y,x = r,g,b
        dst(dst_y, dst_x).z = 0.299 * left.z + 0.587 * left.y + 0.114 * left.x;
        dst(dst_y, dst_x).y = 0.0;
        dst(dst_y, dst_x).x = 0.299 * right.z + 0.587 * right.y + 0.114 * right.x;
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

        // z,y,x = r,g,b
        uchar gray_left = 0.299 * left.z + 0.587 * left.y + 0.114 * left.x;
        uchar gray_right = 0.299 * right.z + 0.587 * right.y + 0.114 * right.x;

        dst(dst_y, dst_x).z = gray_left;
        dst(dst_y, dst_x).y = gray_right;
        dst(dst_y, dst_x).x = gray_right;
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

        // z,y,x = r,g,b
        dst(dst_y, dst_x).z = left.z;
        dst(dst_y, dst_x).y = right.y;
        dst(dst_y, dst_x).x = right.x;
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

        // z,y,x = r,g,b
        dst(dst_y, dst_x).z = 0.299 * left.z + 0.587 * left.y + 0.114 * left.x;
        dst(dst_y, dst_x).y = right.y;
        dst(dst_y, dst_x).x = right.x;
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

        // z,y,x = r,g,b
        dst(dst_y, dst_x).z = 0.7 * left.y + 0.3 * left.x;
        dst(dst_y, dst_x).y = right.y;
        dst(dst_y, dst_x).x = right.x;
    }
}
// anaglyph functions end

// parse anaglyph type
AnaglyphFunction selectAnaglyphFunction(const char *anaglyphType)
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

// =========================================================

inline int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processAnaglyphCUDA(
    cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &dst,
    const AnaglyphFunction &selectedAnaglyph,
    const int blockDimX,
    const int blockDimY)
{
    const dim3 block(blockDimX, blockDimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    selectedAnaglyph<<<grid, block>>>(src, dst, dst.rows, dst.cols);
}

// ------------------------------

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <image> <anaglyphType> <[blockdimx]> <[blockdimy]>" << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    const char *filename = argv[1];
    const char *anaglyphType = argv[2];
    const AnaglyphFunction selectedAnaglyph = selectAnaglyphFunction(anaglyphType);
    int blockDimX = 32;
    int blockDimY = 8;

    if (selectedAnaglyph == nullptr)
    {
        cout << "Invalid anaglyph type: " << anaglyphType << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    if (argc > 4)
    {
        blockDimX = atoi(argv[3]);
        blockDimY = atoi(argv[4]);
    }

    cout << "   Filename: " << filename << endl;
    cout << "   Anaglyph: " << anaglyphType << endl;
    cout << "  Block dim: " << blockDimX << " x " << blockDimY << endl;


    cv::Mat h_src = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat h_dst;

    h_dst.create(h_src.rows, h_src.cols / 2, CV_8UC3);

    cv::cuda::GpuMat d_src, d_dst;

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100;

    for (int i = 0; i < iter; i++)
    {
        d_src.upload(h_src);
        d_dst.upload(h_dst);
        processAnaglyphCUDA(d_src, d_dst, selectedAnaglyph, blockDimX, blockDimY);
        d_dst.download(h_dst);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Time: " << diff.count() << endl;
    cout << "Time/frame: " << diff.count() / iter << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::imwrite("./results/original.png", h_src);
    cv::imwrite("./results/anaglyph.png", h_dst);

    return 0;
}