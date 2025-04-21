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

#include "anaglyphMethods.cuh"

inline int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, const AnaglyphFuncion &selectedAnaglyph)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    selectedAnaglyph<<<grid, block>>>(src, dst, dst.rows, dst.cols);
}

// ------------------------------

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <image> <anaglyphType>" << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    const char *filename = argv[1];
    const char *anaglyphType = argv[2];
    const AnaglyphFuncion selectedAnaglyph = selectAnaglyphFunction(anaglyphType);

    if (selectedAnaglyph == nullptr)
    {
        cout << "Invalid anaglyph type: " << anaglyphType << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

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
        processCUDA(d_src, d_dst, selectedAnaglyph);
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