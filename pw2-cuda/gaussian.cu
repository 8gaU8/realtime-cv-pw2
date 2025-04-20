// === standard libraries ===
#include <iostream>
#include <chrono> // for high_resolution_clock
// #include <cfloat>

// === OpenCV with CUDA ===
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

using namespace std;

void makeGaussianKernel(int kernelSizeDiv2, float sigma, cv::Mat_<float> &kernelMat)
{
    for (int i = 0; i < 2 * kernelSizeDiv2 + 1; i++)
    {
        for (int j = 0; j < 2 * kernelSizeDiv2 + 1; j++)
        {
            float x = i - kernelSizeDiv2;
            float y = j - kernelSizeDiv2;
            kernelMat(i, j) = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
        }
    }
}

__global__ void applyGaussianFilter(
    const cv::cuda::PtrStep<uchar3> src,
    cv::cuda::PtrStep<uchar3> dst,
    int rows,
    int cols,
    int kernelSizeDiv2,
    const cv::cuda::PtrStep<float> kernelMat)

{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int srcCol = cols+ kernelSizeDiv2 * 2;
    const int srcRow = rows+ kernelSizeDiv2 * 2;

    if (dst_x < cols && dst_y < rows)
    {
        float3 sum = make_float3(0.0f);
        for (int y = -kernelSizeDiv2; y <= kernelSizeDiv2; y++)
        {
            for (int x = -kernelSizeDiv2; x <= kernelSizeDiv2; x++)
            {
                int src_x = x + dst_x + kernelSizeDiv2;
                int src_y = y + dst_y + kernelSizeDiv2;

                if (src_x >= 0 && src_x < srcCol && src_y >= 0 && src_y < srcRow)
                {
                    uchar3 pixel = src(src_y, src_x);
                    float weight = kernelMat(y, x);

                    sum.x += pixel.x * weight;
                    sum.y += pixel.y * weight;
                    sum.z += pixel.z * weight;
                }
            }
        }
        dst(dst_y, dst_x).x = min(max(int(sum.x), 0), 255);
        dst(dst_y, dst_x).y = min(max(int(sum.y), 0), 255);
        dst(dst_y, dst_x).z = min(max(int(sum.z), 0), 255);
    }
}

inline int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCUDA(
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &dst,
    const int kernelSizeDiv2,
    const cv::cuda::GpuMat &kernelMat)
{

    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    applyGaussianFilter<<<grid, block>>>(
        cv::cuda::PtrStep<uchar3>(src),
        cv::cuda::PtrStep<uchar3>(dst),
        dst.rows,
        dst.cols,
        kernelSizeDiv2,
        cv::cuda::PtrStep<float>(kernelMat) // Pass the kernel matrix to the kernel
    );
}

// ------------------------------

int main(int argc, char **argv)
{

    if (argc < 4)
    {
        cout << "Usage: " << argv[0] << " <image> <kernelSizeDiv2> <sigma>" << endl;
        return -1;
    }

    // parse arguments
    const char *filename = argv[1];
    const int kernelSizeDiv2 = atoi(argv[2]);
    const float sigma = atof(argv[3]);

    cout << "   Filename: " << filename << endl;
    cout << "Kernel size: " << kernelSizeDiv2 << endl;
    cout << "      Sigma: " << sigma << endl;

    const cv::Mat h_src = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat h_dst;
    h_dst.create(h_src.rows, h_src.cols, CV_8UC3);

    // make padded image
    cv::Mat h_paddedSource;
    h_paddedSource.create(h_src.rows + 2 * kernelSizeDiv2, h_src.cols + 2 * kernelSizeDiv2, CV_8UC3);
    // use `same` padding
    cv::copyMakeBorder(h_src, h_paddedSource, kernelSizeDiv2, kernelSizeDiv2, kernelSizeDiv2, kernelSizeDiv2, cv::BORDER_REPLICATE);

    // gaussian kernel
    cv::Mat_<float> k_kernelMat(2 * kernelSizeDiv2 + 1, 2 * kernelSizeDiv2 + 1);
    makeGaussianKernel(kernelSizeDiv2, sigma, k_kernelMat);

    cout << "Kernel: " << endl;
    float sum = 0;
    for (int i = 0; i < k_kernelMat.rows; i++)
        for (int j = 0; j < k_kernelMat.cols; j++)
            sum += k_kernelMat(i, j);
    cout << "Sum: " << sum << endl;

    // upload kernel to GPU
    cv::cuda::GpuMat d_kernelMat;
    d_kernelMat.upload(k_kernelMat);

    cv::cuda::GpuMat d_src, d_dst;

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100;

    for (int i = 0; i < iter; i++)
    {
        // upload source and destination images
        d_src.upload(h_paddedSource);
        d_dst.upload(h_dst);
        // process
        processCUDA(d_src, d_dst, kernelSizeDiv2, d_kernelMat);
        // download destination image
        d_dst.download(h_dst);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Time: " << diff.count() << endl;
    cout << "Time/frame: " << diff.count() / iter << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::imwrite("./results/original.png", h_src);
    cv::imwrite("./results/gaussian.png", h_dst);

    return 0;
}
