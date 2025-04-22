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
// #include "anaglyphMethods.cuh"

using namespace std;

#define BLOCK_DIM 32
#define BLOCK_ROWS 32
#define KERNEL_RADIUS 3

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
    __shared__ uchar3 sharedMem[BLOCK_DIM + 2 * KERNEL_RADIUS][BLOCK_DIM + 2 * KERNEL_RADIUS];
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    // ↓ Identical to dst_*
    const int global_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_y = threadIdx.y + blockIdx.y * blockDim.y;

    const int local_x = threadIdx.x + KERNEL_RADIUS;
    const int local_y = threadIdx.y + KERNEL_RADIUS;

    // Load data into shared memory
    sharedMem[local_y][local_x] = src(global_y, global_x);
    if (threadIdx.x < KERNEL_RADIUS)
    {
        // Load left halo
        sharedMem[local_y][local_x - KERNEL_RADIUS] = src(global_y, global_x - KERNEL_RADIUS);
        // Load right halo
        sharedMem[local_y][local_x + BLOCK_DIM] = src(global_y, global_x + BLOCK_DIM);
    }
    else if (threadIdx.y < KERNEL_RADIUS)
    {
        // Load top halo
        sharedMem[local_y - KERNEL_RADIUS][local_x] = src(global_y - KERNEL_RADIUS, global_x);
        // Load bottom halo
        sharedMem[local_y + BLOCK_DIM][local_x] = src(global_y + BLOCK_DIM, global_x);
    }
    else if (threadIdx.x < KERNEL_RADIUS && threadIdx.y < KERNEL_RADIUS)
    {
        // Load top left halo
        sharedMem[local_y - KERNEL_RADIUS][local_x - KERNEL_RADIUS] = src(global_y - KERNEL_RADIUS, global_x - KERNEL_RADIUS);
        // Load top right halo
        sharedMem[local_y - KERNEL_RADIUS][local_x + BLOCK_DIM] = src(global_y - KERNEL_RADIUS, global_x + BLOCK_DIM);
        // Load bottom left halo
        sharedMem[local_y + BLOCK_DIM][local_x - KERNEL_RADIUS] = src(global_y + BLOCK_DIM, global_x - KERNEL_RADIUS);
        // Load bottom right halo
        sharedMem[local_y + BLOCK_DIM][local_x + BLOCK_DIM] = src(global_y + BLOCK_DIM, global_x + BLOCK_DIM);
    }
    // for (int y = 0; y < BLOCK_DIM + 2 * KERNEL_RADIUS; y++)
    // {
    //     for (int x = 0; x < BLOCK_DIM + 2 * KERNEL_RADIUS; x++)
    //     {
    //         int global_y = blockIdx.y * BLOCK_DIM - KERNEL_RADIUS + y;
    //         int global_x = blockIdx.x * BLOCK_DIM - KERNEL_RADIUS + x;
    //         sharedMem[y][x] = src(global_y, global_x);
    //     }
    // }
    __syncthreads();

    int minCol = 0;
    int maxCol = cols / 2 - 1;
    if (dst_x >= cols / 2)
    {
        minCol = cols / 2;
        maxCol = cols - 1;
    }

    if (dst_y < rows && dst_x < cols)
    {
        float3 sum = make_float3(0.0f);

        for (int y = -kernelSizeDiv2; y <= kernelSizeDiv2; y++)
        {
            int src_y = clamp(y + dst_y, 0, rows - 1);
            for (int x = -kernelSizeDiv2; x <= kernelSizeDiv2; x++)
            {
                int src_x = clamp(x + dst_x, minCol, maxCol);

                // uchar3 pixel = src(src_y, src_x);
                uchar3 pixel = sharedMem[local_y + KERNEL_RADIUS + y][local_x + KERNEL_RADIUS + x];
                // uchar3 pixel = sharedMem[local_y + KERNEL_RADIUS + y][local_x + KERNEL_RADIUS + x];
                float weight = 1.0 / ((2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1));
                // kernelMat(y + kernelSizeDiv2, x + kernelSizeDiv2);

                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
                sum.z += pixel.z * weight;
            }
        }
        dst(dst_y, dst_x).x = sum.x;
        dst(dst_y, dst_x).y = sum.y;
        dst(dst_y, dst_x).z = sum.z;

        // uchar3 pixel =
        // dst(dst_y, dst_x).x = (sharedMem[local_y - KERNEL_RADIUS][local_x + KERNEL_RADIUS].x + sharedMem[local_y + KERNEL_RADIUS][local_x - KERNEL_RADIUS].x + sharedMem[local_y + KERNEL_RADIUS][local_x + KERNEL_RADIUS].x + sharedMem[local_y - KERNEL_RADIUS][local_x - KERNEL_RADIUS].x) / 4;
        // dst(dst_y, dst_x).y = (sharedMem[local_y - KERNEL_RADIUS][local_x + KERNEL_RADIUS].y + sharedMem[local_y + KERNEL_RADIUS][local_x - KERNEL_RADIUS].y + sharedMem[local_y + KERNEL_RADIUS][local_x + KERNEL_RADIUS].y + sharedMem[local_y - KERNEL_RADIUS][local_x - KERNEL_RADIUS].y) / 4;
        // dst(dst_y, dst_x).z = (sharedMem[local_y - KERNEL_RADIUS][local_x + KERNEL_RADIUS].z + sharedMem[local_y + KERNEL_RADIUS][local_x - KERNEL_RADIUS].z + sharedMem[local_y + KERNEL_RADIUS][local_x + KERNEL_RADIUS].z + sharedMem[local_y - KERNEL_RADIUS][local_x - KERNEL_RADIUS].z) / 4;

    }
}

inline int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processGaussianCUDA(
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &dst,
    const int kernelSizeDiv2,
    const cv::cuda::GpuMat &kernelMat)
{

    const dim3 block(BLOCK_DIM, BLOCK_ROWS);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
    // cout << "Grid: " << grid.x << "x" << grid.y << endl;
    // cout << "Block: " << block.x << "x" << block.y << endl;

    applyGaussianFilter<<<grid, block>>>(
        cv::cuda::PtrStep<uchar3>(src),
        cv::cuda::PtrStep<uchar3>(dst),
        dst.rows,
        dst.cols,
        kernelSizeDiv2,
        cv::cuda::PtrStep<float>(kernelMat));
}

// void processAnaglyphCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, const AnaglyphFunction &selectedAnaglyph)
// {
//     const dim3 block(32, 8);
//     const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

//     selectedAnaglyph<<<grid, block>>>(src, dst, dst.rows, dst.cols);
// }

// ------------------------------

int main(int argc, char **argv)
{

    if (argc < 5)
    {
        cout << "Usage: " << argv[0] << " <image> <anaglyphType> <kernelSizeDiv2> <sigma>" << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    // parse arguments
    const char *filename = argv[1];
    const char *anaglyphType = argv[2];
    const int kernelSizeDiv2 = KERNEL_RADIUS;
    const float sigma = atof(argv[4]);

    cout << "   Filename: " << filename << endl;
    cout << "   Anaglyph: " << anaglyphType << endl;
    cout << "Kernel size: " << kernelSizeDiv2 << endl;
    cout << "      Sigma: " << sigma << endl;

    // const AnaglyphFunction selectedAnaglyph = selectAnaglyphFunction(anaglyphType);

    // if (selectedAnaglyph == nullptr)
    // {
    //     cout << "Invalid anaglyph type: " << anaglyphType << endl;
    //     cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
    //     return -1;
    // }

    const cv::Mat h_src = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat h_dst;
    h_dst.create(h_src.rows, h_src.cols / 2, CV_8UC3);

    // gaussian kernel
    cv::Mat_<float> k_kernelMat(2 * kernelSizeDiv2 + 1, 2 * kernelSizeDiv2 + 1);
    makeGaussianKernel(kernelSizeDiv2, sigma, k_kernelMat);

    // upload kernel to GPU
    cv::cuda::GpuMat d_kernelMat;
    d_kernelMat.upload(k_kernelMat);

    cv::cuda::GpuMat d_src, d_mid, d_dst;

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100;

    for (int i = 0; i < iter; i++)
    {
        // upload source and destination images
        d_src.upload(h_src);
        d_mid.upload(h_src);
        d_dst.upload(h_dst);
        // process
        processGaussianCUDA(d_src, d_mid, kernelSizeDiv2, d_kernelMat);
        // processAnaglyphCUDA(d_mid, d_dst, selectedAnaglyph);
        // download destination image
        d_dst.download(h_dst);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Time: " << diff.count() << endl;
    cout << "Time/frame: " << diff.count() / iter << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::Mat h_mid;
    h_mid.create(h_src.rows, h_src.cols, CV_8UC3);
    d_mid.download(h_mid);
    cv::imwrite("./results/original.png", h_src);
    cv::imwrite("./results/gaussian.png", h_dst);
    cv::imwrite("./results/mid.png", h_mid);

    return 0;
}
