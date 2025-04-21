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
#include <opencv2/cudaarithm.hpp>

#include "helper_math.h"
#include "anaglyphMethods.cuh"

using namespace std;

__global__ void calcDetCovarianceMatrix(
    const cv::cuda::PtrStep<uchar3> src,
    cv::cuda::PtrStep<float> detMatrix,
    int rows,
    int cols,
    int neighborSizeDiv2)

{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int squareNeighborSize = ((2 * neighborSizeDiv2 + 1) * (2 * neighborSizeDiv2 + 1));

    if (dst_x < cols && dst_y < rows)
    {
        // -----------
        // 1. Calculate mean value
        float3 rgbMean = make_float3(0.0f);
        for (int y = -neighborSizeDiv2; y <= neighborSizeDiv2; y++)
        {
            int src_y = y + dst_y;
            for (int x = -neighborSizeDiv2; x <= neighborSizeDiv2; x++)
            {
                int src_x = x + dst_x;

                if (src_x >= 0 && src_x < cols && src_y >= 0 && src_y < rows)
                {
                    uchar3 pixel = (src(src_y, src_x));
                    rgbMean.x += float(pixel.x);
                    rgbMean.y += float(pixel.y);
                    rgbMean.z += float(pixel.z);
                }
            }
        }
        rgbMean.x /= squareNeighborSize;
        rgbMean.y /= squareNeighborSize;
        rgbMean.z /= squareNeighborSize;

        // 2. calculate elements of Covariance matrix
        float3 diff;
        float Sbb = 0;
        float Sbg = 0;
        float Sbr = 0;
        float Sgg = 0;
        float Sgr = 0;
        float Srr = 0;

        for (int y = -neighborSizeDiv2; y <= neighborSizeDiv2; y++)
        {
            int src_y = y + dst_y;
            for (int x = -neighborSizeDiv2; x <= neighborSizeDiv2; x++)
            {
                int src_x = x + dst_x;

                if (src_x >= 0 && src_x < cols && src_y >= 0 && src_y < rows)
                {
                    // pixel-wise diff between pixel-value(RGB) and mean
                    uchar3 pixel = (src(src_y, src_x));
                    diff.x = float(pixel.x - rgbMean.x);
                    diff.y = float(pixel.y - rgbMean.y);
                    diff.z = float(pixel.z - rgbMean.z);

                    Sbb += diff.x * diff.x;
                    Sbg += diff.x * diff.y;
                    Sbr += diff.x * diff.z;
                    Sgg += diff.y * diff.y;
                    Sgr += diff.y * diff.z;
                    Srr += diff.z * diff.z;
                }
            }
        }
        // 3. calculate determinant
        float det = (Sbb * Sgg * Srr) + 2 * (Sbg * Sbr * Sgr) - (Sbb * Sgr * Sgr + Sgg * Sbr * Sbr + Srr * Sbg * Sbg);
        // Theoritically, determinant should be divided by the number of pixels 
        // but it is not necessary since we have `gaussianFactorRatio` parameter
        // det /= (squareNeighborSize * squareNeighborSize * squareNeighborSize);

        // 4. store determinant in the matrix
        detMatrix(dst_y, dst_x) = det;
    }
}

__global__ void applyDynamicGaussianFilter(
    const cv::cuda::PtrStep<uchar3> src,
    cv::cuda::PtrStep<uchar3> dst,
    const int rows,
    const int cols,
    cv::cuda::PtrStep<float> detMatrix,
    const float gaussianFactorRatio,
    const float sigma)

{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int minCol = 0;
    int maxCol = cols / 2 - 1;
    if (dst_x >= cols / 2)
    {
        minCol = cols / 2;
        maxCol = cols - 1;
    }

    if (dst_x < cols && dst_y < rows)
    {

        // 1. pre calculate constants
        float twoSigmaSquare = 2.0 * sigma * sigma;
        float twoPiSigmaSquare = M_PI * twoSigmaSquare;

        // 2. calculate kernel size
        float det = detMatrix(dst_x, dst_y);
        det = min(max(det, 0.05f), 1.0f);
        int kernelSizeDiv2 = int(gaussianFactorRatio / det);

        // 3. convolution
        float3 sum = make_float3(0.0f);
        for (int y = -kernelSizeDiv2; y <= kernelSizeDiv2; y++)
        {
            // 4. check bounds (y)
            int src_y = clamp(y + dst_y, 0, rows - 1);
            for (int x = -kernelSizeDiv2; x <= kernelSizeDiv2; x++)
            {
                // 5. check bounds (x)
                int src_x = clamp(x + dst_x, minCol, maxCol);

                // 5. calculate weight
                float weight = exp(-(float(y * y + x * x) / twoSigmaSquare)) / twoPiSigmaSquare;

                // 6. do convolution
                uchar3 pixel = src(src_y, src_x);
                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
                sum.z += pixel.z * weight;
            }
        }

        // 7. normalize
        dst(dst_y, dst_x).x = min(max(int(sum.x), 0), 255);
        dst(dst_y, dst_x).y = min(max(int(sum.y), 0), 255);
        dst(dst_y, dst_x).z = min(max(int(sum.z), 0), 255);
    }
}

inline int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCovarianceCUDA(
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &detMatrix,
    const int neighborSizeDiv2)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(detMatrix.cols, block.x), divUp(detMatrix.rows, block.y));

    calcDetCovarianceMatrix<<<grid, block>>>(
        cv::cuda::PtrStep<uchar3>(src),
        cv::cuda::PtrStep<float>(detMatrix),
        detMatrix.rows,
        detMatrix.cols,
        neighborSizeDiv2);
    cudaDeviceSynchronize();
}

void processGaussianCUDA(
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &dst,
    cv::cuda::GpuMat &detMatrix,
    const float gaussianFactorRatio,
    const float sigma)
{

    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    applyDynamicGaussianFilter<<<grid, block>>>(
        cv::cuda::PtrStep<uchar3>(src),
        cv::cuda::PtrStep<uchar3>(dst),
        dst.rows,
        dst.cols,
        cv::cuda::PtrStep<float>(detMatrix), // Pass the kernel matrix to the kernel
        gaussianFactorRatio,
        sigma);

    cudaDeviceSynchronize();
}

void processAnaglyphCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, const AnaglyphFunction &selectedAnaglyph)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    selectedAnaglyph<<<grid, block>>>(src, dst, dst.rows, dst.cols);
}

// ------------------------------

int main(int argc, char **argv)
{

    if (argc < 6)
    {
        cout << "Usage: " << argv[0] << " <image> <anaglyphType> <neighborSizeDiv2> <gaussianFactorRatio> <sigma>" << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    // parse arguments
    const char *filename = argv[1];
    const char *anaglyphType = argv[2];
    const int neighborSizeDiv2 = atoi(argv[3]);
    const float gaussianFactorRatio = atof(argv[4]);
    const float sigma = atof(argv[5]);

    cout << "             Filename: " << filename << endl;
    cout << "             Anaglyph: " << anaglyphType << endl;
    cout << "        Neighbor size: " << neighborSizeDiv2 << endl;
    cout << "Gaussian factor ratio: " << gaussianFactorRatio << endl;
    cout << "                Sigma: " << sigma << endl;

    const AnaglyphFunction selectedAnaglyph = selectAnaglyphFunction(anaglyphType);

    if (selectedAnaglyph == nullptr)
    {
        cout << "Invalid anaglyph type: " << anaglyphType << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    const cv::Mat h_src = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat h_anaglyph;
    h_anaglyph.create(h_src.rows, h_src.cols / 2, CV_8UC3);

    cv::Mat h_detMatrix;
    h_detMatrix.create(h_src.rows, h_src.cols, CV_32F);

    cv::cuda::GpuMat d_detMatrix;

    cv::cuda::GpuMat d_src, d_gaussian, d_anaglyph;

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 1;

    for (int i = 0; i < iter; i++)
    {
        // upload source and destination images
        d_src.upload(h_src);
        d_detMatrix.upload(h_detMatrix);
        d_gaussian.upload(h_src);
        d_anaglyph.upload(h_anaglyph);

        // calc determinant and normalize
        processCovarianceCUDA(d_src, d_detMatrix, neighborSizeDiv2);
        cv::cuda::normalize(d_detMatrix, d_detMatrix, 0.0, 10.0, cv::NORM_MINMAX, -1);
        cout << "det done" << endl;

        // apply gaussian filter
        processGaussianCUDA(d_src, d_gaussian, d_detMatrix, gaussianFactorRatio, sigma);
        cout << "gaussian done" << endl;

        // apply anaglyph
        processAnaglyphCUDA(d_gaussian, d_anaglyph, selectedAnaglyph);
        cout << "anaglyph done" << endl;

        d_anaglyph.download(h_anaglyph);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Time: " << diff.count() << endl;
    cout << "Time/frame: " << diff.count() / iter << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::imwrite("./results/original.png", h_src);
    cv::imwrite("./results/denoised_anaglyph.png", h_anaglyph);

    cv::Mat h_gaussian;
    d_gaussian.download(h_gaussian);
    cv::imwrite("./results/gaussian.png", h_gaussian);

    return 0;
}
