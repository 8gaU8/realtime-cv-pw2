#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // for high_resolution_clock
#include <omp.h>

using namespace std;

void applyDynamicGaussianFilter(
    const cv::Mat_<cv::Vec3f> &source,
    cv::Mat_<cv::Vec3b> &destination,
    const cv::Mat_<float> &detMatrixDivGFR,
    const float gaussianFactorRatio,
    const float sigma)
{
    float twoSigmaSquare = 2.0 * sigma * sigma;
    float twoPiSigmaSquare = M_PI * twoSigmaSquare;

#pragma omp parallel for
    for (int imRow = 0; imRow < source.rows; imRow++)
    {
        for (int imCol = 0; imCol < source.cols; imCol++)
        {
            // calculate kernel size
            float det = detMatrixDivGFR(imRow, imCol);
            det = max(det, 0.10f); // avoid division by zero
            int kernelSizeDiv2 = static_cast<int>(gaussianFactorRatio / det);

            // kernelSizeDiv2 = min(kernelSizeDiv2, 1); // limit kernel size to 10
            // kernelSizeDiv2 = max(min(kernelSizeDiv2, 1), 20); // limit kernel size to 10

            // dual loop for kernel
            cv::Vec3f pixelSum = cv::Vec3f(0.0, 0.0, 0.0);
            for (int kernelRow = -kernelSizeDiv2; kernelRow <= kernelSizeDiv2; kernelRow++)
            {
                int newRow = imRow + kernelRow;
                // consider the border
                newRow = min(max(newRow, 0), source.rows - 1);
                for (int kernelCol = -kernelSizeDiv2; kernelCol <= kernelSizeDiv2; kernelCol++)
                {
                    int newCol = imCol + kernelCol;
                    // consider the border
                    newCol = min(max(newCol, 0), source.cols - 1);
                    float weight = exp(-(float(kernelRow * kernelRow + kernelCol * kernelCol) / twoSigmaSquare)) / twoPiSigmaSquare;

                    pixelSum += source(newRow, newCol) * weight;
                }
            }
            destination(imRow, imCol) = pixelSum;
        }
    }
}

void calcDetCovarianceMatrix(
    const cv::Mat_<cv::Vec3f> &paddedSource,
    cv::Mat_<float> &detMatrix,
    const int neighborSizeDiv2)
{
    int squareNeighborSize = ((2 * neighborSizeDiv2 + 1) * (2 * neighborSizeDiv2 + 1));

#pragma omp parallel for
    for (int imRow = neighborSizeDiv2; imRow < paddedSource.rows - neighborSizeDiv2; imRow++)
    {
        for (int imCol = neighborSizeDiv2; imCol < paddedSource.cols - neighborSizeDiv2; imCol++)
        {
            // ーーーーーーーーーーーーーーーーーー
            // 1. calculate mean
            cv::Vec3f rgbMean = cv::Vec3f(0.0, 0.0, 0.0);
            for (int kernelRow = -neighborSizeDiv2; kernelRow <= neighborSizeDiv2; kernelRow++)
            {
                int newRow = imRow + kernelRow;
                for (int kernelCol = -neighborSizeDiv2; kernelCol <= neighborSizeDiv2; kernelCol++)
                {
                    int newCol = imCol + kernelCol;
                    rgbMean += paddedSource(newRow, newCol);
                }
            }

            rgbMean /= squareNeighborSize;

            // 2. calculate elements of Covariance Matrix
            cv::Vec3d diff;
            double Sbb = 0;
            double Sbg = 0;
            double Sbr = 0;
            double Sgg = 0;
            double Sgr = 0;
            double Srr = 0;
            for (int kernelRow = -neighborSizeDiv2; kernelRow <= neighborSizeDiv2; kernelRow++)
            {
                int newRow = imRow + kernelRow;
                for (int kernelCol = -neighborSizeDiv2; kernelCol <= neighborSizeDiv2; kernelCol++)
                {
                    int newCol = imCol + kernelCol;
                    // pixel-wise diff between pixel-value(RGB) and mean
                    diff = paddedSource(newRow, newCol) - rgbMean;
                    Sbb += diff(0) * diff(0);
                    Sbg += diff(0) * diff(1);
                    Sbr += diff(0) * diff(2);
                    Sgg += diff(1) * diff(1);
                    Sgr += diff(1) * diff(2);
                    Srr += diff(2) * diff(2);
                }
            }

            // 3. calculate determinant
            float det = (Sbb * Sgg * Srr) + 2 * (Sbg * Sbr * Sgr) - (Sbb * Sgr * Sgr + Sgg * Sbr * Sbr + Srr * Sbg * Sbg);
            det /= squareNeighborSize * squareNeighborSize * squareNeighborSize;

            // 5. store determinant
            detMatrix(imRow - neighborSizeDiv2, imCol - neighborSizeDiv2) = det;
        }
    }

    // min-max scaling
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(detMatrix, &minVal, &maxVal, &minLoc, &maxLoc);
    cout << "minVal: " << minVal << endl;
    cout << "maxVal: " << maxVal << endl;

    detMatrix = (detMatrix - minVal) / (maxVal - minVal);
    // calculate mean value
    // cv::Scalar mean = cv::mean(detMatrix);
    // cout << "mean: " << mean[0] << endl;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cout << "Usage: " << argv[0] << " <image> <neighborSizeDiv2> <gaussianFactorRatio> <[nbThreads]>" << endl;
        return -1;
    }

    char *filename = argv[1];
    int neighborSizeDiv2 = atoi(argv[2]);
    float gaussianFactorRatio = atof(argv[3]);
    float sigma = atof(argv[4]);

    int nbThreads = -1;
    if (argc == 6)
        nbThreads = atoi(argv[5]);

    if (nbThreads != -1)
        omp_set_num_threads(nbThreads);

    cout << "ARGS:" << endl;
    cout << "ARGC:" << argc << endl;
    cout << "filename: " << filename << endl;
    cout << "neighborSizeDiv2: " << neighborSizeDiv2 << endl;
    cout << "gaussianFactorRatio: " << gaussianFactorRatio << endl;
    cout << "sigma: " << sigma << endl;
    cout << "nbThreads: " << nbThreads << endl;
    cout << "----------------------------------------" << endl;

    const cv::Mat_<cv::Vec3f> source = cv::imread(filename, cv::IMREAD_COLOR);

    cv::Mat_<float> detMatrix(source.rows, source.cols);

    // make padded image
    // use `same` padding
    cv::Mat_<cv::Vec3f> paddedSource = cv::Mat_<cv::Vec3f>(source.rows + 2 * neighborSizeDiv2, source.cols + 2 * neighborSizeDiv2);
    cv::copyMakeBorder(source, paddedSource, neighborSizeDiv2, neighborSizeDiv2, neighborSizeDiv2, neighborSizeDiv2, cv::BORDER_REPLICATE);

    cv::Mat_<cv::Vec3b> destination = cv::Mat_<cv::Vec3b>(source.rows, source.cols);

    int kernelSizeDiv2;

    auto begin = chrono::high_resolution_clock::now();

    // Iteration for benchmarking
    const int iter = 1;
    for (int it = 0; it < iter; it++)
    {

        // calculate determinant of covariance matrix
        calcDetCovarianceMatrix(paddedSource, detMatrix, neighborSizeDiv2);

        // calculate gaussian
        applyDynamicGaussianFilter(source, destination, detMatrix, gaussianFactorRatio, sigma);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::Mat_<cv::Vec3b> destinationUchar;
    destination.convertTo(destinationUchar, CV_8UC3);
    cv::imwrite("results/image_original.png", source);
    cv::imwrite("results/image_det.png", detMatrix * 255);
    cv::imwrite("results/image_denoised.png", destinationUchar);

    return 0;
}