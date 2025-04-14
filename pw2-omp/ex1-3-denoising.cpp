#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // for high_resolution_clock
#include <omp.h>

using namespace std;

void applyDynamicGaussianFilter(
    const cv::Mat_<cv::Vec3f> &source,
    cv::Mat_<cv::Vec3b> &destination,
    const cv::Mat_<float> &detMatrix,
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
            float det = detMatrix(imRow, imCol);
            det = max(det, 0.10f); // avoid division by zero
            int kernelSizeDiv2 = static_cast<int>(gaussianFactorRatio / det);
            // kernelSizeDiv2 = min(max(kernelSizeDiv2, 1), 5); // clamp to [1, 10]

            // dual loop for kernel
            cv::Vec3f pixelSum = cv::Vec3f(0.0, 0.0, 0.0);
            for (int kernelRow = -kernelSizeDiv2; kernelRow <= kernelSizeDiv2; kernelRow++)
            {
                int newRow = imRow + kernelRow;
                newRow = min(max(newRow, 0), source.rows - 1);
                for (int kernelCol = -kernelSizeDiv2; kernelCol <= kernelSizeDiv2; kernelCol++)
                {
                    int newCol = imCol + kernelCol;
                    newCol = min(max(newCol, 0), source.cols - 1);
                    float weight = exp(-(float(kernelRow*kernelRow + kernelCol*kernelCol) / twoSigmaSquare)) / twoPiSigmaSquare;

                    pixelSum += source(newRow, newCol) * weight;
                }
            }
            destination(imRow, imCol) = pixelSum;
            // cout << "pixelSum: " << pixelSum << endl;
        }
    }
}

void calcDetCovarianceMatrix(
    const cv::Mat_<cv::Vec3f> &paddedSource,
    cv::Mat_<float> &detMatrix,
    const int neighborSizeDiv2)
{
    int area = ((2 * neighborSizeDiv2 + 1) * (2 * neighborSizeDiv2 + 1));
    // Calculate Mean

#pragma omp parallel for
    for (int imRow = neighborSizeDiv2; imRow < paddedSource.rows - neighborSizeDiv2; imRow++)
    {
        for (int imCol = neighborSizeDiv2; imCol < paddedSource.cols - neighborSizeDiv2; imCol++)
        {
            // ーーーーーーーーーーーーーーーーーー
            // 1. calculate mean
            cv::Vec3f mean = cv::Vec3f(0.0, 0.0, 0.0);
            for (int kernelRow = -neighborSizeDiv2; kernelRow <= neighborSizeDiv2; kernelRow++)
            {
                int newRow = imRow + kernelRow;
                for (int kernelCol = -neighborSizeDiv2; kernelCol <= neighborSizeDiv2; kernelCol++)
                {
                    int newCol = imCol + kernelCol;
                    mean += paddedSource(newRow, newCol);
                }
            }
            mean /= area;

            // 2. calculate elements of Covariance Matrix
            cv::Vec3f diff;
            cv::Vec<float, 6> _covariance;
            cv::Vec<float, 6> covariance = cv::Vec<float, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for (int kernelRow = -neighborSizeDiv2; kernelRow <= neighborSizeDiv2; kernelRow++)
            {
                int newRow = imRow + kernelRow;
                for (int kernelCol = -neighborSizeDiv2; kernelCol <= neighborSizeDiv2; kernelCol++)
                {
                    int newCol = imCol + kernelCol;
                    diff = paddedSource(newRow, newCol) - mean;
                    float _x00 = diff(0) * diff(0);
                    float _x01 = diff(0) * diff(1);
                    float _x02 = diff(0) * diff(2);
                    float _x11 = diff(1) * diff(1);
                    float _x12 = diff(1) * diff(2);
                    float _x22 = diff(2) * diff(2);
                    _covariance = cv::Vec<float, 6>(_x00, _x01, _x02, _x11, _x12, _x22);
                    covariance += _covariance;
                }
            }

            // 3. calculate determinant
            float x00 = covariance(0);
            float x01 = covariance(1);
            float x02 = covariance(2);
            float x11 = covariance(3);
            float x12 = covariance(4);
            float x22 = covariance(5);

            float det = x00 * x11 * x22 + 2 * x01 * x02 * x12 - (x00 * x12 * x12 + x11 * x02 * x02 + x22 * x01 * x01);
            det /= (area * area * area);

            // 4. store determinant
            detMatrix(imRow - neighborSizeDiv2, imCol - neighborSizeDiv2) = det;
        }
    }
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

    cv::Mat_<cv::Vec3f> source = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat_<float> detMatrix(source.rows, source.cols);
    cv::Mat_<cv::Vec3f> paddedSource = cv::Mat_<cv::Vec3f>(source.rows + 2 * neighborSizeDiv2, source.cols + 2 * neighborSizeDiv2);
    cv::copyMakeBorder(source, paddedSource, neighborSizeDiv2, neighborSizeDiv2, neighborSizeDiv2, neighborSizeDiv2, cv::BORDER_REPLICATE);
    cv::Mat_<cv::Vec3b> destination = cv::Mat_<cv::Vec3b>(source.rows, source.cols);

    int kernelSizeDiv2;

    auto begin = chrono::high_resolution_clock::now();

    // Iteration for benchmarking
    const int iter = 1;
    for (int it = 0; it < iter; it++)
    {
        // make padded image
        // use `same` padding

        // calculate determinant of covariance matrix
        calcDetCovarianceMatrix(paddedSource, detMatrix, neighborSizeDiv2);
        // cout << "detMatrix: " << endl;

        // calculate gaussian
        applyDynamicGaussianFilter(source, destination, detMatrix, gaussianFactorRatio, sigma);
        cout << "gaussian" << endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // // print min and max values of the first channel of destination
    // cv::Mat img_bin;
    // cv::cvtColor(destination, img_bin, cv::COLOR_BGR2GRAY);
    // double minVal, maxVal;
    // cv::Point minLoc, maxLoc;
    // cv::minMaxLoc(img_bin, &minVal, &maxVal, &minLoc, &maxLoc);

    // cout << "Min value: " << minVal << endl;
    // cout << "Max value: " << maxVal << endl;
    // cout << "----------------------------------------" << endl;

    // // cast to uchar
    // cv::Mat detBin;
    // cv::cvtColor(detMatrix, detBin, cv::COLOR_BGR2GRAY);

    // cv::minMaxLoc(detBin, &minVal, &maxVal, &minLoc, &maxLoc);
    // cout << "Min value: " << minVal << endl;
    // cout << "Max value: " << maxVal << endl;
    // cout << "----------------------------------------" << endl;

    cv::Mat_<cv::Vec3b> destinationUchar;
    destination.convertTo(destinationUchar, CV_8UC3);
    cv::imwrite("results/image_original.png", source);
    cv::imwrite("results/image_det.png", detMatrix);
    cv::imwrite("results/image_denoised.png", destinationUchar);

    return 0;
}