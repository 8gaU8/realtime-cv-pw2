#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // for high_resolution_clock
#include <omp.h>

using namespace std;

// anaglyph declares
// ===============================================================
using AnaglyphFunction = void (*)(const cv::Vec3b, const cv::Vec3b, cv::Vec3b &);

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

// process input image to anaglyph image
void processImageToAnaglyph(
    const cv::Mat_<cv::Vec3b> &source,
    cv::Mat_<cv::Vec3b> &destination,
    const AnaglyphFunction anaglyphFunction)
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
// ===============================================================

void applyDynamicGaussianFilter(
    const cv::Mat_<cv::Vec3f> &src,
    cv::Mat_<cv::Vec3b> &destination,
    const cv::Mat_<float> &detMatrixDivGFR,
    const float gaussianFactorRatio,
    const float sigma)
{
    float twoSigmaSquare = 2.0 * sigma * sigma;
    float twoPiSigmaSquare = M_PI * twoSigmaSquare;

#pragma omp parallel for
    for (int imRow = 0; imRow < src.rows; imRow++)
    {
        for (int imCol = 0; imCol < src.cols; imCol++)
        {
            int minCol = 0;
            int maxCol = src.cols / 2 - 1;
            if (imCol >= src.cols / 2)
            {
                minCol = src.cols / 2;
                maxCol = src.cols - 1;
            }

            // calculate kernel size
            float det = detMatrixDivGFR(imRow, imCol);
            det = max(det, 0.10f); // avoid division by zero
            int kernelSizeDiv2 = static_cast<int>(gaussianFactorRatio / det);
            kernelSizeDiv2 = clamp(kernelSizeDiv2, 1, 20); // limit kernel size

            // dual loop for kernel
            cv::Vec3f pixelSum = cv::Vec3f(0.0, 0.0, 0.0);
            for (int kernelRow = -kernelSizeDiv2; kernelRow <= kernelSizeDiv2; kernelRow++)
            {
                int newRow = clamp(imRow + kernelRow, 0, src.rows - 1);
                for (int kernelCol = -kernelSizeDiv2; kernelCol <= kernelSizeDiv2; kernelCol++)
                {
                    int newCol = clamp(imCol + kernelCol, minCol, maxCol);

                    float weight = exp(-(float(kernelRow * kernelRow + kernelCol * kernelCol) / twoSigmaSquare)) / twoPiSigmaSquare;

                    pixelSum += src(newRow, newCol) * weight;
                }
            }
            destination(imRow, imCol) = pixelSum;
        }
    }
}

void calcDetCovarianceMatrix(
    const cv::Mat_<cv::Vec3f> &paddedSrc,
    cv::Mat_<float> &detMatrix,
    const int neighborSizeDiv2)
{
    int squareNeighborSize = ((2 * neighborSizeDiv2 + 1) * (2 * neighborSizeDiv2 + 1));

#pragma omp parallel for
    for (int imRow = neighborSizeDiv2; imRow < paddedSrc.rows - neighborSizeDiv2; imRow++)
    {
        for (int imCol = neighborSizeDiv2; imCol < paddedSrc.cols - neighborSizeDiv2; imCol++)
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
                    rgbMean += paddedSrc(newRow, newCol);
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
                    diff = paddedSrc(newRow, newCol) - rgbMean;
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
            // Theoritically, determinant should be divided by the number of pixels
            // but it is not necessary since we have `gaussianFactorRatio` parameter
            // det /= squareNeighborSize * squareNeighborSize * squareNeighborSize;

            // 5. store determinant
            detMatrix(imRow - neighborSizeDiv2, imCol - neighborSizeDiv2) = det;
        }
    }

    // min-max scaling
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(detMatrix, &minVal, &maxVal, nullptr, nullptr);

    detMatrix = (detMatrix - minVal) / (maxVal - minVal);
}

int main(int argc, char **argv)
{
    // parse the arguments
    if (argc < 6)
    {
        cout << "Usage: " << argv[0] << " <image> <anaglyphType> <neighborSizeDiv2> <gaussianFactorRatio> <sigma>" << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    char *filename = argv[1];
    const char *anaglyphType = argv[2];
    int neighborSizeDiv2 = atoi(argv[3]);
    float gaussianFactorRatio = atof(argv[4]);
    float sigma = atof(argv[5]);

    const AnaglyphFunction selectedAnaglyph = selectAnaglyphFunction(anaglyphType);

    if (selectedAnaglyph == nullptr)
    {
        cout << "Invalid anaglyph type: " << anaglyphType << endl;
        cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    int nbThreads = -1;
    if (argc == 7)
        nbThreads = atoi(argv[6]);

    if (nbThreads != -1)
        omp_set_num_threads(nbThreads);

    cout << "             Filename: " << filename << endl;
    cout << "             Anaglyph: " << anaglyphType << endl;
    cout << "        Neighbor size: " << neighborSizeDiv2 << endl;
    cout << "Gaussian factor ratio: " << gaussianFactorRatio << endl;
    cout << "                Sigma: " << sigma << endl;
    cout << "              Threads: " << nbThreads << endl;

    // --------------------------------------

    // load image
    const cv::Mat_<cv::Vec3f> src = cv::imread(filename, cv::IMREAD_COLOR);

    cv::Mat_<float> detMatrix(src.rows, src.cols);

    // make padded image
    // use `same` padding
    cv::Mat_<cv::Vec3f> paddedSrc = cv::Mat_<cv::Vec3f>(src.rows + 2 * neighborSizeDiv2, src.cols + 2 * neighborSizeDiv2);
    cv::copyMakeBorder(src, paddedSrc, neighborSizeDiv2, neighborSizeDiv2, neighborSizeDiv2, neighborSizeDiv2, cv::BORDER_REPLICATE);

    cv::Mat_<cv::Vec3b> gaussian = cv::Mat_<cv::Vec3b>(src.rows, src.cols);
    cv::Mat_<cv::Vec3b> anaglyph = cv::Mat_<cv::Vec3b>(src.rows, src.cols / 2);

    auto begin = chrono::high_resolution_clock::now();

    // Iteration for benchmarking
    const int iter = 20;
    for (int it = 0; it < iter; it++)
    {

        // calculate determinant of covariance matrix
        calcDetCovarianceMatrix(paddedSrc, detMatrix, neighborSizeDiv2);

        // calculate gaussian
        applyDynamicGaussianFilter(src, gaussian, detMatrix, gaussianFactorRatio, sigma);

        // apply anaglyph
        processImageToAnaglyph(gaussian, anaglyph, selectedAnaglyph);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::imwrite("results/image_original.png", src);
    cv::imwrite("results/image_det.png", detMatrix * 255);
    cv::imwrite("results/image_anaglyph.png", anaglyph);

    return 0;
}
