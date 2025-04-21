#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // for high_resolution_clock
#include <omp.h>

using namespace std;

#include "anaglyphMethods.hpp"

void makeGaussianKernel(int halfSize, float sigma, cv::Mat_<float> &kernelMat)
{
  for (int i = 0; i < 2 * halfSize + 1; i++)
  {
    for (int j = 0; j < 2 * halfSize + 1; j++)
    {
      float x = i - halfSize;
      float y = j - halfSize;
      kernelMat(i, j) = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
    }
  }
}

void applyGaussianFilter(
    const cv::Mat_<cv::Vec3b> &src,
    cv::Mat_<cv::Vec3b> &dst,
    const int kernelSizeDiv2,
    const cv::Mat_<float> &kernelMat)
{

#pragma omp parallel for

  for (int y = 0; y < src.rows; y++)
  {
    cv::Vec3f pixelSum;
    for (int x = 0; x < src.cols; x++)
    {

      int minCol = 0;
      int maxCol = src.cols / 2 - 1;
      if (x >= src.cols / 2)
      {
        minCol = src.cols / 2;
        maxCol = src.cols - 1;
      }

      // dual loop for kernel
      pixelSum = cv::Vec3f(0.0);
      for (int kernelRow = -kernelSizeDiv2; kernelRow <= kernelSizeDiv2; kernelRow++)
      {
        int src_y = clamp(y + kernelRow, 0, src.rows - 1);

        for (int kernelCol = -kernelSizeDiv2; kernelCol <= kernelSizeDiv2; kernelCol++)
        {
          int src_x = clamp(x + kernelCol, minCol, maxCol);

          float weight = kernelMat(kernelRow + kernelSizeDiv2, kernelCol + kernelSizeDiv2);
          pixelSum += src(src_y, src_x) * weight;
        }
      }
      dst(y, x) = pixelSum;
    }
  }
}

int main(int argc, char **argv)
{

  if (argc < 5)
  {
    cout << "Usage: " << argv[0] << " <image> <anaglyphType> <kernelSizeDiv2> <sigma> <[nbThreads]>" << endl;
    cout << "\tanaglyphType: true, gray, color, halfColor, optimized" << endl;
    return -1;
  }
  char *filename = argv[1];
  char *anaglyphType = argv[2];
  int kernelSizeDiv2 = atoi(argv[3]);
  float sigma = atof(argv[4]);

  int nbThreads = -1;
  if (argc == 6)
    nbThreads = atoi(argv[5]);

  if (nbThreads != -1)
    omp_set_num_threads(nbThreads);

  // parse the anaglyph type
  const AnaglyphFunction selectedAnaglyph = selectAnaglyphFunction(anaglyphType);

  if (selectedAnaglyph == nullptr)
  {
    cout << "Invalid anaglyph type: " << anaglyphType << endl;
    cout << "anaglyphType: true, gray, color, halfColor, optimized" << endl;
    return -1;
  }

  cv::Mat_<cv::Vec3b> source = cv::imread(filename, cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> gaussianResult(source.rows, source.cols);
  cv::Mat_<cv::Vec3b> destination(source.rows, source.cols / 2);

  auto begin = chrono::high_resolution_clock::now();

  // make gaussian kernel
  cv::Mat_<float> kernelMat(kernelSizeDiv2 * 2 + 1, kernelSizeDiv2 * 2 + 1);
  makeGaussianKernel(kernelSizeDiv2, sigma, kernelMat);

  // Iteration for benchmarking
  const int iter = 10;
  for (int it = 0; it < iter; it++)
  {
    cout << "Processing image with Gaussian filter..." << endl;
    applyGaussianFilter(source, gaussianResult, kernelSizeDiv2, kernelMat);
    cout << "Processing image to anaglyph..." << endl;
    processImageToAnaglyph(gaussianResult, destination, selectedAnaglyph);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::imwrite("results/image_original.png", source);
  cv::imwrite("results/image_processed.png", gaussianResult);
  cv::imwrite("results/image_anaglyph.png", destination);

  return 0;
}