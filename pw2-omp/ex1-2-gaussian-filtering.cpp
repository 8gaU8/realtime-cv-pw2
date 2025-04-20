#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // for high_resolution_clock
#include <omp.h>

using namespace std;


void gaussianKernel(int halfSize, float sigma, cv::Mat_<float> &kernelMat)
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
    const cv::Mat_<cv::Vec3b> &paddedSource,
    cv::Mat_<cv::Vec3b> &destination,
    const int kernelSizeDiv2,
    const cv::Mat_<float> &kernelMat)
{
#pragma omp parallel for
  for (int imRow = kernelSizeDiv2; imRow < paddedSource.rows - kernelSizeDiv2; imRow++)
  {
    cv::Vec3f pixelSum;
    for (int imCol = kernelSizeDiv2; imCol < paddedSource.cols - kernelSizeDiv2; imCol++)
    {
      // dual loop for kernel
      pixelSum = cv::Vec3f(0.0);
      for (int kernelRow = -kernelSizeDiv2; kernelRow <= kernelSizeDiv2; kernelRow++)
      {
        int newRow = imRow + kernelRow;
        for (int kernelCol = -kernelSizeDiv2; kernelCol <= kernelSizeDiv2; kernelCol++)
        {
          int newCol = imCol + kernelCol;
          float k = kernelMat(kernelRow + kernelSizeDiv2, kernelCol + kernelSizeDiv2);
          pixelSum += k * paddedSource(newRow, newCol);
        }
      }
      destination(imRow - kernelSizeDiv2, imCol - kernelSizeDiv2) = pixelSum;
    }
  }
}

int main(int argc, char **argv)
{
  char *filename = argv[1];
  char *anaglyphType = argv[2];
  int nbThreads = -1;

  if (argc < 3)
  {
    cout << "Usage: " << argv[0] << " <image> <anaglyphType> <[nbThreads]>" << endl;
    cout << "\tanaglyphType: true, gray, color, halfColor, optimized" << endl;
    return -1;
  }
  else if (argc == 4)
    nbThreads = atoi(argv[3]);

  // parse the anaglyph type
  if (nbThreads != -1)
    omp_set_num_threads(nbThreads);

  cv::Mat_<cv::Vec3b> source = cv::imread(filename, cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> gaussianResult(source.rows, source.cols);
  cv::Mat_<cv::Vec3b> destination(source.rows, source.cols / 2);

  auto begin = chrono::high_resolution_clock::now();

  int kernelSizeDiv2 = 0;
  float sigma = 1.5;

  // make padded image
  cv::Mat_<cv::Vec3b> paddedSource = cv::Mat_<cv::Vec3b>(source.rows + 2 * kernelSizeDiv2, source.cols + 2 * kernelSizeDiv2);
  // use `same` padding
  cv::copyMakeBorder(source, paddedSource, kernelSizeDiv2, kernelSizeDiv2, kernelSizeDiv2, kernelSizeDiv2, cv::BORDER_REPLICATE);

  // make gaussian kernel
  cv::Mat_<float> kernelMat(kernelSizeDiv2 * 2 + 1, kernelSizeDiv2 * 2 + 1);
  gaussianKernel(kernelSizeDiv2, sigma, kernelMat);

  // Iteration for benchmarking
  const int iter = 10;
  for (int it = 0; it < iter; it++)
  {
    applyGaussianFilter(paddedSource, gaussianResult, kernelSizeDiv2, kernelMat);
    // processImageToAnaglyph(gaussianResult, destination, selectedAnaglyph);
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