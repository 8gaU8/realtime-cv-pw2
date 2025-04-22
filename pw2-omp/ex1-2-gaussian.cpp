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

  cout << "   Filename: " << filename << endl;
  cout << "   Anaglyph: " << anaglyphType << endl;
  cout << "Kernel size: " << kernelSizeDiv2 << endl;
  cout << "      Sigma: " << sigma << endl;
  cout << "    Threads: " << nbThreads << endl;

  cv::Mat_<cv::Vec3b> source = cv::imread(filename, cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> gaussianResult(source.rows, source.cols);
  cv::Mat_<cv::Vec3b> destination(source.rows, source.cols / 2);

  auto begin = chrono::high_resolution_clock::now();

  // make gaussian kernel
  cv::Mat_<float> kernelMat(kernelSizeDiv2 * 2 + 1, kernelSizeDiv2 * 2 + 1);
  makeGaussianKernel(kernelSizeDiv2, sigma, kernelMat);

  // Iteration for benchmarking
  const int iter = 20;
  for (int it = 0; it < iter; it++)
  {
    applyGaussianFilter(source, gaussianResult, kernelSizeDiv2, kernelMat);
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
