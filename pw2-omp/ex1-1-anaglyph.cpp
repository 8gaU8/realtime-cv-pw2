#include <iostream>
#include <opencv2/opencv.hpp>
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

int main(int argc, char **argv)
{
    char *filename = argv[1];
    char *anaglyphType = argv[2];

    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <image> <anaglyphType> <[nbThreads]>" << endl;
        cout << "\tanaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    // 関数ポインタを選択
    const AnaglyphFunction selectedAnaglyph = selectAnaglyphFunction(anaglyphType);

    if (selectedAnaglyph == nullptr)
    {
        cout << "Unknown anaglyph type: " << anaglyphType << endl;
        cout << "\tanaglyphType: true, gray, color, halfColor, optimized" << endl;
        return -1;
    }

    // parse the number of threads
    int nbThreads = -1;
    if (argc == 4)
        nbThreads = atoi(argv[3]);
    if (nbThreads != -1)
        omp_set_num_threads(nbThreads);

    cout << "Filename: " << filename << endl;
    cout << "Anaglyph: " << anaglyphType << endl;
    cout << " Threads: " << nbThreads << endl;

    cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat_<cv::Vec3b> destination(source.rows, source.cols / 2);

    const int iter = 100;

    auto begin = chrono::high_resolution_clock::now();
    for (int it = 0; it < iter; it++)
    {
        processImageToAnaglyph(source, destination, selectedAnaglyph);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - begin;

    cout << "Time for " << nbThreads << " threads: " << diff.count() / iter << " s" << endl;
    cout << "\t" << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "\t" << "IPS: " << iter / diff.count() << endl;

    cv::imwrite("results/image_original.png", source);
    cv::imwrite("results/image_processed.png", destination);

    return 0;
}
