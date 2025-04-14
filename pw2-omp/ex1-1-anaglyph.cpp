#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono> // for high_resolution_clock
#include <omp.h>

// anaglyph declares
#include "anaglyphMethods.hpp"

using namespace std;

void processImageToAnaglyph(
    const cv::Mat_<cv::Vec3b> &source,
    cv::Mat_<cv::Vec3b> &destination,
    void (*anaglyphFunc)(const cv::Vec3b, const cv::Vec3b, cv::Vec3b &))
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
            anaglyphFunc(source(row, leftCol), source(row, rightCol), result);
            destination(row, col) = result;
        }
    }
}

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

    // parse the anaglyph type
    void (*selectedAnaglyph)(const cv::Vec3b, const cv::Vec3b, cv::Vec3b &);

    if (strcmp(anaglyphType, "true") == 0)
        selectedAnaglyph = &trueAnaglyph;
    else if (strcmp(anaglyphType, "gray") == 0)
        selectedAnaglyph = &grayAnaglyph;
    else if (strcmp(anaglyphType, "color") == 0)
        selectedAnaglyph = &colorAnaglyph;
    else if (strcmp(anaglyphType, "halfColor") == 0)
        selectedAnaglyph = &halfColorAnaglyph;
    else if (strcmp(anaglyphType, "optimized") == 0)
        selectedAnaglyph = &optimizedAnaglyph;
    else
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

    cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat_<cv::Vec3b> destination(source.rows, source.cols / 2);

    const int iter = 1000;

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
