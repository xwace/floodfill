//
// Created by star on 23-3-20.
//

#ifndef FLOODFILL
#define FLOODFILL
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace FLOODFILL_{
    int floodFill( cv::InputOutputArray _image, cv::InputOutputArray _mask,
                   cv::Point seedPoint, cv::Scalar newVal, cv::Rect* rect,
                   cv::Scalar loDiff, cv::Scalar upDiff, int flags );
    void run();
}

#endif //FLOODFILL
