//
// Created by star on 23-3-20.
//

#include "floodFill.h"

using namespace std;
using namespace cv;

namespace FLOODFILL{

    int floodFill(cv::Mat &image,Point seedPoint, int seedVal, int newVal) {

        if ((unsigned) seedPoint.x >= image.cols
            or (unsigned) seedPoint.y >= image.rows)
        {
            return -1;
        }


        uchar* val = image.ptr(seedPoint.y) + seedPoint.x;

        if (*val != seedVal) return -1;
        *val = newVal;

        Point up, down, left, right;
        up    = {seedPoint.x, seedPoint.y - 1};
        down  = {seedPoint.x, seedPoint.y + 1};
        left  = {seedPoint.x - 1, seedPoint.y};
        right = {seedPoint.x + 1, seedPoint.y};

        FLOODFILL::floodFill(image,up,seedVal,newVal);
        FLOODFILL::floodFill(image,down,seedVal,newVal);
        FLOODFILL::floodFill(image,left,seedVal,newVal);
        FLOODFILL::floodFill(image,right,seedVal,newVal);
    }

    int floodFill_(cv::Mat &image,Point seedPoint, int seedVal, int newVal) {

        std::stack<Point>stack;
        stack.emplace(seedPoint);

        while (not stack.empty()) {
            auto p = stack.top();
            stack.pop();

            if ((unsigned) p.x >= image.cols
                or (unsigned) p.y >= image.rows) {
                continue;
            }

            uchar *val = image.ptr(p.y) + p.x;
            if (*val != seedVal) continue;
            *val = newVal;

            Point up, down, left, right;
            up = {p.x, p.y - 1};
            down = {p.x, p.y + 1};
            left = {p.x - 1, p.y};
            right = {p.x + 1, p.y};

            stack.emplace(up);
            stack.emplace(down);
            stack.emplace(left);
            stack.emplace(right);
        }
    }

    void run(){
        cv::Mat img(100,100,0,Scalar(0));
        img(Rect(0,0,10,10)).setTo(37);
        img(Rect(30,30,10,10)).setTo(37);
        img(Rect(70,70,10,10)).setTo(37);
        cv::circle(img,Point(35,35),30,37,-1);

        Point seed =Point(35,35);
        int seedVal = img.at<uchar>(seed);
        int newVal = 5;

        FLOODFILL::floodFill_(img,seed,seedVal, newVal);
        namedWindow("img",2);
        imshow("img",img==5);
        waitKey();
        cout<<img<<endl;
    }
}