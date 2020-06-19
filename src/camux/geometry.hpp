#pragma once

#include <opencv2/imgproc.hpp>

namespace camux {
    struct Point {
        Point(int x, int y) : x(x), y(y) {};
        Point(cv::Point &p): x(p.x), y(p.y) {};

        int x;
        int y;
    };

    struct Rectangle {
        Point topLeft;
        Point bottomRight;
    };

    cv::Point toCvPoint(const Point &p);

    void drawRectangle(cv::Mat &frame, int x, int y, int endX, int endY);
    void drawRectangle(cv::Mat &frame, const Point topLeft, const Point bottomRight);
    void drawRectangle(cv::Mat &frame, const Rectangle coords);
}

