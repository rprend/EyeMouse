#pragma once

#include <opencv2/imgproc.hpp>
#include <iostream>

namespace camux {
    struct Point {
        Point() : x(0), y(0) {};
        Point(int x, int y) : x(x), y(y) {};
        Point(cv::Point &p): x(p.x), y(p.y) {};

        unsigned x;
        unsigned y;
    };

    typedef std::vector<Point> Points;

    struct Rectangle {
        Rectangle(Point &tl, Point &br) : topLeft(tl), bottomRight(br) {};
        Rectangle(const Point &tl, const Point &br) : topLeft(tl), bottomRight(br) {};

        Point topLeft;
        Point bottomRight;
    };

    cv::Point toCvPoint(const Point &p);

    void drawRectangle(cv::Mat &frame, int x, int y, int endX, int endY);
    void drawRectangle(cv::Mat &frame, const Point topLeft, const Point bottomRight);
    void drawRectangle(cv::Mat &frame, const Rectangle &coords);

    /**
     * @brief Find the bounding rectangle for a group of points. O(len(points)).
     * Finds the min and max x & y coords.
     * 
     * @param points A vector of n points.
     * @return Rectangle The bounding rectangle of all of the points. 
     */
    Rectangle boundingRect(Points & points);
}
