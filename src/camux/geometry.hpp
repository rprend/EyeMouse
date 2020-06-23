#pragma once

#include <opencv2/imgproc.hpp>
#include <iostream>

namespace cv {
    typedef Point_<unsigned> Point2u;
}

namespace camux {
    typedef std::vector<cv::Point2u> Points;

    std::ostream& operator<< (std::ostream& os, const cv::Point2u& p);
    std::ostream& operator<< (std::ostream& os, const cv::Rect& r);

    void drawRectangle(cv::Mat &frame, int x, int y, int endX, int endY);
    void drawRectangle(cv::Mat &frame, const cv::Point2u topLeft, const cv::Point2u bottomRight);
    void drawRectangle(cv::Mat &frame, const cv::Rect &coords);

    /**
     * @brief Find the bounding rectangle for a group of points. O(len(points)).
     * Finds the min and max x & y coords.
     * 
     * @param points A vector of n points.
     * @return Rectangle The bounding rectangle of all of the points. 
     */
    cv::Rect boundingRect(const Points& points);


}

