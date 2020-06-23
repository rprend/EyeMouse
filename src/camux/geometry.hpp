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

    /**
     * @brief Find the bounding rectangle for a group of points with p_err extra margin.
     *  O(len(points)). Finds the min and max x & y coords. E.g if bounding rect is x=100, y=100, w=100, y=100
     *  and p_err is .20 (20%), we set the width&height to 125% of their values and subtract half the difference
     *  from the start points (where possible). So our new rectangle has w=120, h=120, x=90, y=90.
     *  If we cannot fit center the new rectangle, we try the best we can. 
     * 
     * @param points A vector of n points.
     * @param p_err 
     * @return cv::Rect 
     */
    cv::Rect boundingRectMargin(const Points& points, float p_err);

}

