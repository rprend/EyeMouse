#include "geometry.hpp"

void camux::drawRectangle(cv::Mat &frame, int x, int y, int endX, int endY) {
    cv::rectangle(frame, cv::Point(x, y), cv::Point(endX, endY), cv::Scalar(0, 0, 255), 2);
}

void camux::drawRectangle(cv::Mat &frame, const cv::Point2u topLeft, const cv::Point2u bottomRight) {
    camux::drawRectangle(frame, topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
}

void camux::drawRectangle(cv::Mat &frame, const cv::Rect &coords) {
    camux::drawRectangle(frame, coords.x, coords.y, coords.x + coords.width, coords.y + coords.height);
}

cv::Rect camux::boundingRect(const camux::Points& points) { 
    if (points.size() == 0) return cv::Rect(); 
    // Note that min_x, min_y = UINT_MAX does NOT do the same assignment as doing them 
    // separately. 
    unsigned min_x = UINT_MAX;
    unsigned min_y = UINT_MAX;
    unsigned max_x = 1;
    unsigned max_y = 1;

    // Iterate through the points, find the min and max x and y values. These define the bounding
    // rectangle to return.
    for (int i = 0; i < points.size(); ++i) {
        cv::Point2u p = points[i];
        max_x = std::max(p.x, max_x);
        max_y = std::max(p.y, max_y);
        min_x = std::min(p.x, min_x);
        min_y = std::min(p.y, min_y);
    }

    return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

cv::Rect camux::boundingRectMargin(const camux::Points& points, const float p_err_width, const float p_err_height) {
    if (points.size() == 0) return cv::Rect(); 
    
    // Find the bounding rectangle normally
    cv::Rect r = camux::boundingRect(points);

    // Increase the width and height of the bounding box by p_err percent.
    unsigned width = r.width * (1+p_err_width);
    unsigned height = r.height * (1+p_err_height);
    
    // Adjust the x and y coordinates so the center of the bounding box stays in the same place.
    // E.g 100x100 px adjusted by 10% gives 110x110. So we decrease the starting x & y by 5.
    // Can't have negative coordinates, so max with 0.
    unsigned x = std::max(r.x - (width - r.width) / 2, (unsigned) 0);
    unsigned y = std::max(r.y - (height - r.height) / 2, (unsigned) 0); 

    return cv::Rect(x, y, width, height);
}
 
// std::ostream& camux::operator<< (std::ostream& os, const cv::Point2u& p) {
//     os << "(" << p.x << ", " << p.y << ")";
//     return os;
// }

// std::ostream& camux::operator<< (std::ostream& os, const cv::Rect& r) {
//     os << "[" << cv::Point2u(r.x, r.y) << ", " << 
//         cv::Point2u(r.x+r.width, r.y+r.height) << "]";
//     return os;
// }