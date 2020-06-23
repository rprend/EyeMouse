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
    unsigned min_x = UINT_MAX;
    unsigned min_y = UINT_MAX;
    unsigned max_x = 1;
    unsigned max_y = 1;

    for (int i = 0; i < points.size(); ++i) {
        cv::Point2u p = points[i];
        max_x = std::max(p.x, max_x);
        max_y = std::max(p.y, max_y);
        min_x = std::min(p.x, min_x);
        min_y = std::min(p.y, min_y);
    }

    return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}
 
std::ostream& camux::operator<< (std::ostream& os, const cv::Point2u& p) {
    os << "(" << p.x << ", " << p.y << ")";
    return os;
}

std::ostream& camux::operator<< (std::ostream& os, const cv::Rect& r) {
    os << "[" << cv::Point2u(r.x, r.y) << ", " << 
        cv::Point2u(r.x+r.width, r.y+r.height) << "]";
    return os;
}