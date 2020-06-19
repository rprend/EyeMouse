#include "geometry.hpp"

cv::Point camux::toCvPoint(const Point &p) {
    return cv::Point(p.x, p.y);
}

void camux::drawRectangle(cv::Mat &frame, int x, int y, int endX, int endY) {
    cv::rectangle(frame, cv::Point(x, y), cv::Point(endX, endY), cv::Scalar(0, 0, 255), 2);
}

void camux::drawRectangle(cv::Mat &frame, const camux::Point topLeft, const camux::Point bottomRight) {
    camux::drawRectangle(frame, topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
}

void drawRectangle(cv::Mat &frame, const camux::Rectangle coords) {
    camux::drawRectangle(frame, coords.topLeft, coords.bottomRight);
}
