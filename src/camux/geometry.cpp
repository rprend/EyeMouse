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

void camux::drawRectangle(cv::Mat &frame, const camux::Rectangle &coords) {
    camux::drawRectangle(frame, coords.topLeft, coords.bottomRight);
}

camux::Rectangle camux::boundingRect(camux::Points & points) {
    unsigned min_x, min_y = INT_MAX;
    unsigned max_x, max_y = 0;

    for (camux::Point p : points) {
        if (p.x > max_x) max_x = p.x;
        else if (p.x < min_x) min_x = p.x;
        if (p.y > max_y) max_y = p.y;
        else if (p.y < min_y) min_y = p.y;
    }

    return camux::Rectangle(camux::Point(min_x, max_y), camux::Point(max_x, min_y));
}
