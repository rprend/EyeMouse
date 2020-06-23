#include "Face.h"

void camux::Face::setCoords(const cv::Point2u topLeft, const cv::Point2u bottomRight) {
    cv::Rect r{topLeft, bottomRight};
    coords_ = r;
}

void camux::Face::setCoords(const unsigned x, const unsigned y, const unsigned endX, const unsigned endY) {
    setCoords(cv::Point2u{x, y}, cv::Point2u{endX, endY});
}

void camux::Face::setCoords(const cv::Rect coords) {
    coords_ = coords;
}

void camux::Face::setConfidence(const float conf) {
    confidence_ = conf;
}
