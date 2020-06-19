#include "Face.h"

void camux::Face::setCoords(const camux::Point topLeft, const camux::Point bottomRight) {
    camux::Rectangle r{topLeft, bottomRight};
    coords_ = r;
}

void camux::Face::setCoords(const int x, const int y, const int endX, const int endY) {
    setCoords(Point{x, y}, Point{endX, endY});
}

void camux::Face::setCoords(const camux::Rectangle coords) {
    coords_ = coords;
}

void camux::Face::setConfidence(const float conf) {
    confidence_ = conf;
}
