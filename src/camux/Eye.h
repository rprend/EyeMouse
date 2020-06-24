#pragma once

#include "geometry.hpp"

namespace camux {
    
    // EyeType enum useful for
    enum EyeType {
        Left,
        Right
    };

    
    class Eye {
    public:
        Eye() :
            type_(Left), coords_{cv::Point2u(0, 0), cv::Point2u(0, 0)} {};

        Eye(const EyeType type, cv::Point2u topLeft, cv::Point2u bottomRight) : 
            type_(type), coords_{topLeft, bottomRight} {};

        Eye(const EyeType& type, const cv::Rect& coords) : 
            type_(type), coords_(coords) {};

        void setCoords(const cv::Rect& coords) { coords_ = coords; }
        cv::Rect getCoords() { return coords_; }
        int getPupilRadius() { return pupil_radius_; }

        cv::Point2u findPupilCenter(cv::Mat& eye);

    private:
        EyeType type_;
        cv::Rect coords_;
        cv::Point center_;
        int pupil_radius_;
    };
}

