#pragma once

#include "geometry.hpp"

namespace camux {
    
    // EyeType enum useful for
    enum EyeType {
        Left,
        Right
    };

    /**
     * @brief 
     * 
     */
    class Eye {
    public:
        Eye() :
            type_(Left), coords_{cv::Point2u(0, 0), cv::Point2u(1, 1)} {};

        Eye(const EyeType type, cv::Point2u topLeft, cv::Point2u bottomRight) : 
            type_(type), coords_{topLeft, bottomRight} {};

        Eye(const EyeType& type, const cv::Rect& coords) : 
            type_(type), coords_(coords) {};

        void setCoords(const cv::Rect& coords) { coords_ = coords; }
        cv::Rect getCoords() { return coords_; }
        int getPupilRadius() { return pupil_radius_; }

        cv::Point2u findPupilCenter(cv::Mat& eye);

        int getEyeArea() { return coords_.height * coords_.width; }

        void setConfidence(double conf) { confidence_ = conf; }
        double getConfidence() { return confidence_; }

    private:
        /**
         * @brief A method of finding a dark ellipse in an image (e.g a pupil in an eye) via
         * a combination of blurring, thresholding, and dilating. 
         * 
         * @param eye The
         * @return cv::Point2u The coordinates of the center of the dark ellipse in the provided image.
         */
        cv::Point2u _blurThresholdDilateIsolation(cv::Mat & eye);

        /**
         * @brief A method of finding a dark ellipse in an image (e.g pupil in an eye) 
         * by finding the point with maximimum number of intersections with image gradients. 
         * Only strong gradients are considered  (in places like the borders of a dark ellipse 
         * and a brighter iris)-- these gradients all point through the center of the dark ellipse. 
         * 
         * See https://www.inb.uni-luebeck.de/fileadmin/files/PUBPDFS/TiBa11b.pdf for algorithm description.
         *  
         * 
         * @param eye The image in which to search for a dark ellipse 
         * @return cv::Point2u The coordinates of the center of the dark ellipse in the provided image.
         */
        cv::Point2u _gradientIntersectionIsolation(cv::Mat & eye);

        double _estimateCenterProbabilityHist();


        EyeType type_;
        cv::Rect coords_;
        cv::Point center_;
        int pupil_radius_;
        double confidence_;
    };
}

