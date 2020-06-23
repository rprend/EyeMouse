#pragma once

#include "geometry.hpp"
#include "Eye.h"

namespace camux {
    /**
     * @brief 
     * 
     */
    class Face {
    public:
        /**
         * @brief Construct a new Face object
         * 
         */
        Face() :
            coords_{cv::Point2u(0, 0), cv::Point2u(0, 0)} {};

        /**
         * @brief Construct a new Face object
         * 
         * @param topLeft 
         * @param bottomLeft 
         * @param topRight 
         * @param bottomRight 
         */
        Face(const cv::Point2u topLeft, const cv::Point2u bottomRight) : 
            coords_{topLeft, bottomRight} {};

        /**
         * @brief Construct a new Face object
         * 
         * @param coords 
         */
        Face(const cv::Rect coords) : 
            coords_(coords) {};

        /**
         * @brief Set the Coords object
         * 
         * @param topLeft 
         * @param bottomLeft 
         * @param topRight 
         * @param bottomRight 
         */
        void setCoords(const cv::Point2u topLeft, const cv::Point2u bottomRight);
        /**
         * @brief Set the Coords object
         * 
         * @param coords 
         */
        void setCoords(const cv::Rect coords);
        /**
         * @brief Set the Coords object
         * 
         * @param x 
         * @param y 
         * @param endX 
         * @param endY 
         */ 
        void setCoords(const unsigned x, const unsigned y, const unsigned endX, const unsigned endY);

        /**
         * @brief Set the Confidence object
         * 
         * @param conf 
         */
        void setConfidence(const float conf);


    private:
        //
        cv::Rect coords_;
        //
        float confidence_;
        //
        Eye left_eye_, right_eye_;
    };
}

