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
            coords_{camux::Point(0, 0), camux::Point(0, 0)} {};

        /**
         * @brief Construct a new Face object
         * 
         * @param topLeft 
         * @param bottomLeft 
         * @param topRight 
         * @param bottomRight 
         */
        Face(const camux::Point topLeft, const camux::Point bottomRight) : 
            coords_{topLeft, bottomRight} {};

        /**
         * @brief Construct a new Face object
         * 
         * @param coords 
         */
        Face(const camux::Rectangle coords) : 
            coords_(coords) {};

        /**
         * @brief Set the Coords object
         * 
         * @param topLeft 
         * @param bottomLeft 
         * @param topRight 
         * @param bottomRight 
         */
        void setCoords(const camux::Point topLeft, const camux::Point bottomRight);
        /**
         * @brief Set the Coords object
         * 
         * @param coords 
         */
        void setCoords(const camux::Rectangle coords);
        /**
         * @brief Set the Coords object
         * 
         * @param x 
         * @param y 
         * @param endX 
         * @param endY 
         */ 
        void setCoords(const int x, const int y, const int endX, const int endY);

        /**
         * @brief Set the Confidence object
         * 
         * @param conf 
         */
        void setConfidence(const float conf);


    private:
        //
        camux::Rectangle coords_;
        //
        float confidence_;
        //
        Eye left_eye_, right_eye_;
    };
}

