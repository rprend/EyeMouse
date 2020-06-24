#pragma once

#include "geometry.hpp"
#include "Eye.h"

namespace camux {

    /**
     * @brief A face detected on camera. Defined by a bounding rectangle, a confidence level
     * of the detection, and objects for facial features (e.g eyes).
     * 
     */
    class Face {
    public:
        /**
         * @brief Construct a new default Face. Constructs the bounding rectangle as 
         * area 0 bounding by the points [(0, 0) (0, 0)]
         * 
         */
        Face() :
            coords_{cv::Point2u(0, 0), cv::Point2u(0, 0)} {};

        /**
         * @brief Construct a new Face with passed points.
         * 
         * @param topLeft The Point of the top left of the bounding box.
         * @param bottomRight The Point of the bottom right of the bounding box.
         */
        Face(const cv::Point2u topLeft, const cv::Point2u bottomRight) : 
            coords_{topLeft, bottomRight} {};

        /**
         * @brief Construct a new Face with bounding box of the passed rectangle.
         * 
         * @param coords The Rectangle defining the bounding box of the face.
         */
        Face(const cv::Rect coords) : 
            coords_(coords) {};

        /**
         * @brief Set the Coords using a top left point and a bottom right point. cv::Points
         * are of the form (x, y)
         * 
         * @param topLeft The cv::Point of the top left of the bounding box.
         * @param bottomRight The cv::Point of the bottom right of the bounding box.
         */
        void setCoords(const cv::Point2u topLeft, const cv::Point2u bottomRight);
        
        /**
         * @brief Set the Coords using a rectangle bounding box. See cv::Rect for 
         * documentation (x, y, width, height)
         * 
         * @param coords The rectangle of the bounding box to set.
         */
        void setCoords(const cv::Rect coords);

        /**
         * @brief Set the Coords using x,y coordinates of the top left and bottom right
         * points of the bounding rectangle of the face.
         * 
         * @param x The x coordinate of the top left of the bounding rectangle.
         * @param y The y coordinate of the top left of the bounding rectangle.
         * @param endX The x coordinate of the bottom right of the bounding rectangle.
         * @param endY The y coordinate of the bottom right of the bounding rectangle.
         */ 
        void setCoords(const unsigned x, const unsigned y, const unsigned endX, const unsigned endY);

        /**
         * @brief Set the confidence attribute of the face.
         * 
         * @param conf The confidence to set to.
         */
        void setConfidence(const float conf);

        cv::Rect getCoords() { return coords_; }

    private:
        // The rectangle bounding box of the face. Coordinates on the screen
        cv::Rect coords_;
        // The confidence level of the detected face. 0 if no face, negative if no confidence
        // computed.
        float confidence_;
        // Two eyes on the face. The eyes are bounded by [(0,0), (0,0)] if none was detected. 
        Eye left_eye_, right_eye_;
    };
}

