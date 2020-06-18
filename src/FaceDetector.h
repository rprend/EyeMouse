#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

// Confidence threshold (>0, <1.0) for deciding if a feature is a face
const float FACE_CONFIDENCE_THRESHOLD = 0.6;

/**
 * @brief 
 * 
 */
enum Detector {
    OpenCV_DNN,
    Dlib_68,
    HaarCascade
};

class FaceDetector {
public:
    /**
     * @brief Construct a default Face Detector object
     * The default FaceDetector object uses the OpenCV DNN
     * for face detection
     */
    FaceDetector();
    FaceDetector(Detector method);

    /**
     * @brief 
     * 
     * @param frame 
     */
    void detectFace(cv::Mat &frame);

    /**
     * @brief Opens the files required for the face detection method and initializes
     * the required data structures (e.g neural net)
     * 
     * @param method The method of detection (neural net, haar cascade, dlib) to use for face detection now
     */
    void changeMethod(Detector method);

private:
    Detector method_;
    int height_ = 720;
    int width_ = 1080;

    // Neural net for the OpenCv face detection method. Initialized when we select
    // OpenCvDNN as our detection method.
    cv::dnn::Net net_;
    dlib::frontal_face_detector dlib_;

    void _detectDNN(cv::Mat &frame);
    void _detectHAAR(cv::Mat &frame);
    void _detectDLIB(cv::Mat &frame);
};