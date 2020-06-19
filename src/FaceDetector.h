#pragma once

#include "camux/Eye.h"
#include "camux/Face.h"
#include "camux/geometry.hpp" 


#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>


// Confidence threshold (>0, <1.0) for deciding if a feature is a face
const float FACE_CONFIDENCE_THRESHOLD = 0.6;

/**
 * @brief The different types of face detection methods
 * 
 * OpenCv_DNN: A pretrained deep neural network trained to find faces.
 *  Pros: Most accurate of the three. Second fastest (50-100 ms latency)
 *  Cons: No eye location provided - need to train a model for that
 * 
 * Dlib 68 landmark locator: Pre trained cascade of regression trees.
 *  Pros: Locates 68 different facial landmarks, including 6 surrounding the eyes
 *  Cons: Slowest (~150 ms latency). Not as accurate as DNN (needs to be front facing)
 * 
 * Haar Cascades:
 *  Pros:
 *  Cons:
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
     * for face detection. Face info will be not be saved (use a 
     * constructor which passes a face object to save that info).
     */
    // FaceDetector() : face_() {};

    /**
     * @brief Construct a default Face Detector object
     * The default FaceDetector object uses the OpenCV DNN
     * for face detection.
     * 
     * @param face The face object to write face detection info
     */
    FaceDetector(camux::Face &face) : face_(face) {
        // Default method is the DNN
        method_ = OpenCV_DNN;

        changeMethod(method_);       
    };
    /**
     * @brief Construct a new Face Detector object with a specified method
     * 
     * @param method The method to use for face detection
     * @param face The face to write face detection info
     */
    FaceDetector(const Detector method, camux::Face &face) : face_(face) {
        // TODO: Implement HaarCascades and remove.
        if (method == HaarCascade) {
            throw std::invalid_argument("Currently, we don't support HaarCascade detection");
        }

        method_ = method;
        changeMethod(method_);
    };

    /**
     * @brief Performs the currently selected facial recognition method on an image.
     * Will modify the image to indicate the face.
     * 
     * @param frame The OpenCV style image to find face on.
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
    // The method of facial recognition to use.
    Detector method_;

    // The height and width of the last frame used.
    int height_ = 720;
    int width_ = 1080;

    // The face object reference to write the most probable face to.
    camux::Face & face_;

    // Neural net for the OpenCv face detection method. Initialized when we select
    // OpenCvDNN as our detection method.
    cv::dnn::Net net_;

    // Loads "shape_predictor_68_face_landmarks.dat" file, which is a pre-trained
    // cascade of regression tree implemented using "One Millisecond face alignment 
    // with an ensemble of regression trees"
    dlib::shape_predictor dlib_sp_;    
    dlib::frontal_face_detector dlib_;

    /**
     * @brief Performs the OpenCVDNN facial recognition method on an image.
     * Will draw a bounding box on the image to indicate the face. 
     * 
     * @param frame The OpenCV style image to find a face on.
     */
    void _detectDNN(cv::Mat &frame);
    /**
     * @brief Performs a Haar Cascade facial recognition method on an image.
     * Will modify the image to indicate the face.
     * 
     * @param frame The OpenCV style image to find a face on.
     */
    void _detectHAAR(cv::Mat &frame);
    /**
     * @brief Performs the Dlib 68 landmark facial recognition method on an image.
     * Will draw the 68 landmarks on the image to indicate the face.
     * 
     * @param frame The OpenCV style image to find a face on.
     */
    void _detectDLIB(cv::Mat &frame);
};