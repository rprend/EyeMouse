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


// Tunable confidence threshold (>0, <1.0) for deciding if a feature is a face
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

/**
 * @brief A detector of a face and its eyes. You construct it with face and eye objects
 *  as well as an image which you want to detect a face and its eyes on and a selection of a 
 *  face detection method you'd like to use (see enum Detector above). Call the detect method
 *  with that image to save the relevant info. Can draw bounding boxes for the face/eyes as well as
 *  a 68 landmark feature summary of the image.
 *
 */
class FaceEyeDetector {
public:
    /**
     * @brief Construct a default Face Detector object
     * The default FaceEyeDetector object uses the OpenCV DNN
     * for face detection.
     *
     * @param face The face object to write face detection info
     * @param l_eye The left eye object to write eye detection info
     * @param r_eye The left eye object to write eye detection info
     */
    FaceEyeDetector(camux::Face &face, camux::Eye &l_eye, camux::Eye &r_eye) : 
      face_(face), left_(l_eye), right_(r_eye) {
        // Default method is the DNN
        method_ = OpenCV_DNN;

        changeMethod(method_);
    };
    /**
     * @brief Construct a new Face Detector object with a specified method
     *
     * @param method The method to use for face detection
     * @param face The face to write face detection info
     * @param l_eye The left eye object to write eye detection info
     * @param r_eye The left eye object to write eye detection info
     */
    FaceEyeDetector(const Detector method, camux::Face &face, camux::Eye &l_eye, camux::Eye &r_eye) :
      face_(face), left_(l_eye), right_(r_eye) {
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

    /**
     * @brief Draw the bounding rectangle of the last detected face on a frame.
     *
     * TODO: See drawEyes() for why this is a poor design for this function.
     *
     * @param frame The cv::Mat image to draw the face onto
     */
    void drawFace(cv::Mat& frame);

    /**
     * @brief Draw the bounding rectangle of the last detected eyes on a frame. Requires
     * that you pass the same frame you used in detecting the face/eyes in order for the coordinates to
     * match. 
     * TODO: That's dumb. Save a reference to the frame when we run detect(), since the bounding rectangle
     * is entirely relative to the coordinate system of that frame. It's useless for frames other than those
     * with the same coordinate system.
     *
     * @param frame The image to draw the face onto.
     */
    void drawEyes(cv::Mat& frame);

    void drawLandmarks(cv::Mat& frame);

    /**
     * @brief Get the Face object
     * 
     * @return camux::Face& 
     */
    camux::Face & getFace() { return face_; }

private:
    // The method of facial recognition to use.
    Detector method_;

    // The height and width of the last frame used.
    int height_ = 720;
    int width_ = 1080;

    // The face object reference to write the most probable face to.
    camux::Face & face_;
    camux::Eye & left_;
    camux::Eye & right_;

    // The facial landmarks (other than those belonging to the eyes)
    std::vector<cv::Point2u> landmarks_;

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
