#include "FaceDetector.h"

#include <exception>

using cv::Scalar;
using cv::Size;
using cv::Point;

std::string caffe_model = "res10_300x300_ssd_iter_140000.caffemodel";
std::string prototxt_file = "deploy.prototxt";

std::string dlib_68_file = "shape_predictor_68_face_landmarks.dat";

FaceDetector::FaceDetector() {
    method_ = OpenCV_DNN;

    changeMethod(method_);
}

FaceDetector::FaceDetector(Detector method) {
    if (method != OpenCV_DNN) {
        throw std::invalid_argument("Currently, we only support OpenCV detection");
    }

    method_ = method;

    changeMethod(method_);
}

void FaceDetector::changeMethod(Detector method) {
    switch (method) {
    case OpenCV_DNN:
        net_ = cv::dnn::readNetFromCaffe(prototxt_file, caffe_model);
        break;
    }
}

void FaceDetector::detectFace(cv::Mat &frame) {
    switch (method_) {
        case OpenCV_DNN:
            _detectDNN(frame);
            break;
        case Dlib_68:
            _detectDLIB(frame);
            break;
        case HaarCascade:
            _detectHAAR(frame);
            break;
    }
}

void FaceDetector::_detectDNN(cv::Mat &frame) {
	// The dimensions (e.g 720x1080) of the image
	height_ = frame.size[0];
	width_ = frame.size[1];

	// From https://github.com/opencv/opencv/tree/master/samples/dnn:
	//   "To achieve the best accuracy run the model on BGR images resized
	//   to 300x300 applying mean subtraction of values (104, 177, 123) for
	//   each blue, green and red channels correspondingly."
	cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0,
						Size(300, 300), Scalar((104, 177, 123)));

	// Perform forward pass on the current frame - returns a matrix of potential faces.
	// For unknown reason, the matrix is always 1x1x200x7. There are 200 potential
	// faces (why the excess dimensions in front?), each with 7 features, the
	// relevant ones being:
	// 				faces[0,0,i,2] - Float showing the confidence (<1) the ith rectangle is a face
	//				faces[0,0,i,3] - Float showing the scale of the width of the starting x coord. Multiply by width
	//												 for pixel value of the starting x coord.
	//				faces[0,0,i,4] - Float showing the scale of the width of the starting y coord. Multiply by width
	//												 for pixel value of the starting y coord.
	//				faces[0,0,i,5] - Float showing the scale of the width of the ending x coord. Multiply by width
	//												 for pixel value of the ending x coord.
	//				faces[0,0,i,6] - Float showing the scale of the width of the ending y coord. Multiply by width
	//												 for pixel value of the ending y coord.
	net_.setInput(blob);
	cv::Mat faces = net_.forward();

	// Iterate through all the potential faces, only draw ones with high confidence.
	for (int i = 0; i < faces.size[2]; ++i) {
		float confidence = faces.at<float>(i, 2);
		// std::cout << "Confidence: " << faces.at<float>(i, 2) << std::endl;

		if (confidence > FACE_CONFIDENCE_THRESHOLD) {
			// Extract the bounding rectangle and draw on the frame
			int x    = faces.at<float>(i, 3) * width_;
			int y    = faces.at<float>(i, 4) * height_;
			int endX = faces.at<float>(i, 5) * width_;
			int endY = faces.at<float>(i, 6) * height_;

			cv::rectangle(frame, Point(x, y), Point(endX, endY), Scalar(0, 0, 255), 2);

			cv::putText(frame, std::to_string(confidence * 100) + "%", cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255));

		}
	}
}

void FaceDetector::_detectDLIB(cv::Mat &frame) {

}

void FaceDetector::_detectHAAR(cv::Mat &frame) {

}

