#include "FaceDetector.h"

#include <exception>

using cv::Scalar;
using cv::Size;
using cv::Point;

// NN model files for OpenCV_DNN
std::string caffe_model = "res10_300x300_ssd_iter_140000.caffemodel";
std::string prototxt_file = "deploy.prototxt";

// Cascade file for the dlib cascade.
std::string dlib_68_file = "shape_predictor_68_face_landmarks.dat";

FaceDetector::FaceDetector() {
    // Default method is the DNN
    method_ = OpenCV_DNN;

    changeMethod(method_);
}

FaceDetector::FaceDetector(Detector method) {
    // TODO: Implement HaarCascades and remove.
    if (method == HaarCascade) {
        throw std::invalid_argument("Currently, we don't support HaarCascade detection");
    }

    method_ = method;
    changeMethod(method_);
}

void FaceDetector::changeMethod(Detector method) {
    // Load any files and initialize any data structures for the selected method.
    switch (method) {
    case OpenCV_DNN:
        // Read the trained neural net in the Caffe format from file
        net_ = cv::dnn::readNetFromCaffe(prototxt_file, caffe_model);
        break;
    case Dlib_68:
        // Initialze the dlib facial detector
        dlib_ = dlib::get_frontal_face_detector();
        // Load the trained cascade of trees from file.
        dlib::deserialize(dlib_68_file) >> dlib_sp_;
        break; 
    }
}

void FaceDetector::detectFace(cv::Mat &frame) {
    // Jump to the private detection function corresponding to the currently
    // selected detection method.
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
    // Generate a greyscaled version of the image
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Convert the opencv frame to a dlib format and run it through the cascade
    dlib::cv_image<dlib::rgb_pixel> dlib_frame(frame);
    std::vector<dlib::rectangle> faces = dlib_(dlib_frame);
    
    // Nothing to draw if we don't detect any faces
    if (faces.empty()) return;

    // Draw an OpenCV rectangle on the outermost boundaries of the landmarks
    int x    = faces[0].left();
    int y    = faces[0].top();
    int endX = x + faces[0].width();
    int endY = y + faces[0].height();
    cv::rectangle(frame, Point(x, y), Point(endX, endY), Scalar(0, 0, 255), 2);

    // We use our shape predictor to get all 68 landmark points from the face detector
    dlib::full_object_detection shape = dlib_sp_(dlib::cv_image<dlib::rgb_pixel>(frame), faces[0]);

    // Go through each of the landmarks, convert it to an OpenCV Point, and draw it on the frame.
    for(int i = 0; i < shape.num_parts(); ++i){
        Point p(shape.part(i).x(), shape.part(i).y());
        cv::circle(frame, p, 2.0, Scalar(255, 0, 0), 1, 8);
    }
}

void FaceDetector::_detectHAAR(cv::Mat &frame) {

}

