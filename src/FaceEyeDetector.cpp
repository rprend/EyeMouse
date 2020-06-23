#include "FaceEyeDetector.h"

#include <exception>

// NN model files for OpenCV_DNN
std::string caffe_model = "res10_300x300_ssd_iter_140000.caffemodel";
std::string prototxt_file = "deploy.prototxt";

// Cascade file for the dlib cascade.
std::string dlib_68_file = "shape_predictor_68_face_landmarks.dat"; 

void FaceEyeDetector::changeMethod(Detector method) {
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

void FaceEyeDetector::detectFace(cv::Mat &frame) {
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

void FaceEyeDetector::detectEyes(cv::Mat &frame) {
  
}

void FaceEyeDetector::_detectDNN(cv::Mat &frame) {
	// The dimensions (e.g 720x1080) of the image
	height_ = frame.size[0];
	width_ = frame.size[1];

	// From https://github.com/opencv/opencv/tree/master/samples/dnn:
	//   "To achieve the best accuracy run the model on BGR images resized
	//   to 300x300 applying mean subtraction of values (104, 177, 123) for
	//   each blue, green and red channels correspondingly."
	cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0,
						cv::Size(300, 300), cv::Scalar((104, 177, 123)));

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

            // Draw the bounding rectangle and save it with our confidence to our face object
            camux::drawRectangle(frame, x, y, endX, endY);
            face_.setCoords(x, y, endX, endY);
            face_.setConfidence(confidence);

			cv::putText(frame, std::to_string(confidence * 100) + "%", cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255));
        }
	}
}

void FaceEyeDetector::_detectDLIB(cv::Mat &frame) {
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

    // Draw the bounding rectangle and save it to our face object
    camux::drawRectangle(frame, x, y, endX, endY);
    face_.setCoords(x, y, endX, endY);

    // We use our shape predictor to get all 68 landmark points from the face detector
    dlib::full_object_detection shape = dlib_sp_(dlib::cv_image<dlib::rgb_pixel>(frame), faces[0]);
    camux::Points l_eye, r_eye;

    // Go through each of the landmarks, convert it to an OpenCV Point, and draw it on the frame.
    for(int i = 0; i < shape.num_parts(); ++i){
        cv::Point2u p(shape.part(i).x(), shape.part(i).y());

        // Left Eye landmarks
        if (i >= 36 && i <= 41) {
            cv::circle(frame, p, 2.0, cv::Scalar(0, 255, 255), 1, 8);
            l_eye.push_back(p);
        }
        // Right Eye landmarks 
        else if (i >= 42 && i <= 47) {
            cv::circle(frame, p, 2.0, cv::Scalar(255, 255, 0), 1, 8);
            r_eye.push_back(p);
        } else {
            cv::circle(frame, p, 2.0, cv::Scalar(255, 0, 0), 1, 8);
        }
    }

    left_.setCoords(camux::boundingRect(l_eye));
    right_.setCoords(camux::boundingRect(r_eye));

    camux::drawRectangle(frame, left_.getCoords());
    camux::drawRectangle(frame, right_.getCoords());
}

void FaceEyeDetector::_detectHAAR(cv::Mat &frame) {
}
