////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

// For debug
#include <iostream>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

using cv::Scalar;
using cv::Size;

// Confidence threshold (>0, <1.0) for deciding if a feature is a face
const float FACE_CONFIDENCE_THRESHOLD = 0.6;
const int MICROSECONDS_IN_SECOND = 1000000;
const int FPS = 30;

std::string caffe_model = "res10_300x300_ssd_iter_140000.caffemodel";
std::string prototxt_file = "deploy.prototxt";

// Frame Index - Iterates each frame from 0 to $FPS-1
int f_idx = 0;
int img_height = 720;
int img_width = 1080;
double total_latency = 0;


void find_faces(cv::Mat &frame, cv::Mat &faces) {
	for (int i = 0; i < faces.size[2]; ++i) {
		float confidence = faces.at<float>(i, 2);
		// std::cout << "Confidence: " << faces.at<float>(i, 2) << std::endl;

		if (confidence > FACE_CONFIDENCE_THRESHOLD) {
			// Extract the bounding rectangle and draw on the frame
			int x    = faces.at<float>(i, 3) * img_width;
			int y    = faces.at<float>(i, 4) * img_height;
			int endX = faces.at<float>(i, 5) * img_width;
			int endY = faces.at<float>(i, 6) * img_height;

			cv::rectangle(frame, cv::Point(x, y), cv::Point(endX, endY),
						Scalar(0, 0, 255), 2);

			cv::putText(frame, std::to_string(confidence * 100) + "%", cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255));
		}
	}

}

void parse_frame(cv::Mat &frame, cv::dnn::Net &net) {

	// The dimensions (e.g 720x1080) of the image
	img_height = frame.size[0];
	img_width = frame.size[1];

	// Timing latencies for debug
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

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
	net.setInput(blob);
	cv::Mat faces = net.forward();

	//Measure the time for a blobbing the image and a forward pass separately
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double forward_pass_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

	begin = end;
	// Iterate through all the potential faces, only draw ones with high confidence.
	find_faces(frame, faces);

	// Timing for the face checking and eye detection process
	end = std::chrono::steady_clock::now();
	double face_detect_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

	// Display the processing time for this frame
	cv::putText(frame, "Frame Latency: " + std::to_string((face_detect_us + forward_pass_us) / MICROSECONDS_IN_SECOND),
								cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 0, 0));

	// Print average latency once per second, skipping the first second
	if (!f_idx && total_latency) {
		std::cout << "\033[1;31mAvg Frame Latency:\033[0m " << total_latency / MICROSECONDS_IN_SECOND / FPS << std::endl;
		total_latency = 0;
	}
	total_latency += face_detect_us + forward_pass_us;


	f_idx = (f_idx + 1) % FPS;

}

/**
 *
 *
 *
 */
int main() {
		cv::VideoCapture cap(0);
		if (!cap.isOpened()) {
			std::cerr << "No webcam detected" << std::endl;
			return -1;
		}

		cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxt_file, caffe_model);
		cv::Mat frame;
		// Iterate through webcam frames until we receive escape
		while(1) {
			cap >> frame;
			parse_frame(frame, net);

			cv::imshow("Webcam Display", frame);

			// Wait 30 ms between frams, and break if escape key is pressed
			if (cv::waitKey(30) == 27) break;
		}
		return 0;
}
