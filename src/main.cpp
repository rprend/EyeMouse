////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FaceDetector.h"

#include <iostream>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using cv::Scalar;
using cv::Size;
using cv::Point;

// Confidence threshold (>0, <1.0) for deciding if a feature is a face
const int MICROSECONDS_IN_SECOND = 1000000;
const int FPS = 30;

// Frame Index - Iterates each frame from 0 to $FPS-1
int f_idx = 0;
int img_height = 720;
int img_width = 1080;
double total_latency = 0;


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

		cv::Mat frame;
		FaceDetector detector;

		// Timing latency variables
		std::chrono::steady_clock::time_point begin;
		std::chrono::steady_clock::time_point end;

		// Iterate through webcam frames until we receive escape
		while(1) {
			cap >> frame;

			// Timing latencies for debug
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

			detector.detectFace(frame);

			// Timing for the face checking and eye detection process
			end = std::chrono::steady_clock::now();
			double frame_latency = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

			// Display the processing time for this frame
			cv::putText(frame, "Frame Latency: " + std::to_string((frame_latency) / MICROSECONDS_IN_SECOND),
										cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 0, 0));

			// Print average latency once per second, skipping the first second
			if (!f_idx && total_latency) {
				std::cout << "\033[1;31mAvg Frame Latency:\033[0m " << total_latency / MICROSECONDS_IN_SECOND / FPS << std::endl;
				total_latency = 0;
			}
			total_latency += frame_latency;

			f_idx = (f_idx + 1) % FPS;

			cv::imshow("Webcam Display", frame);

			// Wait 30 ms between frames, and break if escape key is pressed
			if (cv::waitKey(30) == 27) break;
		}
		return 0;
}
