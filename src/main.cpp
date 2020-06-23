////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// EyeMouse: Webcam-Based Eye-Tracking Mouse Software.
// See README.md for installation and use details
//
//
//
//
//
//
//
//
//
//
//
//
//
// Copyright(c) Ryan Prendergast 2020. All Rights Reserved.
///////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FaceEyeDetector.h"
#include "camux/Face.h"

#include <iostream>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// Confidence threshold (>0, <1.0) for deciding if a feature is a face
const int MICROSECONDS_PER_SECOND = 1000000;
const int FPS = 30;

// Frame Index - Iterates each frame from 0 to $FPS-1. For timing granularity.
int f_idx = 0;
int img_height = 720;
int img_width = 1080;
double total_latency = 0;


/**
 * Run the GazeMouse software. Opens up a webcam, iterates through each frame, and runs
 * 	the implemented tracking softwares (face detector -> eye detector -> pupil detector ->
 * 	gaze detector). Times the above detectors to track latencies.
 *
 */
int main() {
		cv::VideoCapture cap(0);
		if (!cap.isOpened()) {
			std::cerr << "No webcam detected" << std::endl;
			return -1;
		}

		// Initialize the variables to pass to the face detector. Frame holds the image
		// data. Face is written by the face detector to hold the coordinates of the face,
		// among other properties e.g confidence.
		cv::Mat frame, leye_frame, reye_frame;
		camux::Face face;
		camux::Eye left_eye, right_eye;

		// Initialize the face/eye detector itself using any of the implemented methods.
		FaceEyeDetector face_eye_detector(Dlib_68, face, left_eye, right_eye);

		// Initialize the varibales to pass to the eye detector.

		// Timing latency variables
		std::chrono::steady_clock::time_point begin;
		std::chrono::steady_clock::time_point end;

		// Iterate through webcam frames until we receive escape
		while(1) {
			cap >> frame;

			// Timing latencies for debug
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

			face_eye_detector.detectFace(frame);

			// Timing for the face checking and eye detection process
			end = std::chrono::steady_clock::now();
			double frame_latency = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

			// Display the processing time for this frame
			cv::putText(frame, "Frame Latency: " + std::to_string((frame_latency) / MICROSECONDS_PER_SECOND),
										cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0));

			// Print average latency once per second, skipping the first second
			if (!f_idx && total_latency) {
				std::cout << "\033[1;31mAvg Frame Latency:\033[0m " << total_latency / MICROSECONDS_PER_SECOND / FPS << std::endl;
				total_latency = 0;
			}
			total_latency += frame_latency;

			f_idx = (f_idx + 1) % FPS;

			cv::Rect le = left_eye.getCoords();
			cv::Rect re = right_eye.getCoords();

			leye_frame = frame(le);
			reye_frame = frame(re);

			cv::resize(leye_frame, leye_frame, cv::Size(), 2, 2, cv::INTER_CUBIC);
			cv::resize(reye_frame, reye_frame, cv::Size(), 2, 2, cv::INTER_CUBIC);

			cv::cvtColor(leye_frame, leye_frame, cv::COLOR_BGR2GRAY);
			cv::cvtColor(reye_frame, reye_frame, cv::COLOR_BGR2GRAY);

			cv::equalizeHist(leye_frame, leye_frame);
			cv::equalizeHist(reye_frame, reye_frame);

			cv::imshow("Left eye", leye_frame);
			cv::imshow("Right eye", reye_frame);
			
			face_eye_detector.drawFace(frame);
			face_eye_detector.drawEyes(frame);

			cv::imshow("Webcam Display", frame);

			// Wait 30 ms between frames, and break if escape key is pressed
			if (cv::waitKey(1) == 27) break;
		}
		return 0;
}
