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

int low_H = 98, low_S = 43, low_V = 0;
int high_H = 119, high_S = 255, high_V = 156;
int max_H = 179, max_SV = 255;

std::string webcam_window = "Webcam Display";

static void on_low_H_thresh_trackbar(int, void *) {
    low_H = std::min(high_H-1, low_H);
    cv::setTrackbarPos("Low H", webcam_window, low_H);
}

static void on_high_H_thresh_trackbar(int, void *) {
    high_H = std::max(high_H, low_H+1);
    cv::setTrackbarPos("High H", webcam_window, high_H);
}

static void on_low_S_thresh_trackbar(int, void *) {
    low_S = std::min(high_S-1, low_S);
    cv::setTrackbarPos("Low S", webcam_window, low_S);
}

static void on_high_S_thresh_trackbar(int, void *) {
    high_S = std::max(high_S, low_S+1);
    cv::setTrackbarPos("High S", webcam_window, high_S);
}

static void on_low_V_thresh_trackbar(int, void *) {
    low_V = std::min(high_V-1, low_V);
    cv::setTrackbarPos("Low V", webcam_window, low_V);
}

static void on_high_V_thresh_trackbar(int, void *) {
    high_V = std::max(high_V, low_V+1);
    cv::setTrackbarPos("High V", webcam_window, high_V);
}

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
		cv::Mat frame, leye_frame, reye_frame, hsv, hsv_out, face_frame;
		camux::Face face;
		camux::Eye left_eye, right_eye;

		// Initialize the face/eye detector itself using any of the implemented methods.
		FaceEyeDetector face_eye_detector(HaarCascade, face, left_eye, right_eye);

		// Initialize the varibales to pass to the eye detector.

		// Timing latency variables
		std::chrono::steady_clock::time_point begin;
		std::chrono::steady_clock::time_point end;

		cv::namedWindow(webcam_window);

		cv::createTrackbar("Low H", webcam_window, &low_H, max_H, on_low_H_thresh_trackbar);
		cv::createTrackbar("High H", webcam_window, &high_H, max_H, on_high_H_thresh_trackbar);
		cv::createTrackbar("Low S", webcam_window, &low_S, max_SV, on_low_S_thresh_trackbar);
		cv::createTrackbar("High S", webcam_window, &high_S, max_SV, on_high_S_thresh_trackbar);
		cv::createTrackbar("Low V", webcam_window, &low_V, max_SV, on_low_V_thresh_trackbar);
		cv::createTrackbar("High V", webcam_window, &high_V, max_SV, on_high_V_thresh_trackbar);

		// Iterate through webcam frames until we receive escape
		while(1) {
			cap >> frame;
			
			// cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

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

			cv::Rect face_rect = face_eye_detector.getFace().getCoords();
			face_frame = frame(face_rect);

			if (!face_frame.empty()) {
				// Identify the blue on the image (for forehead dot feature)
				cv::cvtColor(face_frame, hsv, cv::COLOR_BGR2HSV);
				// cv::InputArray lower_pink = []; 
				
				// cv::GaussianBlur(hsv, hsv, cv::Size(3, 3), 0);
				cv::inRange(hsv, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), hsv_out);
				// cv::GaussianBlur(hsv_out, hsv_out, cv::Size(3, 3), 0);
			
				cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
				cv::morphologyEx(hsv_out, hsv_out, cv::MORPH_OPEN, kernel);
				
				std::vector<std::vector<cv::Point>> contours;
				cv::findContours(hsv_out, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				cv::drawContours(face_frame, contours, -1, cv::Scalar(225,0,0), 1);

				if (contours.size() > 0) {
					cv::Rect blue_dot_rect = cv::boundingRect(contours[0]);
					camux::drawRectangle(face_frame, blue_dot_rect);
				}

				cv::imshow("Selected parts of the image", hsv_out);
				cv::imshow("Blue circle", face_frame);

				// cv::cvtColor(leye_frame, leye_frame, cv::COLOR_BGR2GRAY);
				// cv::cvtColor(reye_frame, reye_frame, cv::COLOR_BGR2GRAY);

				// cv::equalizeHist(leye_frame, leye_frame);
				// cv::equalizeHist(reye_frame, reye_frame);

				cv::Point left_eye_center = left_eye.findPupilCenter(leye_frame);
				cv::Point right_eye_center = right_eye.findPupilCenter(reye_frame);
		
				left_eye_center.x += left_eye.getCoords().x;
				left_eye_center.y += left_eye.getCoords().y;

				cv::circle(frame, left_eye_center, 3, cv::Scalar(0,255,0), -1);
				// cv::circle(frame, left_eye_center, left_eye.getPupilRadius(), cv::Scalar(0,0,255));

				right_eye_center.x += right_eye.getCoords().x;
				right_eye_center.y += right_eye.getCoords().y;

				cv::circle(frame, right_eye_center, 3, cv::Scalar(0,255,0), -1);
				// cv::circle(frame, right_eye_center, right_eye.getPupilRadius(), cv::Scalar(0,0,255));

				cv::resize(leye_frame, leye_frame, cv::Size(), 2, 2, cv::INTER_CUBIC);
				cv::resize(reye_frame, reye_frame, cv::Size(), 2, 2, cv::INTER_CUBIC);

				cv::imshow("Left eye", leye_frame);
				cv::imshow("Right eye", reye_frame);
				
				face_eye_detector.drawFace(frame);
				face_eye_detector.drawEyes(frame);
				face_eye_detector.drawLandmarks(frame);
			}

			cv::imshow(webcam_window, frame);

			// Wait 30 ms between frames, and break if escape key is pressed
			if (cv::waitKey(1) == 27) break;
		}
		return 0;
}
