#include "Eye.h"

#include <opencv2/highgui.hpp>

cv::Point2u camux::Eye::findPupilCenter(cv::Mat& eye) {
    cv::Mat sobel_x, sobel_y;
    cv::Mat result;

    std::vector<cv::Vec3f> circles;

    if (eye.empty()) return center_;

    cv::cvtColor(eye, eye, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(eye, eye);

    cv::GaussianBlur(eye, eye, cv::Size(3,3), 0);
    cv::threshold(eye, result, 3, 255, 1);

    cv::dilate(result, result, cv::getStructuringElement(0, 
                                cv::Size(5, 5)));

    // cv::Sobel(eye, sobel_x, CV_32F, 1, 0);
    // cv::Sobel(eye, sobel_y, CV_32F, 0, 1);
    // cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, result);

    cv::imshow(std::to_string(type_) + " found areas", result);

    cv::HoughCircles(result, circles, cv::HOUGH_GRADIENT, 1, eye.rows/8, 16, 8, 0, 0 );

    if (circles.size() > 0) {
        center_ = cv::Point(cvRound(circles[0][0]), cvRound(circles[0][1]));
        pupil_radius_ = cvRound(circles[0][2]);
        // std::cout << circles.size() << " circles found!" << center_ << " r= " << pupil_radius_ << std::endl;

        // circle center
        cv::circle(eye, center_, 3, cv::Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        cv::circle(eye, center_, pupil_radius_, cv::Scalar(0,0,255), 3, 8, 0 );
    }

    return center_;
}