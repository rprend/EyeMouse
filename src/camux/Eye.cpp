#include "Eye.h"

#include <opencv2/highgui.hpp>

// For pupil isolation. The pupil boundaries will have a relatively large gradient. We threshold out
// any gradients too small, and we define too small as a multiple of the mean gradient. This parameter defines
// the constant of proportionality. The higher it is, the more pixels we threshold out (meaning we check fewer) for
// being the pupil. This decreases runtime dramatically. But, too high and you risk filtering out the pupil and
// increasing your false detection rate. Right now it is defined statically, in the future it should start out very
// low and be updated/learned to increase over the course of the calibration process.
const double STRONG_GRADIENT_THRESHOLD = 2.5;
const double DARK_PIXEL_THRESHOLD = .8;

cv::Point2u camux::Eye::findPupilCenter(cv::Mat& eye) {
    cv::Mat sobel_x, sobel_y;
    cv::Mat result;

    std::vector<cv::Vec3f> circles;

    if (eye.empty()) return center_;

    cv::Point2u center = _gradientIntersectionIsolation(eye); 

    return center_;

    cv::cvtColor(eye, eye, cv::COLOR_BGR2GRAY);        
    cv::equalizeHist(eye, eye);

    // TODO: Get eye histogram to better select the pupil from the image
    cv::Mat histogram;
    // cv::calcHist(eye, )
    // cv::calcHist(&eye, 1, 0, cv::Mat(), histogram, 256, {0, 256}); 

    // Experiment on different blurring methods
    cv::Size blur_size(3, 3);
    cv::Mat eye_homogeneous_blur, eye_gaussian, eye_median, eye_bilateral;
    // cv::blur(eye, eye_homogeneous_blur, blur_size);
    // cv::GaussianBlur(eye, eye_gaussian, blur_size, 0);
    cv::medianBlur(eye, eye_median, 3);
    // cv::bilateralFilter(eye, eye_bilateral, ) // TODO: Figure out parameters

    // cv::imshow("Homogeneous Blur", eye_homogeneous_blur);
    // cv::imshow("Gaussian Blur", eye_gaussian);
    cv::imshow("Median Blur", eye_median);
    // cv::imshow("Bilateral filter", eye_bilateral);

    // Threshold the eye image to only select the darker parts of the image. TODO:
    // Change this to an adaptive threshold so we don't just select black parts of the image. Or 
    // does equalizeHist() sufficiently blacken the parts of the image that are of interest?
    // Afterwards, dilate the thresholded parts of the image to enusre the pupil area contains any bright
    // reflections that may be the center. TODO: Explore other morphological ops - maybe open to get rid of noise
    // then dilate again?
    // Experiment: Thresholding level (1 vs 3 vs 5). Result: Threshold of 3 works best: 1 sometimes eliminates
    // the whole pupil, and 5 has too much extraneous noise.
    cv::Mat threshold_3, threshold_5, threshold_1;
    cv::threshold(eye_median, threshold_1, 1, 255, 1);
    cv::threshold(eye_median, threshold_3, 3, 255, 1);
    cv::threshold(eye_median, threshold_5, 5, 255, 1);
    cv::imshow("Median threshold 1", threshold_1);
    cv::imshow("Median threshold 3", threshold_3);
    cv::imshow("Median threshold 5", threshold_5);

    // Experiment: Active thresholding (make sure to disable equalizing histogram)
    // Conclusion: Adaptive thresholding finds the contours/boundaries of the eyes well. It does not work when
    // the eye is too small--which from the webcam is anytime you're not super close to the camera. We'll skip for now
    // but if you have the camera on the wearable, come back to this!
    // cv::Mat adaptive_threshold_C3, adaptive_threshold_C5;
    // cv::adaptiveThreshold(eye_median, adaptive_threshold_C3, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 3);
    // cv::adaptiveThreshold(eye_median, adaptive_threshold_C5, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 5);

    // cv::imshow("C=3 Adaptive", adaptive_threshold_C3);
    // cv::imshow("C=5 Adaptive", adaptive_threshold_C5);

    cv::dilate(threshold_3, result, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::imshow("Dilation", result);

    // Approximates the image gradients. Useful for the pupil isolation approach of maximizing the dot 
    // product of an image location->gradient vector with that of the unit gradient vector 
    // cv::Sobel(eye, sobel_x, CV_32F, 1, 0);
    // cv::Sobel(eye, sobel_y, CV_32F, 0, 1);
    // cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, result);

    // cv::imshow(std::to_string(type_) + " found areas", result);

    // Select circular parts of the image. Note this only really works if there is only one prominent circle
    // and that is the pupil!
    // cv::HoughCircles(result, circles, cv::HOUGH_GRADIENT, 1, eye.rows/8, 16, 8, 0, 0 );

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

cv::Point2u camux::Eye::_blurThresholdDilateIsolation(cv::Mat & eye) {
    return center_;
}

cv::Point2u camux::Eye::_gradientIntersectionIsolation(cv::Mat & eye) {
    // 1. Grayscale image. Calculate the Sobel gradients of the grayscale image in the x and y 
    //      direction. Get the total gradient magnitudes. Find the mean of the magnitude squared
    //      of the total gradients. Choose a threshold as some proportion of that mean 
    //      (try sqrt(.6) from Optimeyes). Get normalized gradients in the x and y direction
    cv::Mat gray, sobel_x, sobel_y, sobel_magnitude;
    cv::Mat abs_sobel_x, abs_sobel_y, abs_sobel_magnitude;

    cv::cvtColor(eye, gray, cv::COLOR_BGR2GRAY);

    cv::Sobel(gray, sobel_x, CV_32F, 1, 0);
    cv::Sobel(gray, sobel_y, CV_32F, 0, 1);
    
    cv::magnitude(sobel_x, sobel_y, sobel_magnitude);

    // Convert to 8 bit to display. CV_32 if a float value; if you try to display it will cast any
    // binary number equivalently above 255 in unsigned 8 bit to white.
    cv::convertScaleAbs(sobel_x, abs_sobel_x);
    cv::convertScaleAbs(sobel_y, abs_sobel_y);
    cv::convertScaleAbs(sobel_magnitude, abs_sobel_magnitude);

    // cv::normalize(abs_sobel_x, abs_sobel_x, 1, cv::NORM_L2);
    // cv::normalize(abs_sobel_y, abs_sobel_y, 1, cv::NORM_L2);

    cv::imshow("X sobel", abs_sobel_x);
    cv::imshow("Y Sobel", abs_sobel_y);
    cv::imshow("Sobel magnitude", abs_sobel_magnitude);

    // 2. Create a boolean 2d array that can index into the image (same width and height). An index
    //      in the bool area is true iff the gradient at the corresponding image index is greater than
    //      the threshold we defined. This step severly decreases computational complexity later. We only 
    //      care about strong gradients because these are close to borders of dark areas. 
    //
    //    Remember we observe that the pupil center is the point at which all of the gradient
    //      of the pupil edge intersect. 
    //
    //        \  **  /
    //         y    x
    //       *  \  /  *    The gradients of the border at the ellipse point towards the direction of greatest change,
    //      *    \/    *    which at the change from dark to light is "outwards". See how the gradients at x and y
    //      *    /\    *    point outwards, but if you extend them both as lines they intersect at the center?
    //       *  /  \  *     That's true generally of the relationship between points on the outside of the ellipse
    //         *    *       and the center.
    //           **
    cv::Mat grads_to_use, abs_grads_to_use;
    cv::Mat grad_X, grad_Y;

    // I tried adaptive thresholding here; it was slower and had no better / slightly worse ability to always
    // show the pupil than the "dumb" thresholding.
    cv::Scalar magnitude_threshold = cv::mean(sobel_magnitude) * STRONG_GRADIENT_THRESHOLD;     
    cv::threshold(sobel_magnitude, grads_to_use, magnitude_threshold[0], 255, cv::THRESH_TOZERO);
    
    cv::convertScaleAbs(grads_to_use, abs_grads_to_use);
    cv::imshow("Gradients to check", abs_grads_to_use);

    cv::divide(sobel_x, grads_to_use, grad_X);
    cv::divide(sobel_x, grads_to_use, grad_Y);

    // 3. Perform a binary threshold on the greyscale image as a certain percentage of the mean. Dilate this to
    //      remove bright reflections in the dark ellipse.
    cv::Mat dark_eye;

    cv::Scalar dark_threshold = cv::mean(gray) * DARK_PIXEL_THRESHOLD;
    cv::threshold(gray, dark_eye, dark_threshold[0], 255, cv::THRESH_BINARY);
    cv::dilate(dark_eye, dark_eye, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))); 

    cv::imshow("Dark parts of eye", dark_eye);

    // 4. Get a list of the coordinates of the gradients to use

    // 5. 

    return center_;
}

double camux::Eye::_estimateCenterProbabilityHist() {
 
    return 10;
}

