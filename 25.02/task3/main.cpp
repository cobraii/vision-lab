#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>
#include <filesystem> // Для создания папки

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string output_dir = "C:/Users/555/Desktop/vision-lab/25.02/task3/output";
    filesystem::create_directories(output_dir);
    cout << "Output directory ensured: " << output_dir << endl;

    string image_path = "C:/Users/555/Desktop/vision-lab/25.02/i.jpg";
    cout << "Loading image: " << image_path << endl;
    Mat image = imread(image_path);
    if (image.empty()) {
        cout << "Error: Could not load i.jpg" << endl;
        return -1;
    }
    cout << "Image loaded successfully (" << image.cols << "x" << image.rows << ")" << endl;

    // Преобразование в HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Диапазоны для красного цвета
    Mat red_mask1, red_mask2, red_mask;
    Scalar red_lower1(0, 80, 80), red_upper1(20, 255, 255);
    Scalar red_lower2(160, 80, 80), red_upper2(180, 255, 255);
    inRange(hsv, red_lower1, red_upper1, red_mask1);
    inRange(hsv, red_lower2, red_upper2, red_mask2);
    red_mask = red_mask1 | red_mask2;

    // Применение маски
    Mat red_result;
    bitwise_and(image, image, red_result, red_mask);

    // Преобразование в Grayscale и размытие
    Mat red_gray, blurred;
    cvtColor(red_result, red_gray, COLOR_BGR2GRAY);
    GaussianBlur(red_gray, blurred, Size(9, 9), 0);
    cout << "Grayscale and blur applied" << endl;

    // Обнаружение кругов
    Mat circles_image = image.clone();
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, blurred.rows / 16, 100, 30, 10, 150);
    cout << "Found " << circles.size() << " circles" << endl;
    for (const auto& c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        circle(circles_image, center, radius, Scalar(255, 0, 0), 2);
        cout << "Circle at (" << center.x << ", " << center.y << "), radius: " << radius << endl;
    }

    imshow("Original Image", image);
    imshow("Red Mask (Binary)", red_mask);
    imshow("Red Mask Applied", red_result);
    imshow("Red Grayscale", red_gray);
    imshow("Blurred", blurred);
    imshow("Detected Circles", circles_image);

    string red_mask_path = output_dir + "/red_mask.jpg";
    string circles_path = output_dir + "/circles.jpg";
    if (imwrite(red_mask_path, red_result)) {
        cout << "Saved: " << red_mask_path << endl;
    } else {
        cout << "Error: Failed to save " << red_mask_path << endl;
    }
    if (imwrite(circles_path, circles_image)) {
        cout << "Saved: " << circles_path << endl;
    } else {
        cout << "Error: Failed to save " << circles_path << endl;
    }

    waitKey(0);
    destroyAllWindows();

    return 0;
}