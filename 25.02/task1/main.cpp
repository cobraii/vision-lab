#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat image = imread("C:/Users/555/Desktop/vision-lab/25.02/input_image.jpg");
    if (image.empty()) {
        cout << "Error: Could not load input_image.jpg" << endl;
        return -1;
    }

    // Преобразование в различные цветовые пространства
    Mat rgb = image.clone(); // BGR в OpenCV
    Mat hsv, gray;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Обнаружение линий (HoughLinesP)
    Mat edges;
    Canny(gray, edges, 50, 150);
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 10);
    Mat lines_image = image.clone();
    for (const auto& l : lines) {
        line(lines_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 255), 2);
    }

    // Обнаружение кругов (HoughCircles)
    Mat circles_image = image.clone();
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 200, 100, 0, 0);
    for (const auto& c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        circle(circles_image, center, radius, Scalar(255, 0, 255), 2);
    }

    imshow("Original Image", image);
    imshow("RGB (BGR)", rgb);
    imshow("HSV", hsv);
    imshow("Grayscale", gray);
    imshow("Detected Lines", lines_image);
    imshow("Detected Circles", circles_image);

    imwrite("C:/Users/555/Desktop/vision-lab/25.02/task1/output/rgb_image.jpg", rgb);
    imwrite("C:/Users/555/Desktop/vision-lab/25.02/task1/output/hsv_image.jpg", hsv);
    imwrite("C:/Users/555/Desktop/vision-lab/25.02/task1/output/gray_image.jpg", gray);
    imwrite("C:/Users/555/Desktop/vision-lab/25.02/task1/output/lines_image.jpg", lines_image);
    imwrite("C:/Users/555/Desktop/vision-lab/25.02/task1/output/circles_image.jpg", circles_image);

    waitKey(0);
    destroyAllWindows();

    return 0;
}