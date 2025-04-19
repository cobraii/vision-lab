#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat image = imread("C:/Users/555/Desktop/vision-lab/25.02/input_image.jpg");
    if (image.empty()) {
        cout << "Error: Could not load input_image.jpg" << endl;
        return -1;
    }

    // Преобразование в HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Диапазоны для красного цвета
    Mat red_mask1, red_mask2, red_mask;
    Scalar red_lower1(0, 100, 100), red_upper1(10, 255, 255);
    Scalar red_lower2(170, 100, 100), red_upper2(180, 255, 255);
    inRange(hsv, red_lower1, red_upper1, red_mask1);
    inRange(hsv, red_lower2, red_upper2, red_mask2);
    red_mask = red_mask1 | red_mask2;

    // Диапазон для зелёного цвета
    Mat green_mask;
    Scalar green_lower(40, 100, 100), green_upper(80, 255, 255);
    inRange(hsv, green_lower, green_upper, green_mask);

    Mat red_result, green_result;
    bitwise_and(image, image, red_result, red_mask);
    bitwise_and(image, image, green_result, green_mask);

    imshow("Original Image", image);
    imshow("Red Mask", red_result);
    imshow("Green Mask", green_result);

    imwrite("C:/Users/555/Desktop/vision-lab/25.02/task2/output/red_mask.jpg", red_result);
    imwrite("C:/Users/555/Desktop/vision-lab/25.02/task2/output/green_mask.jpg", green_result);

    waitKey(0);
    destroyAllWindows();

    return 0;
}