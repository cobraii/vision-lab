#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

// Функция для увеличения насыщенности изображения
Mat enhanceSaturation(const Mat& input) {
    Mat hsv;
    cvtColor(input, hsv, COLOR_BGR2HSV);
    vector<Mat> channels;
    split(hsv, channels);
    channels[1] *= 1.5;
    merge(channels, hsv);
    Mat output;
    cvtColor(hsv, output, COLOR_HSV2BGR);
    return output;
}

// Функция для классификации фигур по контурам
string classifyShape(const vector<Point>& contour) {
    // Аппроксимация контура
    vector<Point> approx;
    double peri = arcLength(contour, true);
    approxPolyDP(contour, approx, 0.04 * peri, true);
    int vertices = approx.size();

    if (vertices == 3) {
        return "Triangle";
    }
    else if (vertices == 4) {
        Rect bounding = boundingRect(contour);
        float aspectRatio = (float)bounding.width / bounding.height;
        if (abs(aspectRatio - 1.0) < 0.1) {
            return "Square";
        }
        else {
            return "Rectangle";
        }
    }
    else if (vertices >= 5 && vertices <= 8) {
        double area = contourArea(contour);
        double perimeter = arcLength(contour, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity > 0.85) {
            return "Circle";
        }
        else {
            return "Polygon";
        }
    }
    else if (vertices > 8) {
        return "Circle";
    }
    return "Unknown";
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }
    Mat image = imread(argv[1]);
    if (image.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    Mat saturated = enhanceSaturation(image);

    Mat gray;
    cvtColor(saturated, gray, COLOR_BGR2GRAY);

    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    Mat edges;
    Canny(blurred, edges, 50, 150);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result = image.clone();

    // Классификация и визуализация фигур
    for (size_t i = 0; i < contours.size(); i++) {
        // Пропуск маленьких контуров
        if (contourArea(contours[i]) < 100) continue;

        string shape = classifyShape(contours[i]);

        Scalar color(0, 255, 0); // Зеленый цвет
        drawContours(result, contours, (int)i, color, 2);

        Moments M = moments(contours[i]);
        int cX = int(M.m10 / M.m00);
        int cY = int(M.m01 / M.m00);

        putText(result, shape, Point(cX - 50, cY), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
    }

    imshow("Original Image", image);
    imshow("Edges", edges);
    imshow("Detected Shapes", result);
    waitKey(0);

    imwrite("result_shapes.jpg", result);

    return 0;
}