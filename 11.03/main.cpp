#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

// Структура для хранения информации об объекте
struct TrackedObject {
    Point2f centroid;
    string shape;
    int id;
};

// Функция классификации форм
string classifyShape(const vector<Point>& contour) {
    vector<Point> approx;
    double peri = arcLength(contour, true);
    approxPolyDP(contour, approx, 0.04 * peri, true);
    int vertices = approx.size();

    if (vertices == 3) return "Triangle";
    if (vertices == 4) {
        Rect bounding = boundingRect(contour);
        float aspectRatio = (float)bounding.width / bounding.height;
        return (abs(aspectRatio - 1.0) < 0.1) ? "Square" : "Rectangle";
    }
    if (vertices >= 5) {
        double area = contourArea(contour);
        double perimeter = arcLength(contour, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        return (circularity > 0.85) ? "Circle" : "Polygon";
    }
    return "Unknown";
}

// Функция сопоставления объектов между кадрами
void matchObjects(const vector<TrackedObject>& prevObjects, vector<TrackedObject>& currObjects) {
    for (auto& curr : currObjects) {
        float minDist = FLT_MAX;
        int matchedId = -1;
        for (const auto& prev : prevObjects) {
            float dist = norm(curr.centroid - prev.centroid);
            if (dist < minDist && dist < 50) { // Порог расстояния
                minDist = dist;
                matchedId = prev.id;
            }
        }
        curr.id = (matchedId != -1) ? matchedId : currObjects.size();
    }
}

int main(int argc, char** argv) {
    // Открытие видео
    VideoCapture cap("C:/Users/555/Desktop/vision-lab/11.03/test_shapes_video.avi");
    if (!cap.isOpened()) {
        cout << "Error: Could not open video" << endl;
        return -1;
    }

    // Параметры видео
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    // Создание объекта для записи видео
    VideoWriter out("C:/Users/555/Desktop/vision-lab/11.03/output/result_video.mp4", 
                    VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frameWidth, frameHeight));

    if (!out.isOpened()) {
        cout << "Error: Could not create output video" << endl;
        return -1;
    }

    vector<TrackedObject> prevObjects;
    int objectCounter = 0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Преобразование в градации серого
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Размытие и детекция границ
        Mat blurred;
        GaussianBlur(gray, blurred, Size(5, 5), 0);
        Mat edges;
        Canny(blurred, edges, 50, 150);

        // Поиск контуров
        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<TrackedObject> currObjects;

        // Обработка контуров
        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) < 100) continue; // Пропуск мелких контуров

            string shape = classifyShape(contours[i]);
            Moments M = moments(contours[i]);
            Point2f centroid(M.m10 / M.m00, M.m01 / M.m00);

            TrackedObject obj;
            obj.centroid = centroid;
            obj.shape = shape;
            obj.id = -1;
            currObjects.push_back(obj);

            // Визуализация контура и формы
            drawContours(frame, contours, (int)i, Scalar(0, 255, 0), 2);
            putText(frame, shape + " ID:" + to_string(obj.id), centroid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        }

        // Сопоставление объектов
        if (!prevObjects.empty()) {
            matchObjects(prevObjects, currObjects);
        } else {
            for (size_t i = 0; i < currObjects.size(); ++i) {
                currObjects[i].id = objectCounter++;
            }
        }

        // Визуализация трекинга
        for (const auto& obj : currObjects) {
            putText(frame, obj.shape + " ID:" + to_string(obj.id), obj.centroid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            circle(frame, obj.centroid, 3, Scalar(255, 0, 0), -1);
        }

        // Сохранение кадра в видео
        out.write(frame);

        // Показ кадра
        imshow("Tracking", frame);
        if (waitKey(30) == 27) break; // Выход по ESC

        prevObjects = currObjects;
    }

    // Освобождение ресурсов
    cap.release();
    out.release();
    destroyAllWindows();

    return 0;
}