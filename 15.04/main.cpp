#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    string input_path = "C:/Users/555/Desktop/vision-lab/15.04/input.jpg";
    cout << "Loading image: " << input_path << endl;
    Mat image = imread(input_path);
    if (image.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }
    cout << "Image loaded: " << image.cols << "x" << image.rows << endl;

    CascadeClassifier face_cascade, eye_cascade, smile_cascade;
    string face_cascade_path = "C:/Users/555/Desktop/vision-lab/15.04/haarcascade_frontalface_default.xml";
    string eye_cascade_path = "C:/Users/555/Desktop/vision-lab/15.04/haarcascade_eye.xml";
    string smile_cascade_path = "C:/Users/555/Desktop/vision-lab/15.04/haarcascade_smile.xml";

    if (!face_cascade.load(face_cascade_path)) {
        cout << "Error: Could not load face cascade" << endl;
        return -1;
    }
    if (!eye_cascade.load(eye_cascade_path)) {
        cout << "Error: Could not load eye cascade" << endl;
        return -1;
    }
    if (!smile_cascade.load(smile_cascade_path)) {
        cout << "Error: Could not load smile cascade" << endl;
        return -1;
    }
    cout << "Cascades loaded successfully" << endl;

    // Преобразование в Grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Детекция лиц
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.3, 5);

    // Обработка каждого лица
    for (const auto& face : faces) {
        // Отрисовка прямоугольника вокруг лица (зелёный)
        rectangle(image, face, Scalar(0, 255, 0), 2);

        // Область лица для глаз и улыбки
        Mat faceROI = gray(face);
        Mat faceROI_color = image(face);

        // Детекция глаз
        vector<Rect> eyes;
        eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 3);
        for (const auto& eye : eyes) {
            // Отрисовка прямоугольника вокруг глаза (синий)
            rectangle(image, Point(face.x + eye.x, face.y + eye.y),
                      Point(face.x + eye.x + eye.width, face.y + eye.y + eye.height),
                      Scalar(255, 0, 0), 2);
        }

        // Детекция улыбки
        vector<Rect> smiles;
        smile_cascade.detectMultiScale(faceROI, smiles, 1.8, 20);
        for (const auto& smile : smiles) {
            // Отрисовка прямоугольника вокруг улыбки (красный)
            rectangle(image, Point(face.x + smile.x, face.y + smile.y),
                      Point(face.x + smile.x + smile.width, face.y + smile.y + smile.height),
                      Scalar(0, 0, 255), 2);
        }
    }

    string output_path = "C:/Users/555/Desktop/vision-lab/15.04/lab15_04_detected.jpg";
    if (imwrite(output_path, image)) {
        cout << "Saved: " << output_path << endl;
    } else {
        cout << "Error: Failed to save " << output_path << endl;
    }

    imshow("Face, Eye, and Smile Detection", image);
    waitKey(0);
    destroyAllWindows();

    return 0;
}