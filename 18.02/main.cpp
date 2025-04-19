#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Загрузка изображения
    Mat image = imread("C:/Users/555/Desktop/vision-lab/18.02/input_image.jpg");
    if (image.empty()) {
        cout << "Error: Could not load input_image.jpg" << endl;
        return -1;
    }

    // Запрос размера сетки
    int gridSize;
    cout << "Enter grid size (2 for 2x2, 3 for 3x3, 4 for 4x4): ";
    cin >> gridSize;
    if (gridSize != 2 && gridSize != 3 && gridSize != 4) {
        cout << "Error: Invalid grid size. Use 2, 3, or 4." << endl;
        return -1;
    }

    // Создание копии изображения для обработки
    Mat result = image.clone();

    // Размеры сегментов
    int segmentHeight = image.rows / gridSize;
    int segmentWidth = image.cols / gridSize;

    // Обработка каждого сегмента
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            // Вычисление индекса сегмента (нумерация слева направо, сверху вниз)
            int index = i * gridSize + j + 1;

            // Определение области сегмента
            Rect roi(j * segmentWidth, i * segmentHeight, segmentWidth, segmentHeight);
            Mat segment = result(roi);

            // Применение эффектов
            if (index % 4 == 0) {
                // Кратные 4: Заливка зелёным
                segment.setTo(Scalar(0, 255, 0));
            } else if (index % 3 == 0) {
                // Кратные 3: Оттенки серого
                Mat gray;
                cvtColor(segment, gray, COLOR_BGR2GRAY);
                cvtColor(gray, segment, COLOR_GRAY2BGR);
            } else if (index % 2 == 1) {
                // Нечётные: Инверсия цветов
                bitwise_not(segment, segment);
            }
        }
    }

    // Отображение оригинального и обработанного изображений
    imshow("Original Image", image);
    imshow("Processed Image", result);
    waitKey(0);

    // Сохранение результата
    imwrite("C:/Users/555/Desktop/vision-lab/18.02/output/result_image.jpg", result);

    // Освобождение ресурсов
    destroyAllWindows();

    return 0;
}