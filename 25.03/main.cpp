#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

// Функция для обрезки чёрных областей
Mat cropBlackBorders(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat mask;
    threshold(gray, mask, 1, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Rect bbox = boundingRect(contours[0]);
    for (const auto& cnt : contours) {
        bbox |= boundingRect(cnt);
    }
    return img(Range(bbox.y, bbox.y + bbox.height), Range(bbox.x, bbox.x + bbox.width)).clone();
}

int main() {
    string left_path = "C:/Users/555/Desktop/vision-lab/25.03/left.jpg";
    string right_path = "C:/Users/555/Desktop/vision-lab/25.03/right.jpg";
    cout << "Loading images: " << left_path << ", " << right_path << endl;
    Mat img1 = imread(left_path);
    Mat img2 = imread(right_path);
    if (img1.empty() || img2.empty()) {
        cout << "Error: Could not load images" << endl;
        return -1;
    }
    cout << "Images loaded: left.jpg (" << img1.cols << "x" << img1.rows << "), right.jpg (" << img2.cols << "x" << img2.rows << ")" << endl;

    // Преобразование в Grayscale
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Инициализация ORB-детектора
    Ptr<ORB> orb = ORB::create(1000);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);
    cout << "Keypoints found: " << keypoints1.size() << " (left), " << keypoints2.size() << " (right)" << endl;

    Mat keypoints_img1, keypoints_img2;
    drawKeypoints(img1, keypoints1, keypoints_img1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img2, keypoints2, keypoints_img2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    if (imwrite("C:/Users/555/Desktop/vision-lab/25.03/lab25_03_left_points.jpg", keypoints_img1)) {
        cout << "Saved: lab25_03_left_points.jpg" << endl;
    }
    if (imwrite("C:/Users/555/Desktop/vision-lab/25.03/lab25_03_right_points.jpg", keypoints_img2)) {
        cout << "Saved: lab25_03_right_points.jpg" << endl;
    }

    // Brute-Force Matching с NORM_HAMMING
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Фильтрация совпадений с использованием Lowe's ratio test
    vector<DMatch> good_matches;
    const float ratio_thresh = 0.75f;
    for (const auto& m : knn_matches) {
        if (m[0].distance < ratio_thresh * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }
    cout << "Good matches: " << good_matches.size() << endl;

    // Сохранение визуализации совпадений
    Mat matches_img;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, matches_img, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    if (imwrite("C:/Users/555/Desktop/vision-lab/25.03/lab25_03_matches.jpg", matches_img)) {
        cout << "Saved: lab25_03_matches.jpg" << endl;
    }

    if (good_matches.size() < 4) {
        cout << "Error: Not enough good matches for homography" << endl;
        return -1;
    }

    vector<Point2f> points1, points2;
    for (const auto& m : good_matches) {
        points1.push_back(keypoints1[m.queryIdx].pt);
        points2.push_back(keypoints2[m.trainIdx].pt);
    }

    Mat H = findHomography(points2, points1, RANSAC, 3.0);
    if (H.empty()) {
        cout << "Error: Homography estimation failed" << endl;
        return -1;
    }

    vector<Point2f> projected_points;
    perspectiveTransform(points2, projected_points, H);
    Mat homography_img = img1.clone();
    for (size_t i = 0; i < points1.size(); ++i) {
        Point2f p1 = points1[i];
        Point2f p2 = projected_points[i];
        circle(homography_img, p1, 5, Scalar(0, 255, 0), 2);
        circle(homography_img, p2, 5, Scalar(0, 0, 255), 2);
        line(homography_img, p1, p2, Scalar(255, 0, 0), 1);
    }
    if (imwrite("C:/Users/555/Desktop/vision-lab/25.03/lab25_03_homography.jpg", homography_img)) {
        cout << "Saved: lab25_03_homography.jpg" << endl;
    }

    int w1 = img1.cols, h1 = img1.rows;
    int w2 = img2.cols, h2 = img2.rows;
    vector<Point2f> corners2(4);
    corners2[0] = Point2f(0, 0);
    corners2[1] = Point2f(w2, 0);
    corners2[2] = Point2f(w2, h2);
    corners2[3] = Point2f(0, h2);
    vector<Point2f> corners1(4);
    perspectiveTransform(corners2, corners1, H);

    float min_x = 0, max_x = w1;
    float min_y = 0, max_y = h1;
    for (const auto& p : corners1) {
        min_x = min(min_x, p.x);
        max_x = max(max_x, p.x);
        min_y = min(min_y, p.y);
        max_y = max(max_y, p.y);
    }
    int panorama_width = cvRound(max_x - min_x);
    int panorama_height = cvRound(max_y - min_y);
    Point2f offset(-min_x, -min_y);

    Mat panorama(Size(panorama_width, panorama_height), img1.type(), Scalar(0, 0, 0));
    Mat translation = (Mat_<double>(3, 3) << 1, 0, offset.x, 0, 1, offset.y, 0, 0, 1);
    warpPerspective(img2, panorama, translation * H, panorama.size());
    Mat roi(panorama, Rect(cvRound(offset.x), cvRound(offset.y), w1, h1));
    img1.copyTo(roi);

    Mat cropped_panorama = cropBlackBorders(panorama);
    cout << "Panorama cropped to: " << cropped_panorama.cols << "x" << cropped_panorama.rows << endl;

    imshow("Panorama", cropped_panorama);

    string output_path = "C:/Users/555/Desktop/vision-lab/25.03/lab25_03_panorama.jpg";
    if (imwrite(output_path, cropped_panorama)) {
        cout << "Saved: " << output_path << endl;
    } else {
        cout << "Error: Failed to save " << output_path << endl;
    }

    waitKey(0);
    destroyAllWindows();

    return 0;
}