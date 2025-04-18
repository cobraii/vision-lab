#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace fs = std::filesystem;

struct ImageSample {
	std::string label;
	cv::Mat image_data;
	std::vector<cv::KeyPoint> feature_points;
	cv::Mat feature_descriptors;
};

std::vector<cv::DMatch> filter_matches(const std::vector<std::vector<cv::DMatch>>& knn_matches, double ratio = 0.75) {
	std::vector<cv::DMatch> filtered_matches;

	for (const auto& match_set : knn_matches) {
		if (match_set.size() >= 2 && match_set[0].distance < ratio * match_set[1].distance) filtered_matches.push_back(match_set[0]);
	}

	return filtered_matches;
}

bool load_reference_images(const std::string& directory, std::vector<ImageSample>& samples, cv::Ptr<cv::SIFT> detector) {
	if (!fs::exists(directory)) {
		std::cerr << "Error: Directory '" << directory << "' not found." << std::endl;
		return false;
	}

	for (const auto& entry : fs::directory_iterator(directory)) {
		if (entry.path().extension() == ".png") {
			ImageSample sample;
			sample.label = entry.path().stem().string();
			sample.image_data = cv::imread(entry.path().string());

			if (sample.image_data.empty()) {
				std::cerr << "Warning: Could not load image: " << entry.path() << std::endl;
				continue;
			}

			detector->detectAndCompute(sample.image_data, cv::noArray(), sample.feature_points, sample.feature_descriptors);
			samples.push_back(sample);
		}
	}

	return !samples.empty();
}

void draw_detection(cv::Mat& output_image, const ImageSample& sample, const std::vector<cv::Point2f>& target_corners) {
	for (size_t i = 0; i < 4; ++i) {
		cv::line(
			output_image,
			target_corners[i],
			target_corners[(i + 1) % 4],
			cv::Scalar(0, 255, 0),
			2
		);
	}

	cv::Point2f center(0, 0);
	for (const auto& corner : target_corners) center += corner;
	
	center /= static_cast<float>(target_corners.size());

	cv::putText(
		output_image,
		sample.label,
		center - cv::Point2f(50, 0),
		cv::FONT_HERSHEY_DUPLEX,
		0.8,
		cv::Scalar(0, 0, 255),
		2
	);
}

int main() {
	const double MIN_AREA_THRESHOLD = 1000.0;
	const int MIN_MATCH_COUNT = 4;

	cv::Ptr<cv::SIFT> feature_detector = cv::SIFT::create();
	cv::BFMatcher feature_matcher(cv::NORM_L2);
	std::vector<ImageSample> reference_samples;

	std::string reference_dir = "./cards";
	if (!load_reference_images(reference_dir, reference_samples, feature_detector)) {
		std::cerr << "Error: No valid reference images loaded." << std::endl;
		return -1;
	}

	cv::Mat target_image = cv::imread("./image.png");
	if (target_image.empty()) {
		std::cerr << "Error: Could not load target image." << std::endl;
		return -1;
	}

	std::vector<cv::KeyPoint> target_keypoints;
	cv::Mat target_descriptors;
	feature_detector->detectAndCompute(target_image, cv::noArray(), target_keypoints, target_descriptors);

	std::cout << "Target image keypoints detected: " << target_keypoints.size() << std::endl;

	cv::Mat result_image = target_image.clone();
	bool found_match = false;

	for (const auto& sample : reference_samples) {
		if (sample.feature_descriptors.empty() || target_descriptors.empty()) {
			std::cout << "Skipping " << sample.label << ": No valid descriptors" << std::endl;
			continue;
		}

		std::vector<std::vector<cv::DMatch>> knn_matches;
		feature_matcher.knnMatch(sample.feature_descriptors, target_descriptors, knn_matches, 2);

		auto good_matches = filter_matches(knn_matches);
		std::cout << sample.label << " matches found: " << good_matches.size() << std::endl;

		if (good_matches.size() < MIN_MATCH_COUNT) {
			std::cout << sample.label << ": Insufficient matches for homography" << std::endl;
			continue;
		}

		std::vector<cv::Point2f> sample_points, target_points;
		for (const auto& match : good_matches) {
			sample_points.push_back(sample.feature_points[match.queryIdx].pt);
			target_points.push_back(target_keypoints[match.trainIdx].pt);
		}

		cv::Mat homography = cv::findHomography(sample_points, target_points, cv::RANSAC);
		if (homography.empty()) {
			std::cout << sample.label << ": Homography calculation failed" << std::endl;
			continue;
		}

		std::vector<cv::Point2f> sample_corners = {
			{0, 0},
			{static_cast<float>(sample.image_data.cols), 0},
			{static_cast<float>(sample.image_data.cols), static_cast<float>(sample.image_data.rows)},
			{0, static_cast<float>(sample.image_data.rows)}
		};
		std::vector<cv::Point2f> target_corners;
		cv::perspectiveTransform(sample_corners, target_corners, homography);

		double contour_area = cv::contourArea(target_corners);
		std::cout << sample.label << " contour area: " << contour_area << std::endl;

		if (contour_area < MIN_AREA_THRESHOLD) {
			std::cout << sample.label << ": Contour area too small" << std::endl;
			continue;
		}

		draw_detection(result_image, sample, target_corners);
		found_match = true;
	}

	if (!found_match) {
		std::cout << "No valid matches found for any reference sample." << std::endl;
	}

	cv::imshow("Detected Cards", result_image);
	cv::waitKey(0);
	return 0;
}