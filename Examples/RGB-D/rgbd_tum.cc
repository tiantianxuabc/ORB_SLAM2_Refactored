/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <System.h>

static inline void usleep(int64_t usec) { std::this_thread::sleep_for(std::chrono::microseconds(usec)); }

template <class... Args>
static std::string FormatString(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

static int LoadImages(const char* path, std::vector<std::string>& images, std::vector<std::string>& depths,
	std::vector<double>& timestamps)
{
	std::ifstream ifs(path);
	CV_Assert(!ifs.fail());

	int nframes = 0;
	std::string line, filename1, filename2;
	double stamp1, stamp2;
	while (std::getline(ifs, line))
	{
		std::stringstream ss(line);
		ss >> stamp1 >> filename1 >> stamp2 >> filename2;
		images.push_back(filename1);
		depths.push_back(filename2);
		timestamps.push_back(stamp1);
		nframes++;
	}
	return nframes;
}

int main(int argc, char **argv)
{
	if (argc < 5)
	{
		std::cerr << std::endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association [use_viewer]" << std::endl;
		return 1;
	}

	// Retrieve paths to images
	std::vector<std::string> images;
	std::vector<std::string> depths;
	std::vector<double> timestamps;
	const int nimages = LoadImages(argv[4], images, depths, timestamps);

	// Check consistency in the number of images and depthmaps
	if (images.empty())
	{
		std::cerr << std::endl << "No images found in provided path." << std::endl;
		return 1;
	}
	else if (depths.size() != images.size())
	{
		std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
		return 1;
	}

	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	auto SLAM = ORB_SLAM2::System::Create(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);

	// Vector for tracking time statistics
	std::vector<double> trackTimes;
	trackTimes.resize(nimages);

	std::cout << std::endl << "-------" << std::endl;
	std::cout << "Start processing sequence ..." << std::endl;
	std::cout << "Images in the sequence: " << nimages << std::endl << std::endl;

	// Main loop
	cv::Mat imRGB, imD;
	for (int i = 0; i < nimages; i++)
	{
		// Read image and depthmap from file
		const cv::Mat image = cv::imread(FormatString("%s/%s", argv[3], images[i].c_str()), cv::IMREAD_UNCHANGED);
		const cv::Mat depth = cv::imread(FormatString("%s/%s", argv[3], depths[i].c_str()), cv::IMREAD_UNCHANGED);
		const double timestamp = timestamps[i];

		if (image.empty())
		{
			std::cerr << std::endl << "Failed to load image at: " << std::string(argv[3]) << "/" << images[i] << std::endl;
			return 1;
		}

		const auto t1 = std::chrono::steady_clock::now();

		// Pass the image to the SLAM system
		SLAM->TrackRGBD(image, depth, timestamp);

		const auto t2 = std::chrono::steady_clock::now();

		const double T1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
		const double T2 = i < nimages - 1 ? timestamps[i + 1] - timestamp : timestamp - timestamps[i - 1];

		trackTimes[i] = T1;

		// Wait to load the next frame
		if (T1 < T2)
			usleep(static_cast<int64_t>(1e6 * (T2 - T1)));
	}

	// Stop all threads
	SLAM->Shutdown();

	// Tracking time statistics
	std::sort(std::begin(trackTimes), std::end(trackTimes));
	const double totalTime = std::accumulate(std::begin(trackTimes), std::end(trackTimes), 0.);

	std::cout << "-------" << std::endl << std::endl;
	std::cout << "median tracking time: " << trackTimes[nimages / 2] << std::endl;
	std::cout << "mean tracking time: " << totalTime / nimages << std::endl;

	// Save camera trajectory
	SLAM->SaveTrajectoryTUM("CameraTrajectory.txt");
	SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

	return 0;
}
