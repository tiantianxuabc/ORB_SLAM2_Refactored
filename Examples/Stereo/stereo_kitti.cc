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

static int LoadImages(const char* path, std::vector<std::string>& I0s, std::vector<std::string>& I1s,
	std::vector<double>& timestamps)
{
	std::ifstream ifs(FormatString("%s/times.txt", path));
	CV_Assert(!ifs.fail());

	double timestamp;
	while (ifs >> timestamp)
		timestamps.push_back(timestamp);

	const int nframes = static_cast<int>(timestamps.size());
	I0s.resize(nframes);
	I1s.resize(nframes);
	for (int i = 0; i < nframes; i++)
	{
		I0s[i] = FormatString("%s/image_0/%06d.png", path, i);
		I1s[i] = FormatString("%s/image_1/%06d.png", path, i);
	}
	return nframes;
}

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		std::cerr << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence [use_viewer]" << std::endl;
		return 1;
	}

	// Retrieve paths to images
	std::vector<std::string> ILs;
	std::vector<std::string> IRs;
	std::vector<double> timestamps;

	const int nframes = LoadImages(argv[3], ILs, IRs, timestamps);

	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	const bool useViewer = argc > 4 ? std::stoi(argv[4]) != 0 : true;
	auto SLAM = ORB_SLAM2::System::Create(argv[1], argv[2], ORB_SLAM2::System::STEREO, useViewer);

	// Vector for tracking time statistics
	std::vector<double> trackTimes;
	trackTimes.resize(nframes);

	std::cout << std::endl << "-------" << std::endl;
	std::cout << "Start processing sequence ..." << std::endl;
	std::cout << "Images in the sequence: " << nframes << std::endl << std::endl;

	// Main loop
	for (int i = 0; i < nframes; i++)
	{
		// Read left and right images from file
		const cv::Mat IL = cv::imread(ILs[i], cv::IMREAD_UNCHANGED);
		const cv::Mat IR = cv::imread(IRs[i], cv::IMREAD_UNCHANGED);
		const double timestamp = timestamps[i];

		if (IL.empty())
		{
			std::cerr << std::endl << "Failed to load image at: " << ILs[i] << std::endl;
			return 1;
		}

		const auto t1 = std::chrono::steady_clock::now();

		// Pass the images to the SLAM system
		SLAM->TrackStereo(IL, IR, timestamp);

		const auto t2 = std::chrono::steady_clock::now();

		const double T1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
		const double T2 = i < nframes - 1 ? timestamps[i + 1] - timestamp : timestamp - timestamps[i - 1];

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
	std::cout << "median tracking time: " << trackTimes[nframes / 2] << std::endl;
	std::cout << "mean tracking time: " << totalTime / nframes << std::endl;

	// Save camera trajectory
	SLAM->SaveTrajectoryKITTI("CameraTrajectory.txt");

	return 0;
}
