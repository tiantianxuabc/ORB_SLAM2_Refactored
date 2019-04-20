/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Ra’Yl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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
#include <iomanip>
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

static int LoadImages(const char* I0Path, const char* I1Path, const char* timePath,
	std::vector<std::string>& I0s, std::vector<std::string>& I1s, std::vector<double>& stamps)
{
	std::ifstream ifs(timePath);
	CV_Assert(!ifs.fail());

	int nframes = 0;
	std::string line;
	while (std::getline(ifs, line))
	{
		I0s.push_back(FormatString("%s/%s.png", I0Path, line.c_str()));
		I1s.push_back(FormatString("%s/%s.png", I1Path, line.c_str()));
		stamps.push_back(1e-9 * std::stoull(line));
		nframes++;
	}
	return nframes;
}

class Rectify
{

public:

	enum class View { LEFT, RIGHT };

	Rectify(const cv::FileStorage& settings, View view) : fail_(true)
	{
		cv::Mat K, D, R, P;
		int h, w;
		if (view == View::LEFT)
		{
			settings["LEFT.K"] >> K;
			settings["LEFT.P"] >> P;
			settings["LEFT.R"] >> R;
			settings["LEFT.D"] >> D;
			h = settings["LEFT.height"];
			w = settings["LEFT.width"];
		}
		else if (view == View::RIGHT)
		{
			settings["RIGHT.K"] >> K;
			settings["RIGHT.P"] >> P;
			settings["RIGHT.R"] >> R;
			settings["RIGHT.D"] >> D;
			h = settings["RIGHT.height"];
			w = settings["RIGHT.width"];
		}

		if (K.empty() || D.empty() || R.empty() || P.empty() || h == 0 || w == 0)
			return;

		cv::initUndistortRectifyMap(K, D, R, P(cv::Rect(0, 0, 3, 3)), cv::Size(w, h), CV_32F, M1, M2);

		fail_ = false;
	}

	void operator()(const cv::Mat& src, cv::Mat& dst) const
	{
		cv::remap(src, dst, M1, M2, cv::INTER_LINEAR);
	}

	bool fail() const { return fail_; }

private:
	cv::Mat M1, M2;
	bool fail_;
};

int main(int argc, char* argv[])
{
	if (argc < 6)
	{
		std::cerr << "Usage: ./stereo_euroc path_to_vocabulary path_to_settings path_to_left_folder path_to_right_folder path_to_times_file" << std::endl;
		return 1;
	}

	// Load sequence
	std::vector<std::string> ILs, IRs;
	std::vector<double> timestamps;
	const int nimages = LoadImages(argv[3], argv[4], argv[5], ILs, IRs, timestamps);

	if (ILs.empty() || IRs.empty())
	{
		std::cerr << "ERROR: No images in provided path." << std::endl;
		return 1;
	}

	if (ILs.size() != IRs.size())
	{
		std::cerr << "ERROR: Different number of left and right images." << std::endl;
		return 1;
	}

	// Read rectification parameters
	cv::FileStorage settings(argv[2], cv::FileStorage::READ);
	if (!settings.isOpened())
	{
		std::cerr << "ERROR: Wrong path to settings" << std::endl;
		return 1;
	}

	Rectify rectifyL(settings, Rectify::View::LEFT);
	Rectify rectifyR(settings, Rectify::View::RIGHT);
	if (rectifyL.fail() || rectifyR.fail())
	{
		std::cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << std::endl;
		return 1;
	}

	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	auto SLAM = ORB_SLAM2::System::Create(argv[1], argv[2], ORB_SLAM2::System::STEREO, true);

	// Vector for tracking time statistics
	std::vector<float> trackTimes;
	trackTimes.resize(nimages);

	std::cout << std::endl << "-------" << std::endl;
	std::cout << "Start processing sequence ..." << std::endl;
	std::cout << "Images in the sequence: " << nimages << std::endl << std::endl;

	// Main loop
	cv::Mat IL, IR;
	for (int i = 0; i < nimages; i++)
	{
		const cv::Mat _IL = cv::imread(ILs[i], cv::IMREAD_UNCHANGED);
		const cv::Mat _IR = cv::imread(IRs[i], cv::IMREAD_UNCHANGED);
		if (_IL.empty() || _IR.empty())
		{
			std::cout << "imread failed." << std::endl;
			break;
		}

		rectifyL(_IL, IL);
		rectifyR(_IR, IR);

		const double timestamp = timestamps[i];

		const auto t1 = std::chrono::steady_clock::now();

		// Pass the images to the SLAM system
		SLAM->TrackStereo(IL, IR, timestamp);

		const auto t2 = std::chrono::steady_clock::now();

		// Wait to load the next frame
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

	return 0;
}
