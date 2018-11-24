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

#include "FrameDrawer.h"

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"

namespace ORB_SLAM2
{

enum
{
	MAPPOINT_STATUS_NONE,
	MAPPOINT_STATUS_MAP,
	MAPPOINT_STATUS_VO,
};

FrameDrawer::FrameDrawer(Map* map) : map_(map)
{
	state_ = Tracking::STATE_NOT_READY;
	image_ = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat FrameDrawer::DrawFrame()
{
	cv::Mat image;
	std::vector<cv::KeyPoint> initKeyPoints; // Initialization: KeyPoints in reference frame
	std::vector<int> matches; // Initialization: correspondeces with reference keypoints
	std::vector<cv::KeyPoint> currKeyPoints; // KeyPoints in current frame
	std::vector<int> status; // Tracked MapPoints in current frame
	int state; // Tracking state

	//Copy variables within scoped mutex
	{
		std::unique_lock<std::mutex> lock(mutex_);
		state = state_;
		if (state_ == Tracking::STATE_NOT_READY)
			state_ = Tracking::STATE_NO_IMAGES;

		image_.copyTo(image);

		if (state_ == Tracking::STATE_NOT_INITIALIZED)
		{
			currKeyPoints = currKeyPoints_;
			initKeyPoints = initKeyPoints_;
			matches = initMatches_;
		}
		else if (state_ == Tracking::STATE_OK)
		{
			currKeyPoints = currKeyPoints_;
			status = status_;
		}
		else if (state_ == Tracking::STATE_LOST)
		{
			currKeyPoints = currKeyPoints_;
		}
	} // destroy scoped mutex -> release mutex

	if (image.channels() < 3) //this should be always true
		cv::cvtColor(image, image, CV_GRAY2BGR);

	//Draw
	if (state == Tracking::STATE_NOT_INITIALIZED) //INITIALIZING
	{
		for (size_t i = 0; i < matches.size(); i++)
		{
			const int idx = matches[i];
			if (idx >= 0)
				cv::line(image, initKeyPoints[i].pt, currKeyPoints[idx].pt, cv::Scalar(0, 255, 0));
		}
	}
	else if (state == Tracking::STATE_OK) //TRACKING
	{
		const float r = 5;

		ntracked_ = 0;
		ntrackedVO_ = 0;

		for (size_t i = 0; i < currKeyPoints.size(); i++)
		{
			if (status[i] == MAPPOINT_STATUS_NONE)
				continue;

			cv::Point2f pt1, pt2;
			pt1.x = currKeyPoints[i].pt.x - r;
			pt1.y = currKeyPoints[i].pt.y - r;
			pt2.x = currKeyPoints[i].pt.x + r;
			pt2.y = currKeyPoints[i].pt.y + r;

			// This is a match to a MapPoint in the map
			if (status[i] == MAPPOINT_STATUS_MAP)
			{
				cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
				cv::circle(image, currKeyPoints[i].pt, 2, cv::Scalar(0, 255, 0), -1);
				ntracked_++;
			}
			// This is match to a "visual odometry" MapPoint created in the last frame
			else if (status[i] == MAPPOINT_STATUS_VO)
			{
				cv::rectangle(image, pt1, pt2, cv::Scalar(255, 0, 0));
				cv::circle(image, currKeyPoints[i].pt, 2, cv::Scalar(255, 0, 0), -1);
				ntrackedVO_++;
			}
		}
	}

	cv::Mat imageWithInfo;
	DrawTextInfo(image, state, imageWithInfo);

	return imageWithInfo;
}


void FrameDrawer::DrawTextInfo(cv::Mat& src, int state, cv::Mat& dst)
{
	std::stringstream ss;
	if (state == Tracking::STATE_NO_IMAGES)
	{
		ss << " WAITING FOR IMAGES";
	}
	else if (state == Tracking::STATE_NOT_INITIALIZED)
	{
		ss << " TRYING TO INITIALIZE ";
	}
	else if (state == Tracking::STATE_OK)
	{
		ss << (localizationMode_ ? "LOCALIZATION | " : "SLAM MODE |  ");

		const size_t nkeyframes = map_->KeyFramesInMap();
		const size_t nmappoints = map_->MapPointsInMap();

		ss << "KFs: " << nkeyframes << ", MPs: " << nmappoints << ", Matches: " << ntracked_;
		if (ntrackedVO_ > 0)
			ss << ", + VO matches: " << ntrackedVO_;
	}
	else if (state == Tracking::STATE_LOST)
	{
		ss << " TRACK LOST. TRYING TO RELOCALIZE ";
	}
	else if (state == Tracking::STATE_NOT_READY)
	{
		ss << " LOADING ORB VOCABULARY. PLEASE WAIT...";
	}

	int baseline = 0;
	const cv::Size textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

	dst = cv::Mat(src.rows + textSize.height + 10, src.cols, src.type());
	src.copyTo(dst.rowRange(0, src.rows).colRange(0, src.cols));
	dst.rowRange(src.rows, dst.rows) = cv::Mat::zeros(textSize.height + 10, src.cols, src.type());
	cv::putText(dst, ss.str(), cv::Point(5, dst.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

void FrameDrawer::Update(Tracking* tracker)
{
	std::unique_lock<std::mutex> lock(mutex_);

	tracker->GetImGray().copyTo(image_);
	currKeyPoints_ = tracker->GetCurrentFrame().keypointsL;
	const Frame& currFrame = tracker->GetCurrentFrame();

	const int nkeypoints = static_cast<int>(currKeyPoints_.size());
	status_.assign(nkeypoints, MAPPOINT_STATUS_NONE);
	localizationMode_ = tracker->OnlyTracking();

	const int state = tracker->GetLastProcessedState();
	if (state == Tracking::STATE_NOT_INITIALIZED)
	{
		initKeyPoints_ = tracker->GetInitialFrame().keypointsL;
		initMatches_ = tracker->GetIniMatches();
	}
	else if (state == Tracking::STATE_OK)
	{
		for (int i = 0; i < nkeypoints; i++)
		{
			const MapPoint* mappoint = currFrame.mappoints[i];
			if (!mappoint || currFrame.outlier[i])
				continue;
			status_[i] = mappoint->Observations() > 0 ? MAPPOINT_STATUS_MAP : MAPPOINT_STATUS_VO;
		}
	}
	state_ = state;
}

} //namespace ORB_SLAM
