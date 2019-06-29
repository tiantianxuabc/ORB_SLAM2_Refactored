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
	KeyPoints initKeyPoints; // Initialization: KeyPoints in reference frame
	std::vector<int> matches; // Initialization: correspondeces with reference keypoints
	KeyPoints currKeyPoints; // KeyPoints in current frame
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
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

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

			const cv::Point2f& pt = currKeyPoints[i].pt;
			const cv::Point2f pt1(pt.x - r, pt.y - r);
			const cv::Point2f pt2(pt.x + r, pt.y + r);

			// This is a match to a MapPoint in the map
			if (status[i] == MAPPOINT_STATUS_MAP)
			{
				cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
				cv::circle(image, pt, 2, cv::Scalar(0, 255, 0), -1);
				ntracked_++;
			}
			// This is match to a "visual odometry" MapPoint created in the last frame
			else if (status[i] == MAPPOINT_STATUS_VO)
			{
				cv::rectangle(image, pt1, pt2, cv::Scalar(255, 0, 0));
				cv::circle(image, pt, 2, cv::Scalar(255, 0, 0), -1);
				ntrackedVO_++;
			}
		}
	}

	// Draw text info
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
	const std::string text = ss.str();
	const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);
	const int textH = textSize.height + 10;

	cv::Mat draw = cv::Mat::zeros(image.rows + textH, image.cols, image.type());
	image.copyTo(draw(cv::Rect(0, 0, image.cols, image.rows)));
	cv::putText(draw, text, cv::Point(5, draw.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);

	return draw;
}

void FrameDrawer::Update(const Tracking* tracker, const Frame& currFrame, const cv::Mat& image)
{
	std::unique_lock<std::mutex> lock(mutex_);
	image.copyTo(image_);
	currKeyPoints_ = currFrame.keypoints;
	
	const int nkeypoints = static_cast<int>(currKeyPoints_.size());
	status_.assign(nkeypoints, MAPPOINT_STATUS_NONE);
	localizationMode_ = tracker->OnlyTracking();

	const int state = tracker->GetLastProcessedState();
	if (state == Tracking::STATE_NOT_INITIALIZED)
	{
		initKeyPoints_ = tracker->GetInitialFrame().keypoints;
		initMatches_ = tracker->GetIniMatches();
	}
	else if (state == Tracking::STATE_OK)
	{
		const std::vector<int>& nobservations = tracker->GetNumObservations();
		CV_Assert(nobservations.size() == currKeyPoints_.size());

		for (int i = 0; i < nkeypoints; i++)
		{
			if (nobservations[i] < 0)
				continue;

			status_[i] = nobservations[i] > 0 ? MAPPOINT_STATUS_MAP : MAPPOINT_STATUS_VO;
		}
	}
	state_ = state;
}

} //namespace ORB_SLAM
