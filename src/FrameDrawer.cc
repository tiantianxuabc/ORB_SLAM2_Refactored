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
	std::vector<bool> isVO, isMap; // Tracked MapPoints in current frame
	int state; // Tracking state

	//Copy variables within scoped mutex
	{
		unique_lock<mutex> lock(mMutex);
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
			isVO = isVO_;
			isMap = isMap_;
		}
		else if (state_ == Tracking::STATE_LOST)
		{
			currKeyPoints = currKeyPoints_;
		}
	} // destroy scoped mutex -> release mutex

	if (image.channels() < 3) //this should be always true
		cvtColor(image, image, CV_GRAY2BGR);

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
			if (isVO[i] || isMap[i])
			{
				cv::Point2f pt1, pt2;
				pt1.x = currKeyPoints[i].pt.x - r;
				pt1.y = currKeyPoints[i].pt.y - r;
				pt2.x = currKeyPoints[i].pt.x + r;
				pt2.y = currKeyPoints[i].pt.y + r;

				// This is a match to a MapPoint in the map
				if (isMap[i])
				{
					cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
					cv::circle(image, currKeyPoints[i].pt, 2, cv::Scalar(0, 255, 0), -1);
					ntracked_++;
				}
				else // This is match to a "visual odometry" MapPoint created in the last frame
				{
					cv::rectangle(image, pt1, pt2, cv::Scalar(255, 0, 0));
					cv::circle(image, currKeyPoints[i].pt, 2, cv::Scalar(255, 0, 0), -1);
					ntrackedVO_++;
				}
			}
		}
	}

	cv::Mat imageWithInfo;
	DrawTextInfo(image, state, imageWithInfo);

	return imageWithInfo;
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
	stringstream s;
	if (nState == Tracking::STATE_NO_IMAGES)
		s << " WAITING FOR IMAGES";
	else if (nState == Tracking::STATE_NOT_INITIALIZED)
		s << " TRYING TO INITIALIZE ";
	else if (nState == Tracking::STATE_OK)
	{
		if (!mbOnlyTracking)
			s << "SLAM MODE |  ";
		else
			s << "LOCALIZATION | ";
		int nKFs = map_->KeyFramesInMap();
		int nMPs = map_->MapPointsInMap();
		s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << ntracked_;
		if (ntrackedVO_ > 0)
			s << ", + VO matches: " << ntrackedVO_;
	}
	else if (nState == Tracking::STATE_LOST)
	{
		s << " TRACK LOST. TRYING TO RELOCALIZE ";
	}
	else if (nState == Tracking::STATE_NOT_READY)
	{
		s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
	}

	int baseline = 0;
	cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

	imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
	im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
	imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
	cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
	unique_lock<mutex> lock(mMutex);
	pTracker->GetImGray().copyTo(image_);
	currKeyPoints_ = pTracker->GetCurrentFrame().keypointsL;
	N = currKeyPoints_.size();
	isVO_ = vector<bool>(N, false);
	isMap_ = vector<bool>(N, false);
	mbOnlyTracking = pTracker->OnlyTracking();


	if (pTracker->GetLastProcessedState() == Tracking::STATE_NOT_INITIALIZED)
	{
		initKeyPoints_ = pTracker->GetInitialFrame().keypointsL;
		initMatches_ = pTracker->GetIniMatches();
	}
	else if (pTracker->GetLastProcessedState() == Tracking::STATE_OK)
	{
		for (int i = 0; i < N; i++)
		{
			MapPoint* pMP = pTracker->GetCurrentFrame().mappoints[i];
			if (pMP)
			{
				if (!pTracker->GetCurrentFrame().outlier[i])
				{
					if (pMP->Observations() > 0)
						isMap_[i] = true;
					else
						isVO_[i] = true;
				}
			}
		}
	}
	state_ = static_cast<int>(pTracker->GetLastProcessedState());
}

} //namespace ORB_SLAM
