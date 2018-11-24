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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace ORB_SLAM2
{

class ORBmatcher
{
public:

	ORBmatcher(float nnratio = 0.6, bool checkOri = true);

	// Computes the Hamming distance between two ORB descriptors
	static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

	// Search matches between Frame keypoints and projected MapPoints. Returns number of matches
	// Used to track the local map (Tracking)
	int SearchByProjection(Frame& frame, const std::vector<MapPoint*>& mappoints, float th = 3);

	// Project MapPoints tracked in last frame into the current frame and search matches.
	// Used to track from previous frame (Tracking)
	int SearchByProjection(Frame& currFrame, const Frame& lastFrame, float th, bool monocular);

	// Project MapPoints seen in KeyFrame into the Frame and search matches.
	// Used in relocalisation (Tracking)
	int SearchByProjection(Frame& currFrame, KeyFrame* keyframe, const std::set<MapPoint*>& alreadyFound,
		float th, int ORBdist);

	// Project MapPoints using a Similarity Transformation and search matches.
	// Used in loop detection (Loop Closing)
	int SearchByProjection(const KeyFrame* keyframe, const cv::Mat& Scw, const std::vector<MapPoint*>& mappoints,
		std::vector<MapPoint*>& matched, int th);

	// Search matches between MapPoints in a KeyFrame and ORB in a Frame.
	// Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
	// Used in Relocalisation and Loop Detection
	int SearchByBoW(KeyFrame* keyframe, Frame& frame, std::vector<MapPoint*>& matches);
	int SearchByBoW(KeyFrame* keyframe1, KeyFrame* keyframe2, std::vector<MapPoint*>& matches12);

	// Matching for the Map Initialization (only used in the monocular case)
	int SearchForInitialization(Frame& frame1, Frame& frame2, std::vector<cv::Point2f>& prevMatched, std::vector<int>& matches12, int windowSize = 10);

	// Matching to triangulate new MapPoints. Check Epipolar Constraint.
	int SearchForTriangulation(const KeyFrame* keyframe1, const KeyFrame* keyframe2, const cv::Mat& F12,
		std::vector<std::pair<size_t, size_t>>& matchIds, bool onlyStereo);

	// Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
	// In the stereo and RGB-D case, s12=1
	int SearchBySim3(KeyFrame* keyframe1, KeyFrame* keyframe2, std::vector<MapPoint*>& matches12, float s12, const cv::Mat &R12, const cv::Mat &t12, float th);

	// Project MapPoints into KeyFrame and search for duplicated MapPoints.
	int Fuse(KeyFrame* keyframe, const std::vector<MapPoint*>& mappoints, float th = 3.f);

	// Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
	int Fuse(KeyFrame* keyframe, const cv::Mat& Scw, const std::vector<MapPoint*>& mappoints,
		float th, std::vector<MapPoint*>& replacePoints);

public:

	static const int TH_LOW;
	static const int TH_HIGH;

private:

	float fNNRatio_;
	bool checkOrientation_;
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
