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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <map>
#include <mutex>

#include <opencv2/core/core.hpp>

#include "FrameId.h"

namespace ORB_SLAM2
{

class Frame;
class KeyFrame;
class Map;

class MapPoint
{
public:

	using mappointid_t = long unsigned int;

	MapPoint(const cv::Mat& Xw, KeyFrame* referenceKF, Map* map);
	MapPoint(const cv::Mat& Xw, Map* map, Frame* frame, int idx);

	void SetWorldPos(const cv::Mat &Pos);
	cv::Mat GetWorldPos();

	cv::Mat GetNormal();
	KeyFrame* GetReferenceKeyFrame();

	std::map<KeyFrame*, size_t> GetObservations();
	int Observations();

	void AddObservation(KeyFrame* pKF, size_t idx);
	void EraseObservation(KeyFrame* pKF);

	int GetIndexInKeyFrame(KeyFrame* pKF);
	bool IsInKeyFrame(KeyFrame* pKF);

	void SetBadFlag();
	bool isBad();

	void Replace(MapPoint* pMP);
	MapPoint* GetReplaced();

	void IncreaseVisible(int n = 1);
	void IncreaseFound(int n = 1);
	float GetFoundRatio();
	inline int GetFound() {
		return nfound_;
	}

	void ComputeDistinctiveDescriptors();

	cv::Mat GetDescriptor();

	void UpdateNormalAndDepth();

	float GetMinDistanceInvariance();
	float GetMaxDistanceInvariance();
	int PredictScale(float currentDist, KeyFrame*pKF);
	int PredictScale(float currentDist, Frame* pF);

public:

	mappointid_t id;
	static mappointid_t nextId;
	int firstKFid;
	int firstFrame;
	
	// Variables used by the tracking
	float mTrackProjX;
	float mTrackProjY;
	float mTrackProjXR;
	bool mbTrackInView;
	int mnTrackScaleLevel;
	float mTrackViewCos;
	frameid_t trackReferenceForFrame;
	frameid_t lastFrameSeen;

	// Variables used by local mapping
	frameid_t BALocalForKF;
	frameid_t fuseCandidateForKF;

	// Variables used by loop closing
	frameid_t loopPointForKF;
	frameid_t correctedByKF;
	frameid_t correctedReference;
	cv::Mat mPosGBA;
	frameid_t BAGlobalForKF;


	static std::mutex globalMutex_;

protected:

	// Position in absolute coordinates
	cv::Mat Xw_;

	// Keyframes observing the point and associated index in keyframe
	std::map<KeyFrame*, size_t> observations_;
	int nobservations_;

	// Mean viewing direction
	cv::Mat normal_;

	// Best descriptor to fast matching
	cv::Mat descriptor_;

	// Reference KeyFrame
	KeyFrame* referenceKF_;

	// Tracking counters
	int nvisible_;
	int nfound_;

	// Bad flag (we do not currently erase MapPoint from memory)
	bool bad_;
	MapPoint* replaced_;

	// Scale invariance distances
	float minDistance_;
	float maxDistance_;

	Map* map_;

	std::mutex mutexPos_;
	std::mutex mutexFeatures_;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
