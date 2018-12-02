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


#ifndef TRACKING_H
#define TRACKING_H

#include <memory>

#include <opencv2/opencv.hpp>

#include "ORBVocabulary.h"
#include "Frame.h"

namespace ORB_SLAM2
{

class System;
class Map;
class KeyFrameDatabase;
class LocalMapping;
class LoopClosing;
class Viewer;
class KeyFrame;

struct TrackPoint
{
	CameraPose Tcr;
	const KeyFrame* referenceKF;
	double timestamp;
	bool lost;
	TrackPoint(const Frame& frame, bool lost);
};

using Trajectory = std::vector<TrackPoint>;

class Tracking
{

public:

	using Pointer = std::unique_ptr<Tracking>;

	// Tracking states
	enum State
	{
		STATE_NOT_READY = -1,
		STATE_NO_IMAGES = 0,
		STATE_NOT_INITIALIZED = 1,
		STATE_OK = 2,
		STATE_LOST = 3
	};

	struct Parameters
	{
		//New KeyFrame rules (according to fps)
		int minFrames;
		int maxFrames;

		// Threshold close/far points
		// Points seen as close by the stereo/RGBD sensor are considered reliable
		// and inserted from just one frame. Far points requiere a match in two keyframes.
		float thDepth;

		Parameters(int minFrames, int maxFrames, float thDepth);
	};

	static Pointer Create(System* system, ORBVocabulary* voc, Map* map, KeyFrameDatabase* keyframeDB,
		int sensor, const Parameters& param);

	virtual cv::Mat Update(Frame& currFrame) = 0;
	virtual void SetLocalMapper(LocalMapping* localMapper) = 0;
	virtual void SetLoopClosing(LoopClosing* loopClosing) = 0;

	// Use this function if you have deactivated local mapping and you only want to localize the camera.
	virtual void InformOnlyTracking(bool flag) = 0;

	virtual void Reset() = 0;

	virtual int GetState() const = 0;
	virtual int GetLastProcessedState() const = 0;

	virtual const Frame& GetInitialFrame() const = 0;
	virtual const std::vector<int>& GetIniMatches() const = 0;
	virtual const std::vector<int>& GetNumObservations() const = 0;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	virtual const Trajectory& GetTrajectory() const = 0;

	// True if local mapping is deactivated and we are performing only localization
	virtual bool OnlyTracking() const = 0;

	virtual ~Tracking();
};

} //namespace ORB_SLAM

#endif // TRACKING_H
