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
class FrameDrawer;
class MapDrawer;
class Map;
class KeyFrameDatabase;
class LocalMapping;
class LoopClosing;
class Viewer;
class KeyFrame;

class Tracking
{

public:

	// Tracking states
	enum eTrackingState {
		SYSTEM_NOT_READY = -1,
		NO_IMAGES_YET = 0,
		NOT_INITIALIZED = 1,
		OK = 2,
		LOST = 3
	};

	static std::shared_ptr<Tracking> Create(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
		KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

	// Preprocess the input and call Track(). Extract features and performs stereo matching.
	virtual cv::Mat GrabImageStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) = 0;
	virtual cv::Mat GrabImageRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) = 0;
	virtual cv::Mat GrabImageMonocular(const cv::Mat& image, double timestamp) = 0;

	virtual void SetLocalMapper(LocalMapping* pLocalMapper) = 0;
	virtual void SetLoopClosing(LoopClosing* pLoopClosing) = 0;
	virtual void SetViewer(Viewer* pViewer) = 0;

	// Load new settings
	// The focal lenght should be similar or scale prediction will fail when projecting points
	// TODO: Modify MapPoint::PredictScale to take into account focal lenght
	virtual void ChangeCalibration(const string &strSettingPath) = 0;

	// Use this function if you have deactivated local mapping and you only want to localize the camera.
	virtual void InformOnlyTracking(const bool &flag) = 0;

	virtual void Reset() = 0;

	virtual int GetState() const = 0;
	virtual int GetLastProcessedState() const = 0;

	virtual const Frame& GetCurrentFrame() const = 0;
	virtual const Frame& GetInitialFrame() const = 0;
	virtual cv::Mat GetImGray() const = 0;

	virtual const std::vector<int>& GetIniMatches() const = 0;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	virtual const list<cv::Mat>& GetRelativeFramePoses() const = 0;
	virtual const list<KeyFrame*>& GetReferences() const = 0;
	virtual const list<double>& GetFrameTimes() const = 0;
	virtual const list<bool>& GetLost() const = 0;

	// True if local mapping is deactivated and we are performing only localization
	virtual bool OnlyTracking() const = 0;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
