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


#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <memory>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM2
{

class MapPoint;

class System
{
public:

	enum Sensor { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

	using Pointer = std::unique_ptr<System>;
	using Path = std::string;

	// Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
	static Pointer Create(const Path& vocabularyFile, const Path& settingsFile, Sensor sensor, bool useViewer = true);

	// Proccess the given stereo frame. Images must be synchronized and rectified.
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	virtual cv::Mat TrackStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) = 0;

	// Process the given rgbd frame. Depthmap must be registered to the RGB frame.
	// Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Input depthmap: Float (CV_32F).
	// Returns the camera pose (empty if tracking fails).
	virtual cv::Mat TrackRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) = 0;

	// Proccess the given monocular frame
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	virtual cv::Mat TrackMonocular(const cv::Mat& image, double timestamp) = 0;

	// This stops local mapping thread (map building) and performs only camera tracking.
	virtual void ActivateLocalizationMode() = 0;
	// This resumes local mapping thread and performs SLAM again.
	virtual void DeactivateLocalizationMode() = 0;

	// Returns true if there have been a big map change (loop closure, global BA)
	// since last call to this function
	virtual bool MapChanged() const = 0;

	// Reset the system (clear map)
	virtual void Reset() = 0;

	// All threads will be requested to finish.
	// It waits until all threads have finished.
	// This function must be called before saving the trajectory.
	virtual void Shutdown() = 0;

	// Save camera trajectory in the TUM RGB-D dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	virtual void SaveTrajectoryTUM(const Path &filename) const = 0;

	// Save keyframe poses in the TUM RGB-D dataset format.
	// This method works for all sensor input.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	virtual void SaveKeyFrameTrajectoryTUM(const Path &filename) const = 0;

	// Save camera trajectory in the KITTI dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
	virtual void SaveTrajectoryKITTI(const Path &filename) const = 0;

	// TODO: Save/Load functions
	// SaveMap(const Path &filename);
	// LoadMap(const Path &filename);

	// Information from most recent processed frame
	// You can call this right after TrackMonocular (or stereo or RGBD)
	virtual int GetTrackingState() const = 0;
	virtual std::vector<MapPoint*> GetTrackedMapPoints() const = 0;
	virtual std::vector<cv::KeyPoint> GetTrackedKeyPointsUn() const = 0;
};

} // namespace ORB_SLAM

#endif // SYSTEM_H
