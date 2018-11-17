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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "CameraParameters.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class ORBextractor;

struct ImageBounds
{
	float mnMinX;
	float mnMaxX;
	float mnMinY;
	float mnMaxY;
	float Width() const;
	float Height() const;
	bool Contains(float x, float y) const;
};

struct ScalePyramidInfo
{
	int mnScaleLevels;
	float mfScaleFactor;
	float mfLogScaleFactor;
	std::vector<float> mvScaleFactors;
	std::vector<float> mvInvScaleFactors;
	std::vector<float> mvLevelSigma2;
	std::vector<float> mvInvLevelSigma2;
};

struct CameraPose
{
	// Computes rotation, translation and camera center matrices from the camera pose.
	void Update();

	// Returns inverse of rotation
	cv::Mat GetRotationInverse() const;

	// Camera pose.
	cv::Mat mTcw;

	// Rotation, translation and camera center
	cv::Mat mRcw;
	cv::Mat mtcw;
	cv::Mat mRwc;
	cv::Mat mOw; //==mtwc
};

class FeaturesGrid
{

public:

	FeaturesGrid();
	FeaturesGrid(const std::vector<cv::KeyPoint>& keypoints, const ImageBounds& imageBounds, int nlevels);
	void AssignFeatures(const std::vector<cv::KeyPoint>& keypoints, const ImageBounds& imageBounds, int nlevels);
	std::vector<size_t> GetFeaturesInArea(float x, float y, float r, int minLevel = -1, int maxLevel = -1) const;

private:
	static const int ROWS = 48;
	static const int COLS = 64;
	float mfGridElementWidthInv;
	float mfGridElementHeightInv;
	std::vector<cv::KeyPoint> keypoints_;
	ImageBounds imageBounds_;
	int nlevels_;
	std::vector<std::size_t> mGrid[COLS][ROWS];
};

class Frame
{
public:
	Frame();

	// Copy constructor.
	Frame(const Frame &frame);

	// Constructor for stereo cameras.
	Frame(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp, ORBextractor* extractorL, ORBextractor* extractorR,
		ORBVocabulary* voc, const CameraParams& camera, const cv::Mat& distCoef, float thDepth);

	// Constructor for RGB-D cameras.
	Frame(const cv::Mat& image, const cv::Mat& depth, double timestamp, ORBextractor* extractor,
		ORBVocabulary* voc, const CameraParams& camera, const cv::Mat& distCoef, float thDepth);

	// Constructor for Monocular cameras.
	Frame(const cv::Mat& image, double timestamp, ORBextractor* extractor, ORBVocabulary* voc,
		const CameraParams& camera, const cv::Mat& distCoef, float thDepth);

	// Compute Bag of Words representation.
	void ComputeBoW();

	// Set the camera pose.
	void SetPose(cv::Mat Tcw);

	// Returns the camera center.
	cv::Mat GetCameraCenter() const;

	// Check if a MapPoint is in the frustum of the camera
	// and fill variables of the MapPoint to be used by the tracking
	bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

	std::vector<size_t> GetFeaturesInArea(float x, float y, float r, int minLevel = -1, int maxLevel = -1) const;

	// Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
	cv::Mat UnprojectStereo(const int &i);

public:
	// Vocabulary used for relocalization.
	ORBVocabulary* voc;

	// Frame timestamp.
	double timestamp;

	// Calibration matrix
	CameraParams camera;

	// Threshold close/far points. Close points are inserted from 1 view.
	// Far points are inserted as in the monocular case from 2 views.
	float thDepth;

	// Number of KeyPoints.
	int N;

	// Vector of keypoints (original for visualization) and undistorted (actually used by the system).
	// In the stereo case, mvKeysUn is redundant as images must be rectified.
	// In the RGB-D case, RGB images can be distorted.
	std::vector<cv::KeyPoint> keypointsL, keypointsR;
	std::vector<cv::KeyPoint> keypointsUn;

	// Corresponding stereo coordinate and depth for each keypoint.
	// "Monocular" keypoints have a negative value.
	std::vector<float> uright;
	std::vector<float> depth;

	// Bag of Words Vector structures.
	DBoW2::BowVector mBowVec;
	DBoW2::FeatureVector mFeatVec;

	// ORB descriptor, each row associated to a keypoint.
	cv::Mat descriptorsL, descriptorsR;

	// MapPoints associated to keypoints, NULL pointer if no association.
	std::vector<MapPoint*> mappoints;

	// Flag to identify outlier associations.
	std::vector<bool> outlier;

	// Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
	FeaturesGrid grid;

	// Camera pose.
	CameraPose pose;

	// Current and Next Frame id.
	static long unsigned int nNextId;
	long unsigned int mnId;

	// Reference Keyframe.
	KeyFrame* referenceKF;

	// Scale pyramid info.
	ScalePyramidInfo pyramid;

	// Undistorted Image Bounds (computed once).
	static ImageBounds imageBounds;
	static bool initialComputation;
};

}// namespace ORB_SLAM

#endif // FRAME_H
