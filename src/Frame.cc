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

#include "Frame.h"

#include "ORBmatcher.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"

#include <thread>
#include <functional>

namespace ORB_SLAM2
{

frameid_t Frame::nextId = 0;
bool Frame::initialComputation = true;
ImageBounds Frame::imageBounds;

static inline int Round(float v) { return static_cast<int>(std::round(v)); }
static inline int RoundUp(float v) { return static_cast<int>(std::ceil(v)); }
static inline int RoundDn(float v) { return static_cast<int>(std::floor(v)); }

void GetScalePyramidInfo(const ORBextractor* extractor, ScalePyramidInfo& pyramid)
{
	pyramid.nlevels = extractor->GetLevels();
	pyramid.scaleFactor = extractor->GetScaleFactor();
	pyramid.logScaleFactor = log(pyramid.scaleFactor);
	pyramid.scaleFactors = extractor->GetScaleFactors();
	pyramid.invScaleFactors = extractor->GetInverseScaleFactors();
	pyramid.sigmaSq = extractor->GetScaleSigmaSquares();
	pyramid.invSigmaSq = extractor->GetInverseScaleSigmaSquares();
}

// Undistort keypoints given OpenCV distortion parameters.
// Only for the RGB-D case. Stereo must be already rectified!
// (called in the constructor).
void UndistortKeyPoints(const KeyPoints& src, KeyPoints& dst, const cv::Mat& K, const cv::Mat1f& distCoeffs)
{
	if (distCoeffs(0) == 0.f)
	{
		dst = src;
		return;
	}

	std::vector<cv::Point2f> points(src.size());
	for (size_t i = 0; i < src.size(); i++)
		points[i] = src[i].pt;

	cv::undistortPoints(points, points, K, distCoeffs, cv::Mat(), K);

	dst.resize(src.size());
	for (size_t i = 0; i < src.size(); i++)
	{
		cv::KeyPoint keypoint = src[i];
		keypoint.pt = points[i];
		dst[i] = keypoint;
	}
}

// Computes image bounds for the undistorted image (called in the constructor).
ImageBounds ComputeImageBounds(const cv::Mat& image, const cv::Mat& K, const cv::Mat1f& distCoeffs)
{
	const float h = static_cast<float>(image.rows);
	const float w = static_cast<float>(image.cols);

	if (distCoeffs(0) == 0.f)
		return ImageBounds(0.f, w, 0.f, h);

	std::vector<cv::Point2f> corners = { { 0, 0 }, { w, 0 }, { 0, h }, { w, h } };
	cv::undistortPoints(corners, corners, K, distCoeffs, cv::Mat(), K);

	ImageBounds imageBounds;
	imageBounds.minx = std::min(corners[0].x, corners[2].x);
	imageBounds.maxx = std::max(corners[1].x, corners[3].x);
	imageBounds.miny = std::min(corners[0].y, corners[1].y);
	imageBounds.maxy = std::max(corners[2].y, corners[3].y);
	return imageBounds;
}

// Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
void ComputeStereoFromRGBD(const KeyPoints& keypoints, const KeyPoints& keypointsUn, const cv::Mat& depthImage,
	const CameraParams& camera, std::vector<float>& uright, std::vector<float>& depth)
{
	const int nkeypoints = static_cast<int>(keypoints.size());

	uright.assign(nkeypoints, -1.f);
	depth.assign(nkeypoints, -1.f);

	for (int i = 0; i < nkeypoints; i++)
	{
		const cv::KeyPoint& keypoint = keypoints[i];
		const cv::KeyPoint& keypointUn = keypointsUn[i];

		const int v = static_cast<int>(keypoint.pt.y);
		const int u = static_cast<int>(keypoint.pt.x);
		const float d = depthImage.at<float>(v, u);
		if (d > 0)
		{
			depth[i] = d;
			uright[i] = keypointUn.pt.x - camera.bf / d;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////
// ImageBounds Class
//////////////////////////////////////////////////////////////////////////////////
ImageBounds::ImageBounds(float minx, float maxx, float miny, float maxy) : minx(minx), maxx(maxx), miny(miny), maxy(maxy) {}

float ImageBounds::Width() const
{
	return maxx - minx;
}

float ImageBounds::Height() const
{
	return maxy - miny;
}

bool ImageBounds::Contains(float x, float y) const
{
	return x >= minx && x < maxx && y >= miny && y < maxy;
}

bool ImageBounds::Empty() const
{
	return Width() <= 0.f || Height() <= 0.f;
}

//////////////////////////////////////////////////////////////////////////////////
// FeaturesGrid Class
//////////////////////////////////////////////////////////////////////////////////
FeaturesGrid::FeaturesGrid() {}

FeaturesGrid::FeaturesGrid(const std::vector<cv::KeyPoint>& keypoints, const ImageBounds& imageBounds, int nlevels)
{
	AssignFeatures(keypoints, imageBounds, nlevels);
}

void FeaturesGrid::AssignFeatures(const std::vector<cv::KeyPoint>& keypoints, const ImageBounds& imageBounds, int nlevels)
{
	invW_ = COLS / imageBounds.Width();
	invH_ = ROWS / imageBounds.Height();

	keypoints_ = keypoints;
	imageBounds_ = imageBounds;
	nlevels_ = nlevels;

	const int nkeypoints = static_cast<int>(keypoints.size());

	const int nreserve = nkeypoints / (COLS * ROWS) / 2;
	for (unsigned int i = 0; i < COLS; i++)
		for (unsigned int j = 0; j < ROWS; j++)
			grid_[i][j].reserve(nreserve);

	for (int i = 0; i < nkeypoints; i++)
	{
		const cv::KeyPoint& keypoint = keypoints[i];

		const int cx = Round(invW_ * (keypoint.pt.x - imageBounds.minx));
		const int cy = Round(invH_ * (keypoint.pt.y - imageBounds.miny));

		// Keypoint's coordinates are undistorted, which could cause to go out of the image
		if (cx < 0 || cx >= COLS || cy < 0 || cy >= ROWS)
			continue;

		grid_[cx][cy].push_back(i);
	}
}

std::vector<size_t> FeaturesGrid::GetFeaturesInArea(float x, float y, float r, int minLevel, int maxLevel) const
{
	const int nkeypoints = static_cast<int>(keypoints_.size());

	std::vector<size_t> indices;
	indices.reserve(nkeypoints);

	const float minx = imageBounds_.minx;
	const float miny = imageBounds_.miny;

	const int mincx = std::max(RoundDn(invW_ * (x - r - minx)), 0);
	const int maxcx = std::min(RoundUp(invW_ * (x + r - minx)), COLS - 1);
	const int mincy = std::max(RoundDn(invH_ * (y - r - miny)), 0);
	const int maxcy = std::min(RoundUp(invH_ * (y + r - miny)), ROWS - 1);

	if (mincx >= COLS || maxcx < 0 || mincy >= ROWS || maxcy < 0)
		return indices;

	const bool checkLevels = (minLevel > 0) || (maxLevel >= 0);
	if (maxLevel < 0)
		maxLevel = nlevels_;

	for (int cx = mincx; cx <= maxcx; cx++)
	{
		for (int cy = mincy; cy <= maxcy; cy++)
		{
			for (size_t idx : grid_[cx][cy])
			{
				const cv::KeyPoint& keypoint = keypoints_[idx];
				const int level = keypoint.octave;
				if (checkLevels && (level < minLevel || level > maxLevel))
					continue;

				const float distx = keypoint.pt.x - x;
				const float disty = keypoint.pt.y - y;

				if (fabsf(distx) < r && fabsf(disty) < r)
					indices.push_back(idx);
			}
		}
	}

	return indices;
}

Frame::Frame() {}

//Copy Constructor
Frame::Frame(const Frame& frame)
	: voc(frame.voc), timestamp(frame.timestamp), camera(frame.camera), thDepth(frame.thDepth), N(frame.N),
	keypointsL(frame.keypointsL), keypointsR(frame.keypointsR), keypointsUn(frame.keypointsUn),
	uright(frame.uright), depth(frame.depth), bowVector(frame.bowVector), featureVector(frame.featureVector),
	descriptorsL(frame.descriptorsL.clone()), descriptorsR(frame.descriptorsR.clone()), mappoints(frame.mappoints),
	outlier(frame.outlier), id(frame.id), referenceKF(frame.referenceKF), pyramid(frame.pyramid), grid(frame.grid)
{
	if (!frame.pose.Empty())
		SetPose(frame.pose);
}

Frame::Frame(ORBVocabulary* voc, double timestamp, const CameraParams& camera, float thDepth,
	const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::KeyPoint>& keypointsUn,
	const std::vector<float>& uright, const std::vector<float>& depth, const cv::Mat& descriptors,
	const ScalePyramidInfo& pyramid, const ImageBounds& imageBounds)
	: voc(voc), timestamp(timestamp), camera(camera), thDepth(thDepth), keypointsL(keypoints), keypointsUn(keypointsUn),
	uright(uright), depth(depth), descriptorsL(descriptors.clone()), pyramid(pyramid), referenceKF(nullptr)
{
	// Frame ID
	id = nextId++;

	N = static_cast<int>(keypoints.size());

	mappoints.assign(N, nullptr);
	outlier.assign(N, false);

	// This is done only for the first Frame (or after a change in the calibration)
	if (initialComputation)
	{
		this->imageBounds = imageBounds;
		initialComputation = false;
	}
	grid.AssignFeatures(keypointsUn, imageBounds, pyramid.nlevels);
}

Frame::Frame(ORBVocabulary* voc, double timestamp, const CameraParams& camera, float thDepth,
	const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::KeyPoint>& keypointsUn,
	const cv::Mat& descriptors, const ScalePyramidInfo& pyramid, const ImageBounds& imageBounds)
	: voc(voc), timestamp(timestamp), camera(camera), thDepth(thDepth), keypointsL(keypoints), keypointsUn(keypointsUn),
	descriptorsL(descriptors.clone()), pyramid(pyramid), referenceKF(nullptr)
{
	// Frame ID
	id = nextId++;

	N = static_cast<int>(keypoints.size());

	// Set no stereo information
	uright.assign(N, -1);
	depth.assign(N, -1);

	mappoints.assign(N, nullptr);
	outlier.assign(N, false);

	// This is done only for the first Frame (or after a change in the calibration)
	if (initialComputation)
	{
		this->imageBounds = imageBounds;
		initialComputation = false;
	}
	grid.AssignFeatures(keypointsUn, imageBounds, pyramid.nlevels);
}

void Frame::SetPose(const CameraPose& pose)
{
	this->pose = pose;
}

Point3D Frame::GetCameraCenter() const
{
	return pose.Invt();
}

void Frame::ComputeBoW()
{
	if (!bowVector.empty())
		return;

	voc->transform(Converter::toDescriptorVector(descriptorsL), bowVector, featureVector, 4);
}

std::vector<size_t> Frame::GetFeaturesInArea(float x, float y, float r, int minLevel, int maxLevel) const
{
	return grid.GetFeaturesInArea(x, y, r, minLevel, maxLevel);
}

Point3D Frame::UnprojectStereo(int i) const
{
	const float Zc = depth[i];
	if (Zc <= 0.f)
		return cv::Mat();

	const float invfx = 1.f / camera.fx;
	const float invfy = 1.f / camera.fy;

	const float u = keypointsUn[i].pt.x;
	const float v = keypointsUn[i].pt.y;

	const float Xc = (u - camera.cx) * Zc * invfx;
	const float Yc = (v - camera.cy) * Zc * invfy;

	const Point3D x3Dc(Xc, Yc, Zc);
	return pose.InvR() * x3Dc + pose.Invt();
}

int Frame::PassedFrom(frameid_t from) const
{
	return id - from;
}

} //namespace ORB_SLAM
