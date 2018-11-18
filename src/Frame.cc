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

#include <thread>
#include <functional>

namespace ORB_SLAM2
{

using KeyPoints = std::vector<cv::KeyPoint>;
using Pyramid = std::vector<cv::Mat>;

namespace Converter
{

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

} // namespace Converter

long unsigned int Frame::nNextId = 0;
bool Frame::initialComputation = true;
ImageBounds Frame::imageBounds;

static void GetScalePyramidInfo(ORBextractor* extractor, ScalePyramidInfo& pyramid)
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
static void UndistortKeyPoints(const KeyPoints& src, KeyPoints& dst, const cv::Mat& K, const cv::Mat1f& distCoeffs)
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

static inline int Round(float v) { return static_cast<int>(std::round(v)); }
static inline int RoundUp(float v) { return static_cast<int>(std::ceil(v)); }
static inline int RoundDn(float v) { return static_cast<int>(std::floor(v)); }
static const int PATCH_RADIUS = 5;
static const int PATCH_SIZE = 2 * PATCH_RADIUS + 1;
static const int SEARCH_RADIUS = 5;

static int PatchDistance(const cv::Mat1b& patchL, const cv::Mat1b& patchR)
{
	const int sub = patchL(PATCH_RADIUS, PATCH_RADIUS) - patchR(PATCH_RADIUS, PATCH_RADIUS);
	int sum = 0;
	for (int y = 0; y < PATCH_SIZE; y++)
		for (int x = 0; x < PATCH_SIZE; x++)
			sum += std::abs(patchL(y, x) - patchR(y, x) - sub);
	return sum;
}

// Search a match for each keypoint in the left image to a keypoint in the right image.
// If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
void ComputeStereoMatches(
	const KeyPoints& keypointsL, const cv::Mat& descriptorsL, const Pyramid& pyramidL,
	const KeyPoints& keypointsR, const cv::Mat& descriptorsR, const Pyramid& pyramidR,
	const std::vector<float>& scaleFactors, const std::vector<float>& invScaleFactors, const CameraParams& camera,
	std::vector<float>& uright, std::vector<float>& depth)
{
	const int nkeypointsL = static_cast<int>(keypointsL.size());
	uright.assign(nkeypointsL, -1.f);
	depth.assign(nkeypointsL, -1.f);

	//Assign keypoints to row table
	const int nrows = pyramidL[0].rows;
	std::vector<std::vector<int>> rowIndices(nrows);

	for (int i = 0; i < nrows; i++)
		rowIndices[i].reserve(200);

	const int nkeypointsR = static_cast<int>(keypointsR.size());
	for (int iR = 0; iR < nkeypointsR; iR++)
	{
		const cv::KeyPoint& keypoint = keypointsR[iR];
		const float y0 = keypoint.pt.y;
		const float r = 2.f * scaleFactors[keypoint.octave];
		const int miny = RoundDn(y0 - r);
		const int maxy = RoundUp(y0 + r);
		for (int y = miny; y <= maxy; y++)
			rowIndices[y].push_back(iR);
	}

	// Set limits for search
	const float minZ = camera.baseline;
	const float mind = 0;
	const float maxd = camera.bf / minZ;

	// For each left keypoint search a match in the right image
	std::vector<std::pair<int, int>> distIndices;
	distIndices.reserve(nkeypointsL);

	std::vector<int> distances(2 * SEARCH_RADIUS + 1);

	const int TH_ORB_DIST = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;
	const float eps = 0.01f;

	for (int iL = 0; iL < nkeypointsL; iL++)
	{
		const cv::KeyPoint& keypointL = keypointsL[iL];
		const int octaveL = keypointL.octave;
		const float vL = keypointL.pt.y;
		const float uL = keypointL.pt.x;

		const std::vector<int>& candidates = rowIndices[static_cast<int>(vL)];

		if (candidates.empty())
			continue;

		const float minu = uL - maxd;
		const float maxu = uL - mind;

		if (maxu < 0)
			continue;

		int minDist = ORBmatcher::TH_HIGH;
		int bestIdxR = 0;

		const cv::Mat& descL = descriptorsL.row(iL);

		// Compare descriptor to right keypoints
		for (int iR : candidates)
		{
			const cv::KeyPoint& keypointR = keypointsR[iR];
			const int octaveR = keypointR.octave;

			if (octaveR < octaveL - 1 || octaveR > octaveL + 1)
				continue;

			const float uR = keypointR.pt.x;

			if (uR >= minu && uR <= maxu)
			{
				const cv::Mat& descR = descriptorsR.row(iR);
				const int dist = ORBmatcher::DescriptorDistance(descL, descR);

				if (dist < minDist)
				{
					minDist = dist;
					bestIdxR = iR;
				}
			}
		}

		// Subpixel match by correlation
		if (minDist < TH_ORB_DIST)
		{
			const cv::Mat& imageL = pyramidL[octaveL];
			const cv::Mat& imageR = pyramidR[octaveL];

			// coordinates in image pyramid at keypoint scale
			const float scaleFactor = invScaleFactors[octaveL];
			const int suL = Round(scaleFactor * keypointL.pt.x);
			const int svL = Round(scaleFactor * keypointL.pt.y);
			const int suR = Round(scaleFactor * keypointsR[bestIdxR].pt.x);

			// sliding window search
			const cv::Rect roiL(suL - PATCH_RADIUS, svL - PATCH_RADIUS, PATCH_SIZE, PATCH_SIZE);
			cv::Mat IL = imageL(roiL);

			int minDist = std::numeric_limits<int>::max();
			int bestdxR = 0;

			if (suR + SEARCH_RADIUS - PATCH_RADIUS < 0 || suR + SEARCH_RADIUS + PATCH_RADIUS + 1 >= imageR.cols)
				continue;

			for (int dxR = -SEARCH_RADIUS; dxR <= SEARCH_RADIUS; dxR++)
			{
				const cv::Rect roiR(suR + dxR - PATCH_RADIUS, svL - PATCH_RADIUS, PATCH_SIZE, PATCH_SIZE);
				cv::Mat IR = imageR(roiR);

				const int dist = PatchDistance(IL, IR);
				if (dist < minDist)
				{
					minDist = dist;
					bestdxR = dxR;
				}

				distances[SEARCH_RADIUS + dxR] = dist;
			}

			if (bestdxR == -SEARCH_RADIUS || bestdxR == SEARCH_RADIUS)
				continue;

			// Sub-pixel match (Parabola fitting)
			const int dist1 = distances[SEARCH_RADIUS + bestdxR - 1];
			const int dist2 = distances[SEARCH_RADIUS + bestdxR];
			const int dist3 = distances[SEARCH_RADIUS + bestdxR + 1];

			const float deltaR = (dist1 - dist3) / (2.f * (dist1 + dist3 - 2.f * dist2));

			if (deltaR < -1 || deltaR > 1)
				continue;

			// Re-scaled coordinate
			float bestuR = scaleFactors[octaveL] * (suR + bestdxR + deltaR);

			float disparity = (uL - bestuR);

			if (disparity >= mind && disparity < maxd)
			{
				if (disparity <= 0)
				{
					disparity = eps;
					bestuR = uL - eps;
				}
				depth[iL] = camera.bf / disparity;
				uright[iL] = bestuR;
				distIndices.push_back(std::make_pair(minDist, iL));
			}
		}
	}

	std::sort(std::begin(distIndices), std::end(distIndices), std::greater<std::pair<int, int>>());
	const int m = std::max(static_cast<int>(distIndices.size()) / 2 - 1, 0);
	const int median = distIndices[m].first;
	const float thDist = 1.5f * 1.4f * median;

	for (const auto& v : distIndices)
	{
		const int dist = v.first;
		const int idx = v.second;

		if (dist < thDist)
			break;

		uright[idx] = -1;
		depth[idx] = -1;
	}
}

// Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
static void ComputeStereoFromRGBD(const KeyPoints& keypoints, const KeyPoints& keypointsUn, const cv::Mat& depthImage,
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

void CameraPose::Update()
{
	Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
	Rwc = Rcw.t();
	tcw = Tcw.rowRange(0, 3).col(3);
	Ow = -Rcw.t() * tcw;
}

cv::Mat CameraPose::GetRotationInverse() const
{
	return Rwc.clone();
}

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
	:voc(frame.voc),
	timestamp(frame.timestamp), camera(frame.camera),
	thDepth(frame.thDepth), N(frame.N), keypointsL(frame.keypointsL),
	keypointsR(frame.keypointsR), keypointsUn(frame.keypointsUn), uright(frame.uright),
	depth(frame.depth), bowVector(frame.bowVector), featureVector(frame.featureVector),
	descriptorsL(frame.descriptorsL.clone()), descriptorsR(frame.descriptorsR.clone()),
	mappoints(frame.mappoints), outlier(frame.outlier), mnId(frame.mnId),
	referenceKF(frame.referenceKF), pyramid(frame.pyramid), grid(frame.grid)
{
	if (!frame.pose.Tcw.empty())
		SetPose(frame.pose.Tcw);
}


Frame::Frame(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp, ORBextractor* extractorL,
	ORBextractor* extractorR, ORBVocabulary* voc, const CameraParams& camera, const cv::Mat& distCoef, float thDepth)
	: voc(voc), timestamp(timestamp), camera(camera), thDepth(thDepth), referenceKF(nullptr)
{
	// Frame ID
	mnId = nNextId++;

	// Scale Level Info
	GetScalePyramidInfo(extractorL, pyramid);

	// ORB extraction
	std::thread threadL([&]() { (*extractorL)(imageL, cv::Mat(), keypointsL, descriptorsL); });
	std::thread threadR([&]() { (*extractorR)(imageR, cv::Mat(), keypointsR, descriptorsR); });

	threadL.join();
	threadR.join();

	N = static_cast<int>(keypointsL.size());

	if (keypointsL.empty())
		return;

	UndistortKeyPoints(keypointsL, keypointsUn, camera.Mat(), distCoef);

	ComputeStereoMatches(keypointsL, descriptorsL, extractorL->mvImagePyramid,
		keypointsR, descriptorsR, extractorR->mvImagePyramid,
		pyramid.scaleFactors, pyramid.invScaleFactors, camera, uright, depth);

	mappoints.assign(N, nullptr);
	outlier.assign(N, false);

	// This is done only for the first Frame (or after a change in the calibration)
	if (initialComputation)
	{
		imageBounds = ComputeImageBounds(imageL, camera.Mat(), distCoef);
		initialComputation = false;
	}
	grid.AssignFeatures(keypointsUn, imageBounds, pyramid.nlevels);
}

Frame::Frame(const cv::Mat& image, const cv::Mat& depthImage, double timestamp, ORBextractor* extractor,
	ORBVocabulary* voc, const CameraParams& camera, const cv::Mat& distCoef, float thDepth)
	: voc(voc), timestamp(timestamp), camera(camera), thDepth(thDepth)
{
	// Frame ID
	mnId = nNextId++;

	// Scale Level Info
	GetScalePyramidInfo(extractor, pyramid);

	// ORB extraction
	(*extractor)(image, cv::Mat(), keypointsL, descriptorsL);

	N = static_cast<int>(keypointsL.size());

	if (keypointsL.empty())
		return;

	UndistortKeyPoints(keypointsL, keypointsUn, camera.Mat(), distCoef);

	ComputeStereoFromRGBD(keypointsL, keypointsUn, depthImage, camera, uright, depth);

	mappoints.assign(N, nullptr);
	outlier.assign(N, false);

	// This is done only for the first Frame (or after a change in the calibration)
	if (initialComputation)
	{
		imageBounds = ComputeImageBounds(image, camera.Mat(), distCoef);
		initialComputation = false;
	}
	grid.AssignFeatures(keypointsUn, imageBounds, pyramid.nlevels);
}

Frame::Frame(const cv::Mat& image, double timestamp, ORBextractor* extractor, ORBVocabulary* voc,
	const CameraParams& camera, const cv::Mat& distCoef, float thDepth)
	: voc(voc), timestamp(timestamp), camera(camera), thDepth(thDepth)
{
	// Frame ID
	mnId = nNextId++;

	// Scale Level Info
	GetScalePyramidInfo(extractor, pyramid);

	// ORB extraction
	(*extractor)(image, cv::Mat(), keypointsL, descriptorsL);

	N = static_cast<int>(keypointsL.size());

	if (keypointsL.empty())
		return;

	UndistortKeyPoints(keypointsL, keypointsUn, camera.Mat(), distCoef);

	// Set no stereo information
	uright = vector<float>(N, -1);
	depth = vector<float>(N, -1);

	mappoints.assign(N, nullptr);
	outlier.assign(N, false);

	// This is done only for the first Frame (or after a change in the calibration)
	if (initialComputation)
	{
		imageBounds = ComputeImageBounds(image, camera.Mat(), distCoef);
		initialComputation = false;
	}
	grid.AssignFeatures(keypointsUn, imageBounds, pyramid.nlevels);
}

void Frame::SetPose(cv::Mat Tcw)
{
	pose.Tcw = Tcw.clone();
	pose.Update();
}

cv::Mat Frame::GetCameraCenter() const
{
	return pose.Ow.clone();
}

bool Frame::isInFrustum(MapPoint* mappoint, float viewingCosLimit)
{
	mappoint->mbTrackInView = false;

	// 3D in absolute coordinates
	cv::Mat Xw = mappoint->GetWorldPos();

	// 3D in camera coordinates
	const cv::Mat Xc = pose.Rcw * Xw + pose.tcw;
	const float PcX = Xc.at<float>(0);
	const float PcY = Xc.at<float>(1);
	const float PcZ = Xc.at<float>(2);

	// Check positive depth
	if (PcZ < 0.f)
		return false;

	// Project in image and check it is not outside
	const float invZ = 1.f / PcZ;
	const float u = camera.fx * PcX * invZ + camera.cx;
	const float v = camera.fy * PcY * invZ + camera.cy;

	if (!imageBounds.Contains(u, v))
		return false;

	// Check distance is in the scale invariance region of the MapPoint
	const float maxDistance = mappoint->GetMaxDistanceInvariance();
	const float minDistance = mappoint->GetMinDistanceInvariance();
	const cv::Mat PO = Xw - pose.Ow;
	const float dist = static_cast<float>(cv::norm(PO));

	if (dist < minDistance || dist > maxDistance)
		return false;

	// Check viewing angle
	const cv::Mat Pn = mappoint->GetNormal();

	const float viewCos = static_cast<float>(PO.dot(Pn) / dist);

	if (viewCos < viewingCosLimit)
		return false;

	// Predict scale in the image
	const int scale = mappoint->PredictScale(dist, this);

	// Data used by the tracking
	mappoint->mbTrackInView = true;
	mappoint->mTrackProjX = u;
	mappoint->mTrackProjXR = u - camera.bf * invZ;
	mappoint->mTrackProjY = v;
	mappoint->mnTrackScaleLevel = scale;
	mappoint->mTrackViewCos = viewCos;

	return true;
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

cv::Mat Frame::UnprojectStereo(const int &i)
{
	const float z = depth[i];
	const float invfx = 1.f / camera.fx;
	const float invfy = 1.f / camera.fy;
	const float cx = camera.cx;
	const float cy = camera.cy;
	if (z > 0)
	{
		const float u = keypointsUn[i].pt.x;
		const float v = keypointsUn[i].pt.y;
		const float x = (u - cx)*z*invfx;
		const float y = (v - cy)*z*invfy;
		cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
		return pose.Rwc*x3Dc + pose.Ow;
	}
	else
		return cv::Mat();
}

} //namespace ORB_SLAM
