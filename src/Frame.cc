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

// Search a match for each keypoint in the left image to a keypoint in the right image.
// If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
void ComputeStereoMatches(const KeyPoints& mvKeys, const cv::Mat& mDescriptors, const Pyramid& pyramidL,
	const KeyPoints& mvKeysRight, const cv::Mat& mDescriptorsRight, const Pyramid& pyramidR,
	const std::vector<float>& mvScaleFactors, const std::vector<float>& mvInvScaleFactors, const CameraParams& camera,
	std::vector<float>& mvuRight, std::vector<float>& mvDepth)
{
	const int N = static_cast<int>(mvKeys.size());

	mvuRight = vector<float>(N, -1.0f);
	mvDepth = vector<float>(N, -1.0f);

	const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

	const int nRows = pyramidL[0].rows;

	//Assign keypoints to row table
	vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

	for (int i = 0; i < nRows; i++)
		vRowIndices[i].reserve(200);

	const int Nr = mvKeysRight.size();

	for (int iR = 0; iR < Nr; iR++)
	{
		const cv::KeyPoint &kp = mvKeysRight[iR];
		const float &kpY = kp.pt.y;
		const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
		const int maxr = ceil(kpY + r);
		const int minr = floor(kpY - r);

		for (int yi = minr; yi <= maxr; yi++)
			vRowIndices[yi].push_back(iR);
	}

	// Set limits for search
	const float minZ = camera.baseline;
	const float minD = 0;
	const float maxD = camera.bf / minZ;

	// For each left keypoint search a match in the right image
	vector<pair<int, int> > vDistIdx;
	vDistIdx.reserve(N);

	for (int iL = 0; iL < N; iL++)
	{
		const cv::KeyPoint &kpL = mvKeys[iL];
		const int &levelL = kpL.octave;
		const float &vL = kpL.pt.y;
		const float &uL = kpL.pt.x;

		const vector<size_t> &vCandidates = vRowIndices[vL];

		if (vCandidates.empty())
			continue;

		const float minU = uL - maxD;
		const float maxU = uL - minD;

		if (maxU < 0)
			continue;

		int bestDist = ORBmatcher::TH_HIGH;
		size_t bestIdxR = 0;

		const cv::Mat &dL = mDescriptors.row(iL);

		// Compare descriptor to right keypoints
		for (size_t iC = 0; iC < vCandidates.size(); iC++)
		{
			const size_t iR = vCandidates[iC];
			const cv::KeyPoint &kpR = mvKeysRight[iR];

			if (kpR.octave<levelL - 1 || kpR.octave>levelL + 1)
				continue;

			const float &uR = kpR.pt.x;

			if (uR >= minU && uR <= maxU)
			{
				const cv::Mat &dR = mDescriptorsRight.row(iR);
				const int dist = ORBmatcher::DescriptorDistance(dL, dR);

				if (dist < bestDist)
				{
					bestDist = dist;
					bestIdxR = iR;
				}
			}
		}

		// Subpixel match by correlation
		if (bestDist < thOrbDist)
		{
			// coordinates in image pyramid at keypoint scale
			const float uR0 = mvKeysRight[bestIdxR].pt.x;
			const float scaleFactor = mvInvScaleFactors[kpL.octave];
			const float scaleduL = round(kpL.pt.x*scaleFactor);
			const float scaledvL = round(kpL.pt.y*scaleFactor);
			const float scaleduR0 = round(uR0*scaleFactor);

			// sliding window search
			const int w = 5;
			cv::Mat IL = pyramidL[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
			IL.convertTo(IL, CV_32F);
			IL = IL - IL.at<float>(w, w) *cv::Mat::ones(IL.rows, IL.cols, CV_32F);

			int bestDist = INT_MAX;
			int bestincR = 0;
			const int L = 5;
			vector<float> vDists;
			vDists.resize(2 * L + 1);

			const float iniu = scaleduR0 + L - w;
			const float endu = scaleduR0 + L + w + 1;
			if (iniu < 0 || endu >= pyramidR[kpL.octave].cols)
				continue;

			for (int incR = -L; incR <= +L; incR++)
			{
				cv::Mat IR = pyramidR[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
				IR.convertTo(IR, CV_32F);
				IR = IR - IR.at<float>(w, w) *cv::Mat::ones(IR.rows, IR.cols, CV_32F);

				float dist = cv::norm(IL, IR, cv::NORM_L1);
				if (dist < bestDist)
				{
					bestDist = dist;
					bestincR = incR;
				}

				vDists[L + incR] = dist;
			}

			if (bestincR == -L || bestincR == L)
				continue;

			// Sub-pixel match (Parabola fitting)
			const float dist1 = vDists[L + bestincR - 1];
			const float dist2 = vDists[L + bestincR];
			const float dist3 = vDists[L + bestincR + 1];

			const float deltaR = (dist1 - dist3) / (2.0f*(dist1 + dist3 - 2.0f*dist2));

			if (deltaR < -1 || deltaR>1)
				continue;

			// Re-scaled coordinate
			float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

			float disparity = (uL - bestuR);

			if (disparity >= minD && disparity < maxD)
			{
				if (disparity <= 0)
				{
					disparity = 0.01;
					bestuR = uL - 0.01;
				}
				mvDepth[iL] = camera.bf / disparity;
				mvuRight[iL] = bestuR;
				vDistIdx.push_back(pair<int, int>(bestDist, iL));
			}
		}
	}

	sort(vDistIdx.begin(), vDistIdx.end());
	const float median = vDistIdx[vDistIdx.size() / 2].first;
	const float thDist = 1.5f*1.4f*median;

	for (int i = vDistIdx.size() - 1; i >= 0; i--)
	{
		if (vDistIdx[i].first < thDist)
			break;
		else
		{
			mvuRight[vDistIdx[i].second] = -1;
			mvDepth[vDistIdx[i].second] = -1;
		}
	}
}

// Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
static void ComputeStereoFromRGBD(const KeyPoints& mvKeys, const KeyPoints& mvKeysUn, const cv::Mat &imDepth,
	const CameraParams& camera, std::vector<float>& mvuRight, std::vector<float>& mvDepth)
{
	const int N = static_cast<int>(mvKeys.size());

	mvuRight = vector<float>(N, -1);
	mvDepth = vector<float>(N, -1);

	for (int i = 0; i < N; i++)
	{
		const cv::KeyPoint &kp = mvKeys[i];
		const cv::KeyPoint &kpU = mvKeysUn[i];

		const float &v = kp.pt.y;
		const float &u = kp.pt.x;

		const float d = imDepth.at<float>(v, u);

		if (d > 0)
		{
			mvDepth[i] = d;
			mvuRight[i] = kpU.pt.x - camera.bf / d;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////
// ImageBounds Class
//////////////////////////////////////////////////////////////////////////////////
ImageBounds::ImageBounds(float minx, float maxx, float miny, float maxy)
	: minx(minx), maxx(maxx), miny(miny), maxy(maxy) {}

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
	mfGridElementWidthInv = COLS / imageBounds.Width();
	mfGridElementHeightInv = ROWS / imageBounds.Height();

	keypoints_ = keypoints;
	imageBounds_ = imageBounds;
	nlevels_ = nlevels;

	const int N = static_cast<int>(keypoints.size());

	int nReserve = 0.5f*N / (COLS*ROWS);
	for (unsigned int i = 0; i < COLS; i++)
		for (unsigned int j = 0; j < ROWS; j++)
			mGrid[i][j].reserve(nReserve);

	for (int i = 0; i < N; i++)
	{
		const cv::KeyPoint &kp = keypoints[i];

		int posX, posY;

		posX = (int)round((kp.pt.x - imageBounds.minx)*mfGridElementWidthInv);
		posY = (int)round((kp.pt.y - imageBounds.miny)*mfGridElementHeightInv);

		//Keypoint's coordinates are undistorted, which could cause to go out of the image
		if (posX < 0 || posX >= COLS || posY < 0 || posY >= ROWS)
			continue;

		mGrid[posX][posY].push_back(i);
	}
}

std::vector<size_t> FeaturesGrid::GetFeaturesInArea(float x, float y, float r, int minLevel, int maxLevel) const
{
	const int N = static_cast<int>(keypoints_.size());

	vector<size_t> vIndices;
	vIndices.reserve(N);

	const float mnMinX = imageBounds_.minx;
	const float mnMinY = imageBounds_.miny;

	const int nMinCellX = max(0, (int)floor((x - mnMinX - r)*mfGridElementWidthInv));
	if (nMinCellX >= COLS)
		return vIndices;

	const int nMaxCellX = min((int)COLS - 1, (int)ceil((x - mnMinX + r)*mfGridElementWidthInv));
	if (nMaxCellX < 0)
		return vIndices;

	const int nMinCellY = max(0, (int)floor((y - mnMinY - r)*mfGridElementHeightInv));
	if (nMinCellY >= ROWS)
		return vIndices;

	const int nMaxCellY = min((int)ROWS - 1, (int)ceil((y - mnMinY + r)*mfGridElementHeightInv));
	if (nMaxCellY < 0)
		return vIndices;

	const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

	for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
	{
		for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
		{
			const vector<size_t> vCell = mGrid[ix][iy];
			if (vCell.empty())
				continue;

			for (size_t j = 0, jend = vCell.size(); j < jend; j++)
			{
				const cv::KeyPoint &kpUn = keypoints_[vCell[j]];
				if (bCheckLevels)
				{
					if (kpUn.octave < minLevel)
						continue;
					if (maxLevel >= 0)
						if (kpUn.octave > maxLevel)
							continue;
				}

				const float distx = kpUn.pt.x - x;
				const float disty = kpUn.pt.y - y;

				if (fabs(distx) < r && fabs(disty) < r)
					vIndices.push_back(vCell[j]);
			}
		}
	}

	return vIndices;
}

void CameraPose::Update()
{
	mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
	mRwc = mRcw.t();
	mtcw = mTcw.rowRange(0, 3).col(3);
	mOw = -mRcw.t()*mtcw;
}

cv::Mat CameraPose::GetRotationInverse() const
{
	return mRwc.clone();
}

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
	:voc(frame.voc),
	timestamp(frame.timestamp), camera(frame.camera),
	thDepth(frame.thDepth), N(frame.N), keypointsL(frame.keypointsL),
	keypointsR(frame.keypointsR), keypointsUn(frame.keypointsUn), uright(frame.uright),
	depth(frame.depth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
	descriptorsL(frame.descriptorsL.clone()), descriptorsR(frame.descriptorsR.clone()),
	mappoints(frame.mappoints), outlier(frame.outlier), mnId(frame.mnId),
	referenceKF(frame.referenceKF), pyramid(frame.pyramid), grid(frame.grid)
{
	if (!frame.pose.mTcw.empty())
		SetPose(frame.pose.mTcw);
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

	N = keypointsL.size();

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

	N = keypointsL.size();

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

	N = keypointsL.size();

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
	pose.mTcw = Tcw.clone();
	pose.Update();
}

cv::Mat Frame::GetCameraCenter() const
{
	return pose.mOw.clone();
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
	pMP->mbTrackInView = false;

	// 3D in absolute coordinates
	cv::Mat P = pMP->GetWorldPos();

	// 3D in camera coordinates
	const cv::Mat Pc = pose.mRcw*P + pose.mtcw;
	const float &PcX = Pc.at<float>(0);
	const float &PcY = Pc.at<float>(1);
	const float &PcZ = Pc.at<float>(2);

	// Check positive depth
	if (PcZ < 0.0f)
		return false;

	// Project in image and check it is not outside
	const float invz = 1.0f / PcZ;
	const float u = camera.fx*PcX*invz + camera.cx;
	const float v = camera.fy*PcY*invz + camera.cy;

	if (!imageBounds.Contains(u, v))
		return false;

	// Check distance is in the scale invariance region of the MapPoint
	const float maxDistance = pMP->GetMaxDistanceInvariance();
	const float minDistance = pMP->GetMinDistanceInvariance();
	const cv::Mat PO = P - pose.mOw;
	const float dist = cv::norm(PO);

	if (dist<minDistance || dist>maxDistance)
		return false;

	// Check viewing angle
	cv::Mat Pn = pMP->GetNormal();

	const float viewCos = PO.dot(Pn) / dist;

	if (viewCos < viewingCosLimit)
		return false;

	// Predict scale in the image
	const int nPredictedLevel = pMP->PredictScale(dist, this);

	// Data used by the tracking
	pMP->mbTrackInView = true;
	pMP->mTrackProjX = u;
	pMP->mTrackProjXR = u - camera.bf*invz;
	pMP->mTrackProjY = v;
	pMP->mnTrackScaleLevel = nPredictedLevel;
	pMP->mTrackViewCos = viewCos;

	return true;
}

void Frame::ComputeBoW()
{
	if (mBowVec.empty())
	{
		vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(descriptorsL);
		voc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
	}
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
		return pose.mRwc*x3Dc + pose.mOw;
	}
	else
		return cv::Mat();
}

} //namespace ORB_SLAM
