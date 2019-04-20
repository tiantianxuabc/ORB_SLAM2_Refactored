/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Ra’Yl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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

#include "ORBmatcher.h"
#include "CameraPose.h"
#include "CameraProjection.h"

#include <Thirdparty/DBoW2/DBoW2/FeatureVector.h>

#ifdef _WIN32
#define popcnt32 __popcnt
#define popcnt64 __popcnt64
#else
#define popcnt32 __builtin_popcount
#define popcnt64 __builtin_popcountll
#endif

namespace ORB_SLAM2
{

using MatchIdx = std::pair<int, int>;

// Constant numbers
static const int TH_HIGH = 100;
static const int TH_LOW = 50;
static const int HISTO_LENGTH = 30;

static const int PATCH_RADIUS = 5;
static const int PATCH_SIZE = 2 * PATCH_RADIUS + 1;
static const int SEARCH_RADIUS = 5;

// Inline functions
static inline int Round(float v) { return static_cast<int>(std::round(v)); }
static inline int RoundUp(float v) { return static_cast<int>(std::ceil(v)); }
static inline int RoundDn(float v) { return static_cast<int>(std::floor(v)); }
static inline float RadiusByViewingCos(float viewCos) { return viewCos > 0.998 ? 2.5f : 4.f; }
static inline float NormSq(float x, float y) { return x * x + y * y; }
static inline float NormSq(float x, float y, float z) { return x * x + y * y + z * z; }
template <typename T> static inline T InvalidMatch() { return 0; }
template <> inline MapPoint* InvalidMatch<MapPoint*>() { return nullptr; }
template <> inline int InvalidMatch<int>() { return -1; }

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

	const int TH_ORB_DIST = (TH_HIGH + TH_LOW) / 2;
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

		int minDist = TH_HIGH;
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

template <typename T>
static int CheckOrientation(const KeyPoints& keypoints1, const KeyPoints& keypoints2, const std::vector<MatchIdx>& matchIds,
	std::vector<T>& matchStatus)
{
	CV_Assert(matchStatus.size() == keypoints2.size());

	const float factor = 1.f / HISTO_LENGTH;
	std::vector<int> hist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		hist[i].reserve(500);

	auto diffToBin = [=](float diff)
	{
		if (diff < 0)
			diff += 360;

		int bin = cvRound(factor * diff);
		if (bin == HISTO_LENGTH)
			bin = 0;

		return bin;
	};

	for (const auto& match : matchIds)
	{
		const int i1 = match.first;
		const int i2 = match.second;
		const cv::KeyPoint& keypoint1 = keypoints1[i1];
		const cv::KeyPoint& keypoint2 = keypoints2[i2];
		const int bin = diffToBin(keypoint1.angle - keypoint2.angle);
		CV_Assert(bin >= 0 && bin < HISTO_LENGTH);
		hist[bin].push_back(i2);
	}

	std::sort(std::begin(hist), std::end(hist), [](const std::vector<int>& lhs, const std::vector<int>& rhs)
	{
		return lhs.size() > rhs.size();
	});

	const size_t max1 = hist[0].size();
	const size_t max2 = hist[1].size();
	const size_t max3 = hist[2].size();

	int eraseBin = 3;
	if (max2 < 0.1 * max1)
		eraseBin = 1;
	else if (max3 < 0.1 * max1)
		eraseBin = 2;

	int reduction = 0;
	for (int bin = eraseBin; bin < HISTO_LENGTH; bin++)
	{
		for (int i2 : hist[bin])
		{
			matchStatus[i2] = InvalidMatch<T>();
			reduction++;
		}
	}

	return static_cast<int>(matchIds.size() - reduction);
}

ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : fNNRatio_(nnratio), checkOrientation_(checkOri)
{
}

int ORBmatcher::SearchByProjection(Frame& frame, const std::vector<MapPoint*>& mappoints, float th)
{
	int nmatches = 0;

	for (MapPoint* mappoint : mappoints)
	{
		if (!mappoint->trackInView || mappoint->isBad())
			continue;

		const int predictedScale = mappoint->trackScaleLevel;

		// The size of the window will depend on the viewing direction
		const float r = RadiusByViewingCos(mappoint->trackViewCos);
		const float radius = th * r * frame.pyramid.scaleFactors[predictedScale];
		const float u = mappoint->trackProjX;
		const float v = mappoint->trackProjY;

		const std::vector<size_t> indices = frame.GetFeaturesInArea(u, v, radius, predictedScale - 1, predictedScale);
		if (indices.empty())
			continue;

		const cv::Mat desc1 = mappoint->GetDescriptor();

		int bestDist = 256;
		int bestLevel = -1;
		int secondbestDist = 256;
		int secondBestLevel = -1;
		int bestIdx = -1;

		// Get best and second matches with near keypoints
		for (size_t idx : indices)
		{
			if (frame.mappoints[idx] && frame.mappoints[idx]->Observations() > 0)
				continue;

			if (frame.uright[idx] > 0 && fabsf(mappoint->trackProjXR - frame.uright[idx]) > radius)
				continue;

			const cv::Mat desc2 = frame.descriptors.row(static_cast<int>(idx));
			const int dist = DescriptorDistance(desc1, desc2);
			if (dist < bestDist)
			{
				secondbestDist = bestDist;
				bestDist = dist;
				secondBestLevel = bestLevel;
				bestLevel = frame.keypointsUn[idx].octave;
				bestIdx = static_cast<int>(idx);
			}
			else if (dist < secondbestDist)
			{
				secondBestLevel = frame.keypointsUn[idx].octave;
				secondbestDist = dist;
			}
		}

		// Apply ratio to second match (only if best and second are in the same scale level)
		if (bestDist <= TH_HIGH)
		{
			if (bestLevel == secondBestLevel && bestDist > fNNRatio_ * secondbestDist)
				continue;

			frame.mappoints[bestIdx] = mappoint;
			nmatches++;
		}
	}

	return nmatches;
}

static bool CheckDistEpipolarLine(const cv::KeyPoint& keypoint1, const cv::KeyPoint& keypoint2,
	const cv::Mat1f& F12, const KeyFrame* keyframe2)
{
	const cv::Point2f& pt1 = keypoint1.pt;
	const cv::Point2f& pt2 = keypoint2.pt;

	// Epipolar line in second image l = x1'F12 = [a b c]
	const float a = pt1.x * F12(0, 0) + pt1.y * F12(1, 0) + F12(2, 0);
	const float b = pt1.x * F12(0, 1) + pt1.y * F12(1, 1) + F12(2, 1);
	const float c = pt1.x * F12(0, 2) + pt1.y * F12(1, 2) + F12(2, 2);

	const float num = a * pt2.x + b * pt2.y + c;
	const float den = a * a + b * b;

	if (den == 0)
		return false;

	const float dsqr = num * num / den;

	return dsqr < 3.84 * keyframe2->pyramid.sigmaSq[keypoint2.octave];
}

struct FeatureVectorIterator
{
	using Iterator = DBoW2::FeatureVector::const_iterator;
	using Indices = Iterator::value_type::second_type;

	FeatureVectorIterator(const DBoW2::FeatureVector& fv1, const DBoW2::FeatureVector& fv2) : fv1(fv1), fv2(fv2)
	{
		it1 = std::cbegin(fv1);
		it2 = std::cbegin(fv2);
	}

	bool end() const { return it1 == std::cend(fv1) || it2 == std::cend(fv2); }

	bool next()
	{
		while (!end())
		{
			if (it1->first == it2->first)
			{
				node1 = it1;
				node2 = it2;
				++it1;
				++it2;
				return true;
			}
			else if (it1->first < it2->first)
			{
				it1 = fv1.lower_bound(it2->first);
			}
			else
			{
				it2 = fv2.lower_bound(it1->first);
			}
		}

		return false;
	}

	const Indices& indices1() const { return node1->second; };
	const Indices& indices2() const { return node2->second; };

	const DBoW2::FeatureVector& fv1;
	const DBoW2::FeatureVector& fv2;
	Iterator node1, node2, it1, it2;
};

int ORBmatcher::SearchByBoW(KeyFrame* keyframe, Frame& frame, std::vector<MapPoint*>& matches)
{
	const std::vector<MapPoint*> mappoints1 = keyframe->GetMapPointMatches();

	matches.assign(frame.N, nullptr);

	int nmatches = 0;

	std::vector<MatchIdx> matchIds;
	matchIds.reserve(keyframe->N);

	FeatureVectorIterator iterator(keyframe->featureVector, frame.featureVector);
	while (iterator.next())
	{
		const auto& indices1 = iterator.indices1();
		const auto& indices2 = iterator.indices2();
		for (auto idx1 : indices1)
		{
			MapPoint* mappoint1 = mappoints1[idx1];

			if (!mappoint1 || mappoint1->isBad())
				continue;

			const cv::Mat desc1 = keyframe->descriptorsL.row(idx1);

			int bestDist = 256;
			int bestIdx2 = -1;
			int secondBestDist = 256;

			for (auto idx2 : indices2)
			{
				if (matches[idx2])
					continue;

				const cv::Mat desc2 = frame.descriptors.row(idx2);
				const int dist = DescriptorDistance(desc1, desc2);
				if (dist < bestDist)
				{
					secondBestDist = bestDist;
					bestDist = dist;
					bestIdx2 = static_cast<int>(idx2);
				}
				else if (dist < secondBestDist)
				{
					secondBestDist = dist;
				}
			}

			if (bestDist <= TH_LOW && bestDist < fNNRatio_ * secondBestDist)
			{
				matches[bestIdx2] = mappoint1;
				nmatches++;

				if (checkOrientation_)
					matchIds.push_back(std::make_pair(static_cast<int>(idx1), bestIdx2));
			}

		}
	}

	if (checkOrientation_)
		nmatches = CheckOrientation(keyframe->keypointsUn, frame.keypointsUn, matchIds, matches);

	return nmatches;
}

int ORBmatcher::SearchByProjection(const KeyFrame* keyframe, const Sim3& Scw, const std::vector<MapPoint*>& mappoints,
	std::vector<MapPoint*>& matched, int th)
{
	// Get Calibration Parameters for later projection
	// Decompose Scw
	const CameraPose pose(Scw.R(), Scw.Invs() * Scw.t());
	const CameraProjection proj(pose, keyframe->camera);
	const Point3D Ow = pose.Invt();

	// Set of MapPoints already found in the KeyFrame
	std::set<MapPoint*> alreadyFound(std::begin(matched), std::end(matched));
	alreadyFound.erase(nullptr);

	int nmatches = 0;

	// For each Candidate MapPoint Project and Match
	for (MapPoint* mappoint : mappoints)
	{
		// Discard Bad MapPoints and already found
		if (mappoint->isBad() || alreadyFound.count(mappoint))
			continue;

		// Get 3D Coords.
		const Point3D Xw = mappoint->GetWorldPos();

		// Transform into Camera Coords.
		const Point3D Xc = proj.WorldToCamera(Xw);

		// Depth must be positive
		if (Xc(2) < 0.f)
			continue;

		// Project into Image
		const Point2D pt = proj.CameraToImage(Xc);
		const float u = pt.x;
		const float v = pt.y;

		// Point must be inside the image
		if (!keyframe->IsInImage(u, v))
			continue;

		// Depth must be inside the scale invariance region of the point
		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();
		const Vec3D PO = Xw - Ow;
		const float dist = static_cast<float>(cv::norm(PO));
		if (dist < minDistance || dist > maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		const Vec3D Pn = mappoint->GetNormal();
		if (PO.dot(Pn) < 0.5 * dist)
			continue;

		const int predictedScale = mappoint->PredictScale(dist, keyframe);

		// Search in a radius
		const float radius = th * keyframe->pyramid.scaleFactors[predictedScale];

		const std::vector<size_t> indices = keyframe->GetFeaturesInArea(u, v, radius);
		if (indices.empty())
			continue;

		// Match to the most similar keypoint in the radius
		const cv::Mat desc1 = mappoint->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (size_t idx : indices)
		{
			if (matched[idx])
				continue;

			const int scale = keyframe->keypointsUn[idx].octave;
			if (scale < predictedScale - 1 || scale > predictedScale)
				continue;

			const cv::Mat desc2 = keyframe->descriptorsL.row(static_cast<int>(idx));
			const int dist = DescriptorDistance(desc1, desc2);
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = static_cast<int>(idx);
			}
		}

		if (bestDist <= TH_LOW)
		{
			matched[bestIdx] = mappoint;
			nmatches++;
		}
	}

	return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame& frame1, Frame& frame2, std::vector<cv::Point2f>& prevMatched,
	std::vector<int>& matches12, int windowSize)
{
	int nmatches = 0;
	matches12.assign(frame1.keypointsUn.size(), -1);

	std::vector<int> matchedDistance(frame2.keypointsUn.size(), std::numeric_limits<int>::max());
	std::vector<int> matches21(frame2.keypointsUn.size(), -1);

	std::vector<MatchIdx> matchIds;
	matchIds.reserve(frame1.keypointsUn.size());

	const float radius = static_cast<float>(windowSize);

	for (size_t idx1 = 0; idx1 < frame1.keypointsUn.size(); idx1++)
	{
		const cv::KeyPoint& keypoint1 = frame1.keypointsUn[idx1];
		const int level1 = keypoint1.octave;
		if (level1 > 0)
			continue;

		const float u = prevMatched[idx1].x;
		const float v = prevMatched[idx1].y;
		const std::vector<size_t> indices2 = frame2.GetFeaturesInArea(u, v, radius, level1, level1);
		if (indices2.empty())
			continue;

		const cv::Mat desc1 = frame1.descriptors.row(static_cast<int>(idx1));

		int bestDist = std::numeric_limits<int>::max();
		int secondBestDist = std::numeric_limits<int>::max();
		int bestIdx2 = -1;

		for (size_t idx2 : indices2)
		{
			const cv::Mat desc2 = frame2.descriptors.row(static_cast<int>(idx2));
			const int dist = DescriptorDistance(desc1, desc2);

			if (matchedDistance[idx2] <= dist)
				continue;

			if (dist < bestDist)
			{
				secondBestDist = bestDist;
				bestDist = dist;
				bestIdx2 = static_cast<int>(idx2);
			}
			else if (dist < secondBestDist)
			{
				secondBestDist = dist;
			}
		}

		if (bestDist <= TH_LOW && bestDist < secondBestDist * fNNRatio_)
		{
			if (matches21[bestIdx2] >= 0)
			{
				matches12[matches21[bestIdx2]] = -1;
				nmatches--;
			}

			matches12[idx1] = bestIdx2;
			matches21[bestIdx2] = static_cast<int>(idx1);
			matchedDistance[bestIdx2] = bestDist;
			nmatches++;

			if (checkOrientation_)
				matchIds.push_back(std::make_pair(bestIdx2, static_cast<int>(idx1)));
		}
	}

	if (checkOrientation_)
		nmatches = CheckOrientation(frame2.keypointsUn, frame1.keypointsUn, matchIds, matches12);

	// Update prev matched
	for (size_t i1 = 0, iend1 = matches12.size(); i1 < iend1; i1++)
		if (matches12[i1] >= 0)
			prevMatched[i1] = frame2.keypointsUn[matches12[i1]].pt;

	return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame* keyframe1, KeyFrame* keyframe2, std::vector<MapPoint*>& matches12)
{
	const KeyPoints& keypoints1 = keyframe1->keypointsUn;
	const KeyPoints& keypoints2 = keyframe2->keypointsUn;
	const std::vector<MapPoint*> mappoints1 = keyframe1->GetMapPointMatches();
	const std::vector<MapPoint*> mappoints2 = keyframe2->GetMapPointMatches();
	const cv::Mat& descriptors1 = keyframe1->descriptorsL;
	const cv::Mat& descriptors2 = keyframe2->descriptorsL;

	int nmatches = 0;

	matches12.assign(mappoints1.size(), nullptr);
	std::vector<bool> matched2(mappoints2.size(), false);

	std::vector<MatchIdx> matchIds;
	matchIds.reserve(keypoints1.size());

	FeatureVectorIterator iterator(keyframe1->featureVector, keyframe2->featureVector);
	while (iterator.next())
	{
		const auto& indices1 = iterator.indices1();
		const auto& indices2 = iterator.indices2();
		for (auto idx1 : indices1)
		{
			MapPoint* mappoint1 = mappoints1[idx1];
			if (!mappoint1 || mappoint1->isBad())
				continue;

			const cv::Mat desc1 = descriptors1.row(idx1);

			int bestDist = 256;
			int bestIdx2 = -1;
			int secondBestDist = 256;

			for (auto idx2 : indices2)
			{
				MapPoint* mappoint2 = mappoints2[idx2];
				if (matched2[idx2] || !mappoint2 || mappoint2->isBad())
					continue;

				const cv::Mat desc2 = descriptors2.row(idx2);
				const int dist = DescriptorDistance(desc1, desc2);
				if (dist < bestDist)
				{
					secondBestDist = bestDist;
					bestDist = dist;
					bestIdx2 = idx2;
				}
				else if (dist < secondBestDist)
				{
					secondBestDist = dist;
				}
			}

			if (bestDist < TH_LOW && bestDist < fNNRatio_ * secondBestDist)
			{
				matches12[idx1] = mappoints2[bestIdx2];
				matched2[bestIdx2] = true;
				nmatches++;

				if (checkOrientation_)
					matchIds.push_back(std::make_pair(bestIdx2, static_cast<int>(idx1)));
			}
		}
	}

	if (checkOrientation_)
		nmatches = CheckOrientation(keypoints2, keypoints1, matchIds, matches12);

	return nmatches;
}

int ORBmatcher::SearchForTriangulation(const KeyFrame* keyframe1, const KeyFrame* keyframe2, const cv::Mat& F12,
	std::vector<std::pair<size_t, size_t>>& matchIds, bool onlyStereo)
{
	//Compute epipole in second image
	const CameraProjection proj2(keyframe2->GetPose(), keyframe2->camera);
	const Point2D ep2 = proj2.WorldToImage(keyframe1->GetCameraCenter());

	// Find matches between not tracked keypoints
	// Matching speed-up by ORB Vocabulary
	// Compare only ORB that share the same node

	int nmatches = 0;
	std::vector<bool> matched2(keyframe2->N, false);
	std::vector<int> matches12(keyframe1->N, -1);

	std::vector<MatchIdx> tmpMatchIds;
	tmpMatchIds.reserve(keyframe1->N);

	FeatureVectorIterator iterator(keyframe1->featureVector, keyframe2->featureVector);
	while (iterator.next())
	{
		const auto& indices1 = iterator.indices1();
		const auto& indices2 = iterator.indices2();
		for (auto idx1 : indices1)
		{
			MapPoint* mappoint1 = keyframe1->GetMapPoint(idx1);

			// If there is already a MapPoint skip
			if (mappoint1)
				continue;

			const bool stereo1 = keyframe1->uright[idx1] >= 0;
			if (onlyStereo && !stereo1)
				continue;

			const cv::KeyPoint& keypoint1 = keyframe1->keypointsUn[idx1];
			const cv::Mat desc1 = keyframe1->descriptorsL.row(idx1);

			int bestDist = TH_LOW;
			int bestIdx2 = -1;

			for (auto idx2 : indices2)
			{
				MapPoint* mappoint2 = keyframe2->GetMapPoint(idx2);

				// If we have already matched or there is a MapPoint skip
				if (matched2[idx2] || mappoint2)
					continue;

				const bool stereo2 = keyframe2->uright[idx2] >= 0;
				if (onlyStereo && !stereo2)
					continue;

				const cv::Mat desc2 = keyframe2->descriptorsL.row(idx2);
				const int dist = DescriptorDistance(desc1, desc2);
				if (dist > TH_LOW || dist > bestDist)
					continue;

				const cv::KeyPoint& keypoint2 = keyframe2->keypointsUn[idx2];

				if (!stereo1 && !stereo2)
				{
					const Point2D diff = ep2 - keypoint2.pt;
					if (NormSq(diff.x, diff.y) < 100 * keyframe2->pyramid.scaleFactors[keypoint2.octave])
						continue;
				}

				if (CheckDistEpipolarLine(keypoint1, keypoint2, F12, keyframe2))
				{
					bestIdx2 = static_cast<int>(idx2);
					bestDist = dist;
				}
			}

			if (bestIdx2 >= 0)
			{
				matches12[idx1] = bestIdx2;
				nmatches++;

				if (checkOrientation_)
					tmpMatchIds.push_back(std::make_pair(bestIdx2, static_cast<int>(idx1)));
			}
		}
	}

	if (checkOrientation_)
		nmatches = CheckOrientation(keyframe2->keypointsUn, keyframe1->keypointsUn, tmpMatchIds, matches12);

	matchIds.clear();
	matchIds.reserve(nmatches);

	for (size_t idx1 = 0; idx1 < matches12.size(); idx1++)
	{
		if (matches12[idx1] >= 0)
			matchIds.push_back(std::make_pair(idx1, matches12[idx1]));
	}

	return nmatches;
}

int ORBmatcher::Fuse(KeyFrame* keyframe, const std::vector<MapPoint*>& mappoints, float th)
{
	const CameraProjection proj(keyframe->GetPose(), keyframe->camera);
	const Vec3D Ow = keyframe->GetCameraCenter();
	int nfused = 0;

	for (MapPoint* mappoint : mappoints)
	{
		if (!mappoint || mappoint->isBad() || mappoint->IsInKeyFrame(keyframe))
			continue;

		const Point3D Xw = mappoint->GetWorldPos();
		const Point3D Xc = proj.WorldToCamera(Xw);

		// Depth must be positive
		if (Xc(2) < 0.f)
			continue;

		const Point2D pt = proj.CameraToImage(Xc);
		const float u = pt.x;
		const float v = pt.y;

		// Point must be inside the image
		if (!keyframe->IsInImage(u, v))
			continue;

		const float ur = u - proj.DepthToDisparity(Xc(2));

		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();
		const Vec3D PO = Xw - Ow;
		const float dist3D = static_cast<float>(cv::norm(PO));

		// Depth must be inside the scale pyramid of the image
		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		const Vec3D Pn = mappoint->GetNormal();
		if (PO.dot(Pn) < 0.5 * dist3D)
			continue;

		const int predictedScale = mappoint->PredictScale(dist3D, keyframe);

		// Search in a radius
		const float radius = th * keyframe->pyramid.scaleFactors[predictedScale];

		const std::vector<size_t> indices = keyframe->GetFeaturesInArea(u, v, radius);
		if (indices.empty())
			continue;

		// Match to the most similar keypoint in the radius

		const cv::Mat desc1 = mappoint->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (size_t idx : indices)
		{
			const cv::KeyPoint& keypoint = keyframe->keypointsUn[idx];
			const int scale = keypoint.octave;

			if (scale < predictedScale - 1 || scale > predictedScale)
				continue;

			const Point2D diff = pt - keypoint.pt;
			if (keyframe->uright[idx] >= 0)
			{
				// Check reprojection error in stereo
				const float diffz = ur - keyframe->uright[idx];
				if (NormSq(diff.x, diff.y, diffz) * keyframe->pyramid.invSigmaSq[scale] > 7.8)
					continue;
			}
			else
			{
				if (NormSq(diff.x, diff.y) * keyframe->pyramid.invSigmaSq[scale] > 5.99)
					continue;
			}

			const cv::Mat desc2 = keyframe->descriptorsL.row(static_cast<int>(idx));
			const int dist = DescriptorDistance(desc1, desc2);
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = static_cast<int>(idx);
			}
		}

		// If there is already a MapPoint replace otherwise add new measurement
		if (bestDist <= TH_LOW)
		{
			MapPoint* MPInKF = keyframe->GetMapPoint(bestIdx);
			if (MPInKF)
			{
				if (!MPInKF->isBad())
				{
					if (MPInKF->Observations() > mappoint->Observations())
						mappoint->Replace(MPInKF);
					else
						MPInKF->Replace(mappoint);
				}
			}
			else
			{
				mappoint->AddObservation(keyframe, bestIdx);
				keyframe->AddMapPoint(mappoint, bestIdx);
			}
			nfused++;
		}
	}

	return nfused;
}

int ORBmatcher::Fuse(KeyFrame* keyframe, const Sim3& Scw, const std::vector<MapPoint*>& mappoints,
	float th, std::vector<MapPoint*>& replacePoints)
{
	// Get Calibration Parameters for later projection
	// Decompose Scw
	const CameraPose pose(Scw.R(), Scw.Invs() * Scw.t());
	const CameraProjection proj(pose, keyframe->camera);
	const Point3D Ow = pose.Invt();

	// Set of MapPoints already found in the KeyFrame
	const std::set<MapPoint*> alreadyFound = keyframe->GetMapPoints();

	int nfused = 0;

	// For each candidate MapPoint project and match
	//for (MapPoint* mappoint : mappoints)
	for (size_t i = 0; i < mappoints.size(); i++)
	{
		MapPoint* mappoint = mappoints[i];
		// Discard Bad MapPoints and already found
		if (mappoint->isBad() || alreadyFound.count(mappoint))
			continue;

		// Get 3D Coords.
		const Point3D Xw = mappoint->GetWorldPos();

		// Transform into Camera Coords.
		const Point3D Xc = proj.WorldToCamera(Xw);

		// Depth must be positive
		if (Xc(2) < 0.f)
			continue;

		// Project into Image
		const Point2D pt = proj.CameraToImage(Xc);
		const float u = pt.x;
		const float v = pt.y;

		// Point must be inside the image
		if (!keyframe->IsInImage(u, v))
			continue;

		// Depth must be inside the scale pyramid of the image
		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();
		const Vec3D PO = Xw - Ow;
		const float dist3D = static_cast<float>(cv::norm(PO));

		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		const Vec3D Pn = mappoint->GetNormal();
		if (PO.dot(Pn) < 0.5 * dist3D)
			continue;

		// Compute predicted scale level
		const int predictedScale = mappoint->PredictScale(dist3D, keyframe);

		// Search in a radius
		const float radius = th*keyframe->pyramid.scaleFactors[predictedScale];

		const std::vector<size_t> indices = keyframe->GetFeaturesInArea(u, v, radius);
		if (indices.empty())
			continue;

		// Match to the most similar keypoint in the radius

		const cv::Mat desc1 = mappoint->GetDescriptor();

		int bestDist = std::numeric_limits<int>::max();
		int bestIdx = -1;
		for (size_t idx : indices)
		{
			const int scale = keyframe->keypointsUn[idx].octave;
			if (scale < predictedScale - 1 || scale > predictedScale)
				continue;

			const cv::Mat &desc2 = keyframe->descriptorsL.row(static_cast<int>(idx));
			int dist = DescriptorDistance(desc1, desc2);
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = static_cast<int>(idx);
			}
		}

		// If there is already a MapPoint replace otherwise add new measurement
		if (bestDist <= TH_LOW)
		{
			MapPoint* MPInKF = keyframe->GetMapPoint(bestIdx);
			if (MPInKF)
			{
				if (!MPInKF->isBad())
					replacePoints[i] = MPInKF;
			}
			else
			{
				mappoint->AddObservation(keyframe, bestIdx);
				keyframe->AddMapPoint(mappoint, bestIdx);
			}
			nfused++;
		}
	}

	return nfused;
}

int ORBmatcher::SearchBySim3(KeyFrame* keyframe1, KeyFrame* keyframe2, std::vector<MapPoint*>& matches12,
	const Sim3& S12, float th)
{
	// Camera 1 from world
	const CameraProjection proj1(keyframe1->GetPose(), keyframe1->camera);

	//Camera 2 from world
	const CameraProjection proj2(keyframe2->GetPose(), keyframe2->camera);

	//Transformation between cameras
	const Sim3 S21 = S12.Inverse();

	const std::vector<MapPoint*> mappoints1 = keyframe1->GetMapPointMatches();
	const std::vector<MapPoint*> mappoints2 = keyframe2->GetMapPointMatches();

	const int N1 = static_cast<int>(mappoints1.size());
	const int N2 = static_cast<int>(mappoints2.size());

	std::vector<bool> alreadyMatched1(N1, false);
	std::vector<bool> alreadyMatched2(N2, false);

	for (int i = 0; i < N1; i++)
	{
		MapPoint* mappoint = matches12[i];
		if (mappoint)
		{
			alreadyMatched1[i] = true;
			const int idx2 = mappoint->GetIndexInKeyFrame(keyframe2);
			if (idx2 >= 0 && idx2 < N2)
				alreadyMatched2[idx2] = true;
		}
	}

	std::vector<int> match1(N1, -1);
	std::vector<int> match2(N2, -1);

	// Transform from KF1 to KF2 and search
	for (int i1 = 0; i1 < N1; i1++)
	{
		MapPoint* mappoint1 = mappoints1[i1];
		if (!mappoint1 || alreadyMatched1[i1] || mappoint1->isBad())
			continue;

		const Point3D Xw1 = mappoint1->GetWorldPos();
		const Point3D Xc1 = proj1.WorldToCamera(Xw1);
		const Point3D Xc2 = S21.Map(Xc1);

		// Depth must be positive
		if (Xc2(2) < 0.f)
			continue;

		const Point2D pt = proj2.CameraToImage(Xc2);
		const float u = pt.x;
		const float v = pt.y;

		// Point must be inside the image
		if (!keyframe2->IsInImage(u, v))
			continue;

		const float maxDistance = mappoint1->GetMaxDistanceInvariance();
		const float minDistance = mappoint1->GetMinDistanceInvariance();
		const float dist3D = static_cast<float>(cv::norm(Xc2));

		// Depth must be inside the scale invariance region
		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		// Compute predicted octave
		const int predictedScale = mappoint1->PredictScale(dist3D, keyframe2);

		// Search in a radius
		const float radius = th*keyframe2->pyramid.scaleFactors[predictedScale];

		const std::vector<size_t> indices = keyframe2->GetFeaturesInArea(u, v, radius);
		if (indices.empty())
			continue;

		// Match to the most similar keypoint in the radius
		const cv::Mat desc1 = mappoint1->GetDescriptor();

		int bestDist = std::numeric_limits<int>::max();
		int bestIdx = -1;
		for (size_t idx : indices)
		{
			const cv::KeyPoint& keypoint2 = keyframe2->keypointsUn[idx];
			if (keypoint2.octave < predictedScale - 1 || keypoint2.octave > predictedScale)
				continue;

			const cv::Mat desc2 = keyframe2->descriptorsL.row(static_cast<int>(idx));
			const int dist = DescriptorDistance(desc1, desc2);
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = static_cast<int>(idx);
			}
		}

		if (bestDist <= TH_HIGH)
		{
			match1[i1] = bestIdx;
		}
	}

	// Transform from KF2 to KF1 and search
	for (int i2 = 0; i2 < N2; i2++)
	{
		MapPoint* mappoint2 = mappoints2[i2];
		if (!mappoint2 || alreadyMatched2[i2] || mappoint2->isBad())
			continue;

		const Point3D Xw2 = mappoint2->GetWorldPos();
		const Point3D Xc2 = proj2.WorldToCamera(Xw2);
		const Point3D Xc1 = S12.Map(Xc2);

		// Depth must be positive
		if (Xc1(2) < 0.f)
			continue;

		const Point2D pt = proj1.CameraToImage(Xc1);
		const float u = pt.x;
		const float v = pt.y;

		// Point must be inside the image
		if (!keyframe1->IsInImage(u, v))
			continue;

		const float maxDistance = mappoint2->GetMaxDistanceInvariance();
		const float minDistance = mappoint2->GetMinDistanceInvariance();
		const float dist3D = static_cast<float>(cv::norm(Xc1));

		// Depth must be inside the scale pyramid of the image
		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		// Compute predicted octave
		const int predictedScale = mappoint2->PredictScale(dist3D, keyframe1);

		// Search in a radius of 2.5*sigma(ScaleLevel)
		const float radius = th * keyframe1->pyramid.scaleFactors[predictedScale];

		const std::vector<size_t> indices = keyframe1->GetFeaturesInArea(u, v, radius);
		if (indices.empty())
			continue;

		// Match to the most similar keypoint in the radius
		const cv::Mat desc2 = mappoint2->GetDescriptor();

		int bestDist = std::numeric_limits<int>::max();
		int bestIdx = -1;
		for (size_t idx : indices)
		{
			const cv::KeyPoint& keypoints1 = keyframe1->keypointsUn[idx];
			if (keypoints1.octave < predictedScale - 1 || keypoints1.octave > predictedScale)
				continue;

			const cv::Mat desc1 = keyframe1->descriptorsL.row(static_cast<int>(idx));
			const int dist = DescriptorDistance(desc2, desc1);
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = static_cast<int>(idx);
			}
		}

		if (bestDist <= TH_HIGH)
		{
			match2[i2] = bestIdx;
		}
	}

	// Check agreement
	int nfound = 0;
	for (int i1 = 0; i1 < N1; i1++)
	{
		const int idx2 = match1[i1];
		if (idx2 >= 0)
		{
			const int idx1 = match2[idx2];
			if (idx1 == i1)
			{
				matches12[i1] = mappoints2[idx2];
				nfound++;
			}
		}
	}

	return nfound;
}

int ORBmatcher::SearchByProjection(Frame& currFrame, const Frame& lastFrame, float th, bool monocular)
{
	int nmatches = 0;

	const CameraParams& camera = currFrame.camera;
	const CameraProjection proj(currFrame.pose, camera);

	const auto tlc = lastFrame.pose.R() * currFrame.pose.Invt() + lastFrame.pose.t();
	const bool forward = tlc(2) > camera.baseline && !monocular;
	const bool backward = -tlc(2) > camera.baseline && !monocular;

	std::vector<MatchIdx> matchIds;
	matchIds.reserve(lastFrame.N);

	for (int idx1 = 0; idx1 < lastFrame.N; idx1++)
	{
		MapPoint* mappoint1 = lastFrame.mappoints[idx1];
		if (!mappoint1 || lastFrame.outlier[idx1])
			continue;

		// Project
		const Point3D Xw = mappoint1->GetWorldPos();
		const Point3D Xc = proj.WorldToCamera(Xw);
		if (Xc(2) < 0.f)
			continue;

		const Point2D pt = proj.CameraToImage(Xc);
		const float u = pt.x;
		const float v = pt.y;
		const float ur = u - proj.DepthToDisparity(Xc(2));

		if (!currFrame.imageBounds.Contains(u, v))
			continue;

		const int octave1 = lastFrame.keypoints[idx1].octave;

		// Search in a window. Size depends on scale
		const float radius = th*currFrame.pyramid.scaleFactors[octave1];

		const int minLevel = forward ? octave1 : (backward ? 0       : octave1 - 1);
		const int maxLevel = forward ? -1      : (backward ? octave1 : octave1 + 1);

		const std::vector<size_t> indices2 = currFrame.GetFeaturesInArea(u, v, radius, minLevel, maxLevel);
		if (indices2.empty())
			continue;

		const cv::Mat desc1 = mappoint1->GetDescriptor();

		int bestDist = 256;
		int bestIdx2 = -1;
		for (size_t idx2 : indices2)
		{
			MapPoint* mappoint2 = currFrame.mappoints[idx2];
			if (mappoint2 && mappoint2->Observations() > 0)
				continue;

			if (currFrame.uright[idx2] > 0 && fabsf(ur - currFrame.uright[idx2]) > radius)
				continue;

			const cv::Mat desc2 = currFrame.descriptors.row(static_cast<int>(idx2));
			const int dist = DescriptorDistance(desc1, desc2);
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx2 = static_cast<int>(idx2);
			}
		}

		if (bestDist <= TH_HIGH)
		{
			currFrame.mappoints[bestIdx2] = mappoint1;
			nmatches++;

			if (checkOrientation_)
				matchIds.push_back(std::make_pair(idx1, bestIdx2));
		}
	}

	// Apply rotation consistency
	if (checkOrientation_)
		nmatches = CheckOrientation(lastFrame.keypointsUn, currFrame.keypointsUn, matchIds, currFrame.mappoints);

	return nmatches;
}

int ORBmatcher::SearchByProjection(Frame& frame, KeyFrame* keyframe, const std::set<MapPoint*>& alreadyFound,
	float th, int ORBdist)
{
	int nmatches = 0;

	const CameraProjection proj(frame.pose, frame.camera);
	const Point3D Ow = frame.GetCameraCenter();

	const std::vector<MapPoint*> mappoints = keyframe->GetMapPointMatches();

	std::vector<MatchIdx> matchIds;
	matchIds.reserve(mappoints.size());

	for (size_t idx1 = 0; idx1 < mappoints.size(); idx1++)
	{
		MapPoint* mappoint = mappoints[idx1];
		if (!mappoint || mappoint->isBad() || alreadyFound.count(mappoint))
			continue;

		//Project
		const Point3D Xw = mappoint->GetWorldPos();
		const Point2D pt = proj.WorldToImage(Xw);
		const float u = pt.x;
		const float v = pt.y;

		if (!frame.imageBounds.Contains(u, v))
			continue;

		// Compute predicted scale level
		const Vec3D PO = Xw - Ow;
		const float dist3D = static_cast<float>(cv::norm(PO));

		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();

		// Depth must be inside the scale pyramid of the image
		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		const int predictedScale = mappoint->PredictScale(dist3D, &frame);

		// Search in a window
		const float radius = th * frame.pyramid.scaleFactors[predictedScale];

		const std::vector<size_t> indices = frame.GetFeaturesInArea(u, v, radius, predictedScale - 1, predictedScale + 1);
		if (indices.empty())
			continue;

		const cv::Mat desc1 = mappoint->GetDescriptor();

		int bestDist = 256;
		int bestIdx2 = -1;

		for (size_t idx2 : indices)
		{
			if (frame.mappoints[idx2])
				continue;

			const cv::Mat desc2 = frame.descriptors.row(static_cast<int>(idx2));
			const int dist = DescriptorDistance(desc1, desc2);
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx2 = static_cast<int>(idx2);
			}
		}

		if (bestDist <= ORBdist)
		{
			frame.mappoints[bestIdx2] = mappoint;
			nmatches++;

			if (checkOrientation_)
				matchIds.push_back(std::make_pair(static_cast<int>(idx1), bestIdx2));
		}
	}

	if (checkOrientation_)
		nmatches = CheckOrientation(keyframe->keypointsUn, frame.keypointsUn, matchIds, frame.mappoints);

	return nmatches;
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
	const int* ptra = a.ptr<int32_t>();
	const int* ptrb = b.ptr<int32_t>();
	int dist = 0;
	for (int i = 0; i < 8; i++)
		dist += static_cast<int>(popcnt32(*ptra++ ^ *ptrb++));
	return dist;
}

} //namespace ORB_SLAM
