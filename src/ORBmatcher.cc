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

#include "ORBmatcher.h"
#include "CameraPose.h"

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

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
static const int HISTO_LENGTH = 30;

static inline float RadiusByViewingCos(float viewCos)
{
	return viewCos > 0.998 ? 2.5f : 4.f;
}

using MatchIdx = std::pair<int, int>;

template <typename T> static inline T InvalidMatch() { return 0; }
template <> static inline MapPoint* InvalidMatch<MapPoint*>() { return nullptr; }
template <> static inline int InvalidMatch<int>() { return -1; }

template <typename T>
static int CheckOrientation(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
	const std::vector<MatchIdx>& matchIds, std::vector<T>& matchStatus)
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
		if (!mappoint->mbTrackInView || mappoint->isBad())
			continue;

		const int predictedScale = mappoint->mnTrackScaleLevel;

		// The size of the window will depend on the viewing direction
		const float r = RadiusByViewingCos(mappoint->mTrackViewCos);
		const float radius = th * r * frame.pyramid.scaleFactors[predictedScale];
		const float u = mappoint->mTrackProjX;
		const float v = mappoint->mTrackProjY;

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

			if (frame.uright[idx] > 0)
			{
				if (fabsf(mappoint->mTrackProjXR - frame.uright[idx]) > radius)
					continue;
			}

			const cv::Mat desc2 = frame.descriptorsL.row(static_cast<int>(idx));
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
	const vector<MapPoint*> mappoints1 = keyframe->GetMapPointMatches();

	matches.assign(frame.N, nullptr);

	int nmatches = 0;

	vector<int> rotHist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);
	const float factor = 1.0f / HISTO_LENGTH;

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

				const cv::Mat desc2 = frame.descriptorsL.row(idx2);
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

int ORBmatcher::SearchByProjection(const KeyFrame* keyframe, const cv::Mat& Scw, const std::vector<MapPoint*>& mappoints,
	std::vector<MapPoint*>& matched, int th)
{
	// Get Calibration Parameters for later projection
	const float fx = keyframe->camera.fx;
	const float fy = keyframe->camera.fy;
	const float cx = keyframe->camera.cx;
	const float cy = keyframe->camera.cy;

	// Decompose Scw
	const cv::Mat sRcw = GetR(Scw);
	const float scale = static_cast<float>(sqrt(sRcw.row(0).dot(sRcw.row(0))));
	const cv::Mat Rcw = sRcw / scale;
	const cv::Mat tcw = Gett(Scw) / scale;
	const cv::Mat Ow = -Rcw.t() * tcw;

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
		const cv::Mat Xw = mappoint->GetWorldPos();

		// Transform into Camera Coords.
		const cv::Mat Xc = Rcw * Xw + tcw;

		// Depth must be positive
		if (Xc.at<float>(2) < 0.f)
			continue;

		// Project into Image
		const float invZ = 1 / Xc.at<float>(2);
		const float u = fx * Xc.at<float>(0) * invZ + cx;
		const float v = fy * Xc.at<float>(1) * invZ + cy;

		// Point must be inside the image
		if (!keyframe->IsInImage(u, v))
			continue;

		// Depth must be inside the scale invariance region of the point
		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();
		cv::Mat PO = Xw - Ow;
		const float dist = static_cast<float>(cv::norm(PO));
		if (dist < minDistance || dist > maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		cv::Mat Pn = mappoint->GetNormal();
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

		const cv::Mat desc1 = frame1.descriptorsL.row(static_cast<int>(idx1));

		int bestDist = std::numeric_limits<int>::max();
		int secondBestDist = std::numeric_limits<int>::max();
		int bestIdx2 = -1;

		for (size_t idx2 : indices2)
		{
			const cv::Mat desc2 = frame2.descriptorsL.row(static_cast<int>(idx2));
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
	const std::vector<cv::KeyPoint>& keypoints1 = keyframe1->keypointsUn;
	const std::vector<cv::KeyPoint>& keypoints2 = keyframe2->keypointsUn;
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
	const cv::Mat Cw = keyframe1->GetCameraCenter();
	const cv::Mat R2w = cv::Mat(keyframe2->GetPose().R());
	const cv::Mat t2w = cv::Mat(keyframe2->GetPose().t());
	const cv::Mat C2 = R2w * Cw + t2w;
	const float invZ = 1.f / C2.at<float>(2);
	const float epx = keyframe2->camera.fx * C2.at<float>(0) * invZ + keyframe2->camera.cx;
	const float epy = keyframe2->camera.fy * C2.at<float>(1) * invZ + keyframe2->camera.cy;

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
					const float dx = epx - keypoint2.pt.x;
					const float dy = epy - keypoint2.pt.y;
					if (dx * dx + dy * dy < 100 * keyframe2->pyramid.scaleFactors[keypoint2.octave])
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
	cv::Mat Rcw = cv::Mat(keyframe->GetPose().R());
	cv::Mat tcw = cv::Mat(keyframe->GetPose().t());

	const float fx = keyframe->camera.fx;
	const float fy = keyframe->camera.fy;
	const float cx = keyframe->camera.cx;
	const float cy = keyframe->camera.cy;
	const float bf = keyframe->camera.bf;

	const cv::Mat Ow = keyframe->GetCameraCenter();
	int nfused = 0;

	for (MapPoint* mappoint : mappoints)
	{
		if (!mappoint || mappoint->isBad() || mappoint->IsInKeyFrame(keyframe))
			continue;

		const cv::Mat Xw = mappoint->GetWorldPos();
		const cv::Mat Xc = Rcw * Xw + tcw;

		// Depth must be positive
		if (Xc.at<float>(2) < 0.f)
			continue;

		const float invZ = 1 / Xc.at<float>(2);
		const float u = fx * Xc.at<float>(0) * invZ + cx;
		const float v = fy * Xc.at<float>(1) * invZ + cy;

		// Point must be inside the image
		if (!keyframe->IsInImage(u, v))
			continue;

		const float ur = u - bf * invZ;

		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();
		const cv::Mat PO = Xw - Ow;
		const float dist3D = static_cast<float>(cv::norm(PO));

		// Depth must be inside the scale pyramid of the image
		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		const cv::Mat Pn = mappoint->GetNormal();
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

			if (keyframe->uright[idx] >= 0)
			{
				// Check reprojection error in stereo
				const float dx = u - keypoint.pt.x;
				const float dy = v - keypoint.pt.y;
				const float dz = ur - keyframe->uright[idx];
				const float errorSq = dx * dx + dy * dy + dz * dz;
				if (errorSq * keyframe->pyramid.invSigmaSq[scale] > 7.8)
					continue;
			}
			else
			{
				const float dx = u - keypoint.pt.x;
				const float dy = v - keypoint.pt.y;
				const float errorSq = dx * dx + dy * dy;
				if (errorSq * keyframe->pyramid.invSigmaSq[scale] > 5.99)
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

int ORBmatcher::Fuse(KeyFrame* keyframe, const cv::Mat& Scw, const std::vector<MapPoint*>& mappoints,
	float th, std::vector<MapPoint*>& replacePoints)
{
	// Get Calibration Parameters for later projection
	const float fx = keyframe->camera.fx;
	const float fy = keyframe->camera.fy;
	const float cx = keyframe->camera.cx;
	const float cy = keyframe->camera.cy;

	// Decompose Scw
	const cv::Mat sRcw = GetR(Scw);
	const float scale = static_cast<float>(sqrt(sRcw.row(0).dot(sRcw.row(0))));
	const cv::Mat Rcw = sRcw / scale;
	const cv::Mat tcw = Gett(Scw) / scale;
	const cv::Mat Ow = -Rcw.t() * tcw;

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
		const cv::Mat Xw = mappoint->GetWorldPos();

		// Transform into Camera Coords.
		const cv::Mat Xc = Rcw * Xw + tcw;

		// Depth must be positive
		if (Xc.at<float>(2) < 0.f)
			continue;

		// Project into Image
		const float invZ = 1.f / Xc.at<float>(2);
		const float u = fx * Xc.at<float>(0) * invZ + cx;
		const float v = fy * Xc.at<float>(1) * invZ + cy;

		// Point must be inside the image
		if (!keyframe->IsInImage(u, v))
			continue;

		// Depth must be inside the scale pyramid of the image
		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();
		const cv::Mat PO = Xw - Ow;
		const float dist3D = static_cast<float>(cv::norm(PO));

		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		const cv::Mat Pn = mappoint->GetNormal();
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
	float s12, const cv::Mat &R12, const cv::Mat &t12, float th)
{
	const float fx = keyframe1->camera.fx;
	const float fy = keyframe1->camera.fy;
	const float cx = keyframe1->camera.cx;
	const float cy = keyframe1->camera.cy;

	// Camera 1 from world
	const cv::Mat R1w = cv::Mat(keyframe1->GetPose().R());
	const cv::Mat t1w = cv::Mat(keyframe1->GetPose().t());

	//Camera 2 from world
	const cv::Mat R2w = cv::Mat(keyframe2->GetPose().R());
	const cv::Mat t2w = cv::Mat(keyframe2->GetPose().t());

	//Transformation between cameras
	cv::Mat sR12 = s12 * R12;
	cv::Mat sR21 = (1.0 / s12) * R12.t();
	cv::Mat t21 = -sR21 * t12;

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

		const cv::Mat Xw1 = mappoint1->GetWorldPos();
		const cv::Mat Xc1 = R1w * Xw1 + t1w;
		const cv::Mat Xc2 = sR21 * Xc1 + t21;

		// Depth must be positive
		if (Xc2.at<float>(2) < 0.f)
			continue;

		const float invZ = 1.f / Xc2.at<float>(2);
		const float u = fx * Xc2.at<float>(0) * invZ + cx;
		const float v = fy * Xc2.at<float>(1) * invZ + cy;

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
		//for (vector<size_t>::const_iterator vit = indices.begin(), vend = indices.end(); vit != vend; vit++)
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

	// Transform from KF2 to KF2 and search
	for (int i2 = 0; i2 < N2; i2++)
	{
		MapPoint* mappoint2 = mappoints2[i2];
		if (!mappoint2 || alreadyMatched2[i2] || mappoint2->isBad())
			continue;

		const cv::Mat Xw2 = mappoint2->GetWorldPos();
		const cv::Mat Xc2 = R2w * Xw2 + t2w;
		const cv::Mat Xc1 = sR12 * Xc2 + t12;

		// Depth must be positive
		if (Xc1.at<float>(2) < 0.f)
			continue;

		const float invZ = 1.f / Xc1.at<float>(2);
		const float u = fx * Xc1.at<float>(0) * invZ + cx;
		const float v = fy * Xc1.at<float>(1) * invZ + cy;

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

	const cv::Mat Rcw = cv::Mat(currFrame.pose.R());
	const cv::Mat tcw = cv::Mat(currFrame.pose.t());

	const cv::Mat Rlw = cv::Mat(lastFrame.pose.R());
	const cv::Mat tlw = cv::Mat(lastFrame.pose.t());

	const cv::Mat twc = -Rcw.t() * tcw;
	const cv::Mat tlc = Rlw * twc + tlw;

	const CameraParams& camera = currFrame.camera;

	const bool forward = tlc.at<float>(2) > camera.baseline && !monocular;
	const bool backward = -tlc.at<float>(2) > camera.baseline && !monocular;

	std::vector<MatchIdx> matchIds;
	matchIds.reserve(lastFrame.N);

	for (int idx1 = 0; idx1 < lastFrame.N; idx1++)
	{
		MapPoint* mappoint1 = lastFrame.mappoints[idx1];
		if (!mappoint1 || lastFrame.outlier[idx1])
			continue;

		// Project
		cv::Mat x3Dw = mappoint1->GetWorldPos();
		cv::Mat x3Dc = Rcw * x3Dw + tcw;

		const float xc = x3Dc.at<float>(0);
		const float yc = x3Dc.at<float>(1);
		const float invZc = 1.f / x3Dc.at<float>(2);

		if (invZc < 0)
			continue;

		const float u = camera.fx * xc * invZc + camera.cx;
		const float v = camera.fy * yc * invZc + camera.cy;

		if (!currFrame.imageBounds.Contains(u, v))
			continue;

		const int octave1 = lastFrame.keypointsL[idx1].octave;

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

			if (currFrame.uright[idx2] > 0)
			{
				const float ur = u - camera.bf * invZc;
				if (fabsf(ur - currFrame.uright[idx2]) > radius)
					continue;
			}

			const cv::Mat desc2 = currFrame.descriptorsL.row(static_cast<int>(idx2));
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

	const cv::Mat Rcw = cv::Mat(frame.pose.R());
	const cv::Mat tcw = cv::Mat(frame.pose.t());
	const cv::Mat Ow = -Rcw.t() * tcw;

	const float fx = frame.camera.fx;
	const float fy = frame.camera.fy;
	const float cx = frame.camera.cx;
	const float cy = frame.camera.cy;

	// Rotation Histogram (to check rotation consistency)
	vector<int> rotHist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);
	const float factor = 1.0f / HISTO_LENGTH;

	const std::vector<MapPoint*> mappoints = keyframe->GetMapPointMatches();

	std::vector<MatchIdx> matchIds;
	matchIds.reserve(mappoints.size());

	for (size_t idx1 = 0; idx1 < mappoints.size(); idx1++)
	{
		MapPoint* mappoint = mappoints[idx1];
		if (!mappoint || mappoint->isBad() || alreadyFound.count(mappoint))
			continue;

		//Project
		const cv::Mat Xw = mappoint->GetWorldPos();
		const cv::Mat Xc = Rcw * Xw + tcw;

		const float invZc = 1.f / Xc.at<float>(2);
		const float u = fx * Xc.at<float>(0) * invZc + cx;
		const float v = fy * Xc.at<float>(1) * invZc + cy;

		if (!frame.imageBounds.Contains(u, v))
			continue;

		// Compute predicted scale level
		const cv::Mat PO = Xw - Ow;
		const float dist3D = static_cast<float>(cv::norm(PO));

		const float maxDistance = mappoint->GetMaxDistanceInvariance();
		const float minDistance = mappoint->GetMinDistanceInvariance();

		// Depth must be inside the scale pyramid of the image
		if (dist3D < minDistance || dist3D > maxDistance)
			continue;

		const int predictedScale = mappoint->PredictScale(dist3D, &frame);

		// Search in a window
		const float radius = th * frame.pyramid.scaleFactors[predictedScale];

		const std::vector<size_t> indices =  frame.GetFeaturesInArea(u, v, radius, predictedScale - 1, predictedScale + 1);
		if (indices.empty())
			continue;

		const cv::Mat desc1 = mappoint->GetDescriptor();

		int bestDist = 256;
		int bestIdx2 = -1;

		for (size_t idx2 : indices)
		{
			if (frame.mappoints[idx2])
				continue;

			const cv::Mat desc2 = frame.descriptorsL.row(static_cast<int>(idx2));
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
