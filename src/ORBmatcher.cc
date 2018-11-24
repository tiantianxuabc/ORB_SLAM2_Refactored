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

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<cstdint>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
static const int HISTO_LENGTH = 30;

static float RadiusByViewingCos(const float &viewCos)
{
	if (viewCos > 0.998)
		return 2.5;
	else
		return 4.0;
}

static void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
	int max1 = 0;
	int max2 = 0;
	int max3 = 0;

	for (int i = 0; i < L; i++)
	{
		const int s = histo[i].size();
		if (s > max1)
		{
			max3 = max2;
			max2 = max1;
			max1 = s;
			ind3 = ind2;
			ind2 = ind1;
			ind1 = i;
		}
		else if (s > max2)
		{
			max3 = max2;
			max2 = s;
			ind3 = ind2;
			ind2 = i;
		}
		else if (s > max3)
		{
			max3 = s;
			ind3 = i;
		}
	}

	if (max2 < 0.1f*(float)max1)
	{
		ind2 = -1;
		ind3 = -1;
	}
	else if (max3 < 0.1f*(float)max1)
	{
		ind3 = -1;
	}
}

using MatchIdx = std::pair<int, int>;
static int CheckOrientation(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
	const std::vector<MatchIdx>& matches12, std::vector<MapPoint*>& mappoints2)
{
	CV_Assert(mappoints2.size() == keypoints2.size());

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

	for (const auto& match : matches12)
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
			mappoints2[i2] = nullptr;
			reduction++;
		}
	}

	return static_cast<int>(matches12.size() - reduction);
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

			const cv::Mat desc2 = frame.descriptorsL.row(idx);
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

static bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame* pKF2)
{
	// Epipolar line in second image l = x1'F12 = [a b c]
	const float a = kp1.pt.x*F12.at<float>(0, 0) + kp1.pt.y*F12.at<float>(1, 0) + F12.at<float>(2, 0);
	const float b = kp1.pt.x*F12.at<float>(0, 1) + kp1.pt.y*F12.at<float>(1, 1) + F12.at<float>(2, 1);
	const float c = kp1.pt.x*F12.at<float>(0, 2) + kp1.pt.y*F12.at<float>(1, 2) + F12.at<float>(2, 2);

	const float num = a*kp2.pt.x + b*kp2.pt.y + c;

	const float den = a*a + b*b;

	if (den == 0)
		return false;

	const float dsqr = num*num / den;

	return dsqr < 3.84*pKF2->pyramid.sigmaSq[kp2.octave];
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

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
	// Get Calibration Parameters for later projection
	const float &fx = pKF->camera.fx;
	const float &fy = pKF->camera.fy;
	const float &cx = pKF->camera.cx;
	const float &cy = pKF->camera.cy;

	// Decompose Scw
	cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
	const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
	cv::Mat Rcw = sRcw / scw;
	cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
	cv::Mat Ow = -Rcw.t()*tcw;

	// Set of MapPoints already found in the KeyFrame
	set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
	spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

	int nmatches = 0;

	// For each Candidate MapPoint Project and Match
	for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
	{
		MapPoint* pMP = vpPoints[iMP];

		// Discard Bad MapPoints and already found
		if (pMP->isBad() || spAlreadyFound.count(pMP))
			continue;

		// Get 3D Coords.
		cv::Mat p3Dw = pMP->GetWorldPos();

		// Transform into Camera Coords.
		cv::Mat p3Dc = Rcw*p3Dw + tcw;

		// Depth must be positive
		if (p3Dc.at<float>(2) < 0.0)
			continue;

		// Project into Image
		const float invz = 1 / p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pKF->IsInImage(u, v))
			continue;

		// Depth must be inside the scale invariance region of the point
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw - Ow;
		const float dist = cv::norm(PO);

		if (dist<minDistance || dist>maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		cv::Mat Pn = pMP->GetNormal();

		if (PO.dot(Pn) < 0.5*dist)
			continue;

		int nPredictedLevel = pMP->PredictScale(dist, pKF);

		// Search in a radius
		const float radius = th*pKF->pyramid.scaleFactors[nPredictedLevel];

		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;
			if (vpMatched[idx])
				continue;

			const int &kpLevel = pKF->keypointsUn[idx].octave;

			if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
				continue;

			const cv::Mat &dKF = pKF->descriptorsL.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);

			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}

		if (bestDist <= TH_LOW)
		{
			vpMatched[bestIdx] = pMP;
			nmatches++;
		}

	}

	return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
	int nmatches = 0;
	vnMatches12 = vector<int>(F1.keypointsUn.size(), -1);

	vector<int> rotHist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);
	const float factor = 1.0f / HISTO_LENGTH;

	vector<int> vMatchedDistance(F2.keypointsUn.size(), INT_MAX);
	vector<int> vnMatches21(F2.keypointsUn.size(), -1);

	for (size_t i1 = 0, iend1 = F1.keypointsUn.size(); i1 < iend1; i1++)
	{
		cv::KeyPoint kp1 = F1.keypointsUn[i1];
		int level1 = kp1.octave;
		if (level1 > 0)
			continue;

		vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1);

		if (vIndices2.empty())
			continue;

		cv::Mat d1 = F1.descriptorsL.row(i1);

		int bestDist = INT_MAX;
		int bestDist2 = INT_MAX;
		int bestIdx2 = -1;

		for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
		{
			size_t i2 = *vit;

			cv::Mat d2 = F2.descriptorsL.row(i2);

			int dist = DescriptorDistance(d1, d2);

			if (vMatchedDistance[i2] <= dist)
				continue;

			if (dist < bestDist)
			{
				bestDist2 = bestDist;
				bestDist = dist;
				bestIdx2 = i2;
			}
			else if (dist < bestDist2)
			{
				bestDist2 = dist;
			}
		}

		if (bestDist <= TH_LOW)
		{
			if (bestDist < (float)bestDist2*fNNRatio_)
			{
				if (vnMatches21[bestIdx2] >= 0)
				{
					vnMatches12[vnMatches21[bestIdx2]] = -1;
					nmatches--;
				}
				vnMatches12[i1] = bestIdx2;
				vnMatches21[bestIdx2] = i1;
				vMatchedDistance[bestIdx2] = bestDist;
				nmatches++;

				if (checkOrientation_)
				{
					float rot = F1.keypointsUn[i1].angle - F2.keypointsUn[bestIdx2].angle;
					if (rot < 0.0)
						rot += 360.0f;
					int bin = round(rot*factor);
					if (bin == HISTO_LENGTH)
						bin = 0;
					assert(bin >= 0 && bin < HISTO_LENGTH);
					rotHist[bin].push_back(i1);
				}
			}
		}

	}

	if (checkOrientation_)
	{
		int ind1 = -1;
		int ind2 = -1;
		int ind3 = -1;

		ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

		for (int i = 0; i < HISTO_LENGTH; i++)
		{
			if (i == ind1 || i == ind2 || i == ind3)
				continue;
			for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
			{
				int idx1 = rotHist[i][j];
				if (vnMatches12[idx1] >= 0)
				{
					vnMatches12[idx1] = -1;
					nmatches--;
				}
			}
		}

	}

	//Update prev matched
	for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
		if (vnMatches12[i1] >= 0)
			vbPrevMatched[i1] = F2.keypointsUn[vnMatches12[i1]].pt;

	return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
	const vector<cv::KeyPoint> &vKeysUn1 = pKF1->keypointsUn;
	const DBoW2::FeatureVector &vFeatVec1 = pKF1->featureVector;
	const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	const cv::Mat &Descriptors1 = pKF1->descriptorsL;

	const vector<cv::KeyPoint> &vKeysUn2 = pKF2->keypointsUn;
	const DBoW2::FeatureVector &vFeatVec2 = pKF2->featureVector;
	const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
	const cv::Mat &Descriptors2 = pKF2->descriptorsL;

	vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(), static_cast<MapPoint*>(NULL));
	vector<bool> vbMatched2(vpMapPoints2.size(), false);

	vector<int> rotHist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);

	const float factor = 1.0f / HISTO_LENGTH;

	int nmatches = 0;

	DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
	DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
	DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
	DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

	while (f1it != f1end && f2it != f2end)
	{
		if (f1it->first == f2it->first)
		{
			for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
			{
				const size_t idx1 = f1it->second[i1];

				MapPoint* pMP1 = vpMapPoints1[idx1];
				if (!pMP1)
					continue;
				if (pMP1->isBad())
					continue;

				const cv::Mat &d1 = Descriptors1.row(idx1);

				int bestDist1 = 256;
				int bestIdx2 = -1;
				int bestDist2 = 256;

				for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
				{
					const size_t idx2 = f2it->second[i2];

					MapPoint* pMP2 = vpMapPoints2[idx2];

					if (vbMatched2[idx2] || !pMP2)
						continue;

					if (pMP2->isBad())
						continue;

					const cv::Mat &d2 = Descriptors2.row(idx2);

					int dist = DescriptorDistance(d1, d2);

					if (dist < bestDist1)
					{
						bestDist2 = bestDist1;
						bestDist1 = dist;
						bestIdx2 = idx2;
					}
					else if (dist < bestDist2)
					{
						bestDist2 = dist;
					}
				}

				if (bestDist1 < TH_LOW)
				{
					if (static_cast<float>(bestDist1) < fNNRatio_*static_cast<float>(bestDist2))
					{
						vpMatches12[idx1] = vpMapPoints2[bestIdx2];
						vbMatched2[bestIdx2] = true;

						if (checkOrientation_)
						{
							float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
							if (rot < 0.0)
								rot += 360.0f;
							int bin = round(rot*factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin < HISTO_LENGTH);
							rotHist[bin].push_back(idx1);
						}
						nmatches++;
					}
				}
			}

			f1it++;
			f2it++;
		}
		else if (f1it->first < f2it->first)
		{
			f1it = vFeatVec1.lower_bound(f2it->first);
		}
		else
		{
			f2it = vFeatVec2.lower_bound(f1it->first);
		}
	}

	if (checkOrientation_)
	{
		int ind1 = -1;
		int ind2 = -1;
		int ind3 = -1;

		ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

		for (int i = 0; i < HISTO_LENGTH; i++)
		{
			if (i == ind1 || i == ind2 || i == ind3)
				continue;
			for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
			{
				vpMatches12[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
				nmatches--;
			}
		}
	}

	return nmatches;
}

int ORBmatcher::SearchForTriangulation(const KeyFrame* keyframe1, const KeyFrame* keyframe2, const cv::Mat& F12,
	std::vector<std::pair<size_t, size_t>>& matchIds, bool onlyStereo)
{
	//Compute epipole in second image
	cv::Mat Cw = keyframe1->GetCameraCenter();
	cv::Mat R2w = keyframe2->GetRotation();
	cv::Mat t2w = keyframe2->GetTranslation();
	cv::Mat C2 = R2w*Cw + t2w;
	const float invz = 1.0f / C2.at<float>(2);
	const float ex = keyframe2->camera.fx*C2.at<float>(0)*invz + keyframe2->camera.cx;
	const float ey = keyframe2->camera.fy*C2.at<float>(1)*invz + keyframe2->camera.cy;

	// Find matches between not tracked keypoints
	// Matching speed-up by ORB Vocabulary
	// Compare only ORB that share the same node

	int nmatches = 0;
	vector<bool> vbMatched2(keyframe2->N, false);
	vector<int> vMatches12(keyframe1->N, -1);

	vector<int> rotHist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);

	const float factor = 1.0f / HISTO_LENGTH;

	FeatureVectorIterator iterator(keyframe1->featureVector, keyframe2->featureVector);
	while (iterator.next())
	{
		const auto& indices1 = iterator.indices1();
		const auto& indices2 = iterator.indices2();
		for (size_t i1 = 0, iend1 = indices1.size(); i1 < iend1; i1++)
		{
			const size_t idx1 = indices1[i1];

			MapPoint* pMP1 = keyframe1->GetMapPoint(idx1);

			// If there is already a MapPoint skip
			if (pMP1)
				continue;

			const bool bStereo1 = keyframe1->uright[idx1] >= 0;

			if (onlyStereo)
				if (!bStereo1)
					continue;

			const cv::KeyPoint &kp1 = keyframe1->keypointsUn[idx1];

			const cv::Mat &d1 = keyframe1->descriptorsL.row(idx1);

			int bestDist = TH_LOW;
			int bestIdx2 = -1;

			for (size_t i2 = 0, iend2 = indices2.size(); i2 < iend2; i2++)
			{
				size_t idx2 = indices2[i2];

				MapPoint* pMP2 = keyframe2->GetMapPoint(idx2);

				// If we have already matched or there is a MapPoint skip
				if (vbMatched2[idx2] || pMP2)
					continue;

				const bool bStereo2 = keyframe2->uright[idx2] >= 0;

				if (onlyStereo)
					if (!bStereo2)
						continue;

				const cv::Mat &d2 = keyframe2->descriptorsL.row(idx2);

				const int dist = DescriptorDistance(d1, d2);

				if (dist > TH_LOW || dist > bestDist)
					continue;

				const cv::KeyPoint &kp2 = keyframe2->keypointsUn[idx2];

				if (!bStereo1 && !bStereo2)
				{
					const float distex = ex - kp2.pt.x;
					const float distey = ey - kp2.pt.y;
					if (distex*distex + distey*distey < 100 * keyframe2->pyramid.scaleFactors[kp2.octave])
						continue;
				}

				if (CheckDistEpipolarLine(kp1, kp2, F12, keyframe2))
				{
					bestIdx2 = idx2;
					bestDist = dist;
				}
			}

			if (bestIdx2 >= 0)
			{
				const cv::KeyPoint &kp2 = keyframe2->keypointsUn[bestIdx2];
				vMatches12[idx1] = bestIdx2;
				nmatches++;

				if (checkOrientation_)
				{
					float rot = kp1.angle - kp2.angle;
					if (rot < 0.0)
						rot += 360.0f;
					int bin = round(rot*factor);
					if (bin == HISTO_LENGTH)
						bin = 0;
					assert(bin >= 0 && bin < HISTO_LENGTH);
					rotHist[bin].push_back(idx1);
				}
			}
		}
	}

	if (checkOrientation_)
	{
		int ind1 = -1;
		int ind2 = -1;
		int ind3 = -1;

		ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

		for (int i = 0; i < HISTO_LENGTH; i++)
		{
			if (i == ind1 || i == ind2 || i == ind3)
				continue;
			for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
			{
				vMatches12[rotHist[i][j]] = -1;
				nmatches--;
			}
		}

	}

	matchIds.clear();
	matchIds.reserve(nmatches);

	for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
	{
		if (vMatches12[i] < 0)
			continue;
		matchIds.push_back(make_pair(i, vMatches12[i]));
	}

	return nmatches;
}

int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
	cv::Mat Rcw = pKF->GetRotation();
	cv::Mat tcw = pKF->GetTranslation();

	const float &fx = pKF->camera.fx;
	const float &fy = pKF->camera.fy;
	const float &cx = pKF->camera.cx;
	const float &cy = pKF->camera.cy;
	const float &bf = pKF->camera.bf;

	cv::Mat Ow = pKF->GetCameraCenter();

	int nFused = 0;

	const int nMPs = vpMapPoints.size();

	for (int i = 0; i < nMPs; i++)
	{
		MapPoint* pMP = vpMapPoints[i];

		if (!pMP)
			continue;

		if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
			continue;

		cv::Mat p3Dw = pMP->GetWorldPos();
		cv::Mat p3Dc = Rcw*p3Dw + tcw;

		// Depth must be positive
		if (p3Dc.at<float>(2) < 0.0f)
			continue;

		const float invz = 1 / p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pKF->IsInImage(u, v))
			continue;

		const float ur = u - bf*invz;

		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw - Ow;
		const float dist3D = cv::norm(PO);

		// Depth must be inside the scale pyramid of the image
		if (dist3D<minDistance || dist3D>maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		cv::Mat Pn = pMP->GetNormal();

		if (PO.dot(Pn) < 0.5*dist3D)
			continue;

		int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

		// Search in a radius
		const float radius = th*pKF->pyramid.scaleFactors[nPredictedLevel];

		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius

		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;

			const cv::KeyPoint &kp = pKF->keypointsUn[idx];

			const int &kpLevel = kp.octave;

			if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
				continue;

			if (pKF->uright[idx] >= 0)
			{
				// Check reprojection error in stereo
				const float &kpx = kp.pt.x;
				const float &kpy = kp.pt.y;
				const float &kpr = pKF->uright[idx];
				const float ex = u - kpx;
				const float ey = v - kpy;
				const float er = ur - kpr;
				const float e2 = ex*ex + ey*ey + er*er;

				if (e2*pKF->pyramid.invSigmaSq[kpLevel] > 7.8)
					continue;
			}
			else
			{
				const float &kpx = kp.pt.x;
				const float &kpy = kp.pt.y;
				const float ex = u - kpx;
				const float ey = v - kpy;
				const float e2 = ex*ex + ey*ey;

				if (e2*pKF->pyramid.invSigmaSq[kpLevel] > 5.99)
					continue;
			}

			const cv::Mat &dKF = pKF->descriptorsL.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);

			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}

		// If there is already a MapPoint replace otherwise add new measurement
		if (bestDist <= TH_LOW)
		{
			MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
			if (pMPinKF)
			{
				if (!pMPinKF->isBad())
				{
					if (pMPinKF->Observations() > pMP->Observations())
						pMP->Replace(pMPinKF);
					else
						pMPinKF->Replace(pMP);
				}
			}
			else
			{
				pMP->AddObservation(pKF, bestIdx);
				pKF->AddMapPoint(pMP, bestIdx);
			}
			nFused++;
		}
	}

	return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
	// Get Calibration Parameters for later projection
	const float &fx = pKF->camera.fx;
	const float &fy = pKF->camera.fy;
	const float &cx = pKF->camera.cx;
	const float &cy = pKF->camera.cy;

	// Decompose Scw
	cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
	const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
	cv::Mat Rcw = sRcw / scw;
	cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
	cv::Mat Ow = -Rcw.t()*tcw;

	// Set of MapPoints already found in the KeyFrame
	const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

	int nFused = 0;

	const int nPoints = vpPoints.size();

	// For each candidate MapPoint project and match
	for (int iMP = 0; iMP < nPoints; iMP++)
	{
		MapPoint* pMP = vpPoints[iMP];

		// Discard Bad MapPoints and already found
		if (pMP->isBad() || spAlreadyFound.count(pMP))
			continue;

		// Get 3D Coords.
		cv::Mat p3Dw = pMP->GetWorldPos();

		// Transform into Camera Coords.
		cv::Mat p3Dc = Rcw*p3Dw + tcw;

		// Depth must be positive
		if (p3Dc.at<float>(2) < 0.0f)
			continue;

		// Project into Image
		const float invz = 1.0 / p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pKF->IsInImage(u, v))
			continue;

		// Depth must be inside the scale pyramid of the image
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw - Ow;
		const float dist3D = cv::norm(PO);

		if (dist3D<minDistance || dist3D>maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		cv::Mat Pn = pMP->GetNormal();

		if (PO.dot(Pn) < 0.5*dist3D)
			continue;

		// Compute predicted scale level
		const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

		// Search in a radius
		const float radius = th*pKF->pyramid.scaleFactors[nPredictedLevel];

		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius

		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = INT_MAX;
		int bestIdx = -1;
		for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++)
		{
			const size_t idx = *vit;
			const int &kpLevel = pKF->keypointsUn[idx].octave;

			if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
				continue;

			const cv::Mat &dKF = pKF->descriptorsL.row(idx);

			int dist = DescriptorDistance(dMP, dKF);

			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}

		// If there is already a MapPoint replace otherwise add new measurement
		if (bestDist <= TH_LOW)
		{
			MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
			if (pMPinKF)
			{
				if (!pMPinKF->isBad())
					vpReplacePoint[iMP] = pMPinKF;
			}
			else
			{
				pMP->AddObservation(pKF, bestIdx);
				pKF->AddMapPoint(pMP, bestIdx);
			}
			nFused++;
		}
	}

	return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
	const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
	const float &fx = pKF1->camera.fx;
	const float &fy = pKF1->camera.fy;
	const float &cx = pKF1->camera.cx;
	const float &cy = pKF1->camera.cy;

	// Camera 1 from world
	cv::Mat R1w = pKF1->GetRotation();
	cv::Mat t1w = pKF1->GetTranslation();

	//Camera 2 from world
	cv::Mat R2w = pKF2->GetRotation();
	cv::Mat t2w = pKF2->GetTranslation();

	//Transformation between cameras
	cv::Mat sR12 = s12*R12;
	cv::Mat sR21 = (1.0 / s12)*R12.t();
	cv::Mat t21 = -sR21*t12;

	const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	const int N1 = vpMapPoints1.size();

	const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
	const int N2 = vpMapPoints2.size();

	vector<bool> vbAlreadyMatched1(N1, false);
	vector<bool> vbAlreadyMatched2(N2, false);

	for (int i = 0; i < N1; i++)
	{
		MapPoint* pMP = vpMatches12[i];
		if (pMP)
		{
			vbAlreadyMatched1[i] = true;
			int idx2 = pMP->GetIndexInKeyFrame(pKF2);
			if (idx2 >= 0 && idx2 < N2)
				vbAlreadyMatched2[idx2] = true;
		}
	}

	vector<int> vnMatch1(N1, -1);
	vector<int> vnMatch2(N2, -1);

	// Transform from KF1 to KF2 and search
	for (int i1 = 0; i1 < N1; i1++)
	{
		MapPoint* pMP = vpMapPoints1[i1];

		if (!pMP || vbAlreadyMatched1[i1])
			continue;

		if (pMP->isBad())
			continue;

		cv::Mat p3Dw = pMP->GetWorldPos();
		cv::Mat p3Dc1 = R1w*p3Dw + t1w;
		cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

		// Depth must be positive
		if (p3Dc2.at<float>(2) < 0.0)
			continue;

		const float invz = 1.0 / p3Dc2.at<float>(2);
		const float x = p3Dc2.at<float>(0)*invz;
		const float y = p3Dc2.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pKF2->IsInImage(u, v))
			continue;

		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const float dist3D = cv::norm(p3Dc2);

		// Depth must be inside the scale invariance region
		if (dist3D<minDistance || dist3D>maxDistance)
			continue;

		// Compute predicted octave
		const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

		// Search in a radius
		const float radius = th*pKF2->pyramid.scaleFactors[nPredictedLevel];

		const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = INT_MAX;
		int bestIdx = -1;
		for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;

			const cv::KeyPoint &kp = pKF2->keypointsUn[idx];

			if (kp.octave<nPredictedLevel - 1 || kp.octave>nPredictedLevel)
				continue;

			const cv::Mat &dKF = pKF2->descriptorsL.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);

			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}

		if (bestDist <= TH_HIGH)
		{
			vnMatch1[i1] = bestIdx;
		}
	}

	// Transform from KF2 to KF2 and search
	for (int i2 = 0; i2 < N2; i2++)
	{
		MapPoint* pMP = vpMapPoints2[i2];

		if (!pMP || vbAlreadyMatched2[i2])
			continue;

		if (pMP->isBad())
			continue;

		cv::Mat p3Dw = pMP->GetWorldPos();
		cv::Mat p3Dc2 = R2w*p3Dw + t2w;
		cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

		// Depth must be positive
		if (p3Dc1.at<float>(2) < 0.0)
			continue;

		const float invz = 1.0 / p3Dc1.at<float>(2);
		const float x = p3Dc1.at<float>(0)*invz;
		const float y = p3Dc1.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pKF1->IsInImage(u, v))
			continue;

		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const float dist3D = cv::norm(p3Dc1);

		// Depth must be inside the scale pyramid of the image
		if (dist3D<minDistance || dist3D>maxDistance)
			continue;

		// Compute predicted octave
		const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

		// Search in a radius of 2.5*sigma(ScaleLevel)
		const float radius = th*pKF1->pyramid.scaleFactors[nPredictedLevel];

		const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = INT_MAX;
		int bestIdx = -1;
		for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;

			const cv::KeyPoint &kp = pKF1->keypointsUn[idx];

			if (kp.octave<nPredictedLevel - 1 || kp.octave>nPredictedLevel)
				continue;

			const cv::Mat &dKF = pKF1->descriptorsL.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);

			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}

		if (bestDist <= TH_HIGH)
		{
			vnMatch2[i2] = bestIdx;
		}
	}

	// Check agreement
	int nFound = 0;

	for (int i1 = 0; i1 < N1; i1++)
	{
		int idx2 = vnMatch1[i1];

		if (idx2 >= 0)
		{
			int idx1 = vnMatch2[idx2];
			if (idx1 == i1)
			{
				vpMatches12[i1] = vpMapPoints2[idx2];
				nFound++;
			}
		}
	}

	return nFound;
}

int ORBmatcher::SearchByProjection(Frame& currFrame, const Frame& lastFrame, float th, bool monocular)
{
	int nmatches = 0;

	const cv::Mat Rcw = CameraPose::GetR(currFrame.pose.Tcw);
	const cv::Mat tcw = CameraPose::Gett(currFrame.pose.Tcw);

	const cv::Mat Rlw = CameraPose::GetR(lastFrame.pose.Tcw);
	const cv::Mat tlw = CameraPose::Gett(lastFrame.pose.Tcw);

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

			const cv::Mat desc2 = currFrame.descriptorsL.row(idx2);
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

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist)
{
	int nmatches = 0;

	const cv::Mat Rcw = CurrentFrame.pose.Tcw.rowRange(0, 3).colRange(0, 3);
	const cv::Mat tcw = CurrentFrame.pose.Tcw.rowRange(0, 3).col(3);
	const cv::Mat Ow = -Rcw.t()*tcw;

	// Rotation Histogram (to check rotation consistency)
	vector<int> rotHist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);
	const float factor = 1.0f / HISTO_LENGTH;

	const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

	for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
	{
		MapPoint* pMP = vpMPs[i];

		if (pMP)
		{
			if (!pMP->isBad() && !sAlreadyFound.count(pMP))
			{
				//Project
				cv::Mat x3Dw = pMP->GetWorldPos();
				cv::Mat x3Dc = Rcw*x3Dw + tcw;

				const float xc = x3Dc.at<float>(0);
				const float yc = x3Dc.at<float>(1);
				const float invzc = 1.0 / x3Dc.at<float>(2);

				const float u = CurrentFrame.camera.fx*xc*invzc + CurrentFrame.camera.cx;
				const float v = CurrentFrame.camera.fy*yc*invzc + CurrentFrame.camera.cy;

				/*if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
					continue;
				if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
					continue;*/
				if (!CurrentFrame.imageBounds.Contains(u, v))
					continue;

				// Compute predicted scale level
				cv::Mat PO = x3Dw - Ow;
				float dist3D = cv::norm(PO);

				const float maxDistance = pMP->GetMaxDistanceInvariance();
				const float minDistance = pMP->GetMinDistanceInvariance();

				// Depth must be inside the scale pyramid of the image
				if (dist3D<minDistance || dist3D>maxDistance)
					continue;

				int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

				// Search in a window
				const float radius = th*CurrentFrame.pyramid.scaleFactors[nPredictedLevel];

				const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

				if (vIndices2.empty())
					continue;

				const cv::Mat dMP = pMP->GetDescriptor();

				int bestDist = 256;
				int bestIdx2 = -1;

				for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
				{
					const size_t i2 = *vit;
					if (CurrentFrame.mappoints[i2])
						continue;

					const cv::Mat &d = CurrentFrame.descriptorsL.row(i2);

					const int dist = DescriptorDistance(dMP, d);

					if (dist < bestDist)
					{
						bestDist = dist;
						bestIdx2 = i2;
					}
				}

				if (bestDist <= ORBdist)
				{
					CurrentFrame.mappoints[bestIdx2] = pMP;
					nmatches++;

					if (checkOrientation_)
					{
						float rot = pKF->keypointsUn[i].angle - CurrentFrame.keypointsUn[bestIdx2].angle;
						if (rot < 0.0)
							rot += 360.0f;
						int bin = round(rot*factor);
						if (bin == HISTO_LENGTH)
							bin = 0;
						assert(bin >= 0 && bin < HISTO_LENGTH);
						rotHist[bin].push_back(bestIdx2);
					}
				}

			}
		}
	}

	if (checkOrientation_)
	{
		int ind1 = -1;
		int ind2 = -1;
		int ind3 = -1;

		ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

		for (int i = 0; i < HISTO_LENGTH; i++)
		{
			if (i != ind1 && i != ind2 && i != ind3)
			{
				for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
				{
					CurrentFrame.mappoints[rotHist[i][j]] = NULL;
					nmatches--;
				}
			}
		}
	}

	return nmatches;
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
	const int *pa = a.ptr<int32_t>();
	const int *pb = b.ptr<int32_t>();

	int dist = 0;

	for (int i = 0; i < 8; i++, pa++, pb++)
	{
		unsigned  int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}

	return dist;
}

} //namespace ORB_SLAM
