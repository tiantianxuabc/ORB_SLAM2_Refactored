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

#include "MapPoint.h"

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2
{

MapPoint::mappointid_t MapPoint::nextId_ = 0;
std::mutex MapPoint::globalMutex_;

MapPoint::MapPoint(const cv::Mat& Xw, KeyFrame* referenceKF, Map* map) :
	firstKFid(referenceKF->id), firstFrame(referenceKF->frameId), nobservations(0), trackReferenceForFrame(0),
	lastFrameSeen(0), BALocalForKF(0), fuseCandidateForKF(0), loopPointForKF(0), correctedByKF(0),
	correctedReference(0), BAGlobalForKF(0), referenceKF_(referenceKF), nvisible_(1), nfound_(1), bad_(false),
	replaced_(nullptr), minDistance_(0), maxDistance_(0), map_(map)
{
	Xw.copyTo(Xw_);
	normal_ = cv::Mat::zeros(3, 1, CV_32F);

	// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	std::unique_lock<std::mutex> lock(map_->mutexPointCreation);
	id = nextId_++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF) :
	firstKFid(-1), firstFrame(pFrame->id), nobservations(0), trackReferenceForFrame(0), lastFrameSeen(0),
	BALocalForKF(0), fuseCandidateForKF(0), loopPointForKF(0), correctedByKF(0),
	correctedReference(0), BAGlobalForKF(0), referenceKF_(static_cast<KeyFrame*>(NULL)), nvisible_(1),
	nfound_(1), bad_(false), replaced_(NULL), map_(pMap)
{
	Pos.copyTo(Xw_);
	cv::Mat Ow = pFrame->GetCameraCenter();
	normal_ = Xw_ - Ow;
	normal_ = normal_ / cv::norm(normal_);

	cv::Mat PC = Pos - Ow;
	const float dist = cv::norm(PC);
	const int level = pFrame->keypointsUn[idxF].octave;
	const float levelScaleFactor = pFrame->pyramid.scaleFactors[level];
	const int nLevels = pFrame->pyramid.nlevels;

	maxDistance_ = dist*levelScaleFactor;
	minDistance_ = maxDistance_ / pFrame->pyramid.scaleFactors[nLevels - 1];

	pFrame->descriptorsL.row(idxF).copyTo(mDescriptor);

	// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	std::unique_lock<std::mutex> lock(map_->mutexPointCreation);
	id = nextId_++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
	std::unique_lock<std::mutex> lock2(globalMutex_);
	std::unique_lock<std::mutex> lock(mutexPos_);
	Pos.copyTo(Xw_);
}

cv::Mat MapPoint::GetWorldPos()
{
	std::unique_lock<std::mutex> lock(mutexPos_);
	return Xw_.clone();
}

cv::Mat MapPoint::GetNormal()
{
	std::unique_lock<std::mutex> lock(mutexPos_);
	return normal_.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	return referenceKF_;
}

void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	if (mObservations.count(pKF))
		return;
	mObservations[pKF] = idx;

	if (pKF->uright[idx] >= 0)
		nobservations += 2;
	else
		nobservations++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
	bool bBad = false;
	{
		std::unique_lock<std::mutex> lock(mutexFeatures_);
		if (mObservations.count(pKF))
		{
			int idx = mObservations[pKF];
			if (pKF->uright[idx] >= 0)
				nobservations -= 2;
			else
				nobservations--;

			mObservations.erase(pKF);

			if (referenceKF_ == pKF)
				referenceKF_ = mObservations.begin()->first;

			// If only 2 observations or less, discard point
			if (nobservations <= 2)
				bBad = true;
		}
	}

	if (bBad)
		SetBadFlag();
}

std::map<KeyFrame*, size_t> MapPoint::GetObservations()
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	return mObservations;
}

int MapPoint::Observations()
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	return nobservations;
}

void MapPoint::SetBadFlag()
{
	std::map<KeyFrame*, size_t> obs;
	{
		std::unique_lock<std::mutex> lock1(mutexFeatures_);
		std::unique_lock<std::mutex> lock2(mutexPos_);
		bad_ = true;
		obs = mObservations;
		mObservations.clear();
	}
	for (std::map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
	{
		KeyFrame* pKF = mit->first;
		pKF->EraseMapPointMatch(mit->second);
	}

	map_->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
	std::unique_lock<std::mutex> lock1(mutexFeatures_);
	std::unique_lock<std::mutex> lock2(mutexPos_);
	return replaced_;
}

void MapPoint::Replace(MapPoint* pMP)
{
	if (pMP->id == this->id)
		return;

	int nvisible, nfound;
	std::map<KeyFrame*, size_t> obs;
	{
		std::unique_lock<std::mutex> lock1(mutexFeatures_);
		std::unique_lock<std::mutex> lock2(mutexPos_);
		obs = mObservations;
		mObservations.clear();
		bad_ = true;
		nvisible = nvisible;
		nfound = nfound;
		replaced_ = pMP;
	}

	for (std::map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
	{
		// Replace measurement in keyframe
		KeyFrame* pKF = mit->first;

		if (!pMP->IsInKeyFrame(pKF))
		{
			pKF->ReplaceMapPointMatch(mit->second, pMP);
			pMP->AddObservation(pKF, mit->second);
		}
		else
		{
			pKF->EraseMapPointMatch(mit->second);
		}
	}
	pMP->IncreaseFound(nfound);
	pMP->IncreaseVisible(nvisible);
	pMP->ComputeDistinctiveDescriptors();

	map_->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	std::unique_lock<std::mutex> lock2(mutexPos_);
	return bad_;
}

void MapPoint::IncreaseVisible(int n)
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	nvisible_ += n;
}

void MapPoint::IncreaseFound(int n)
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	nfound_ += n;
}

float MapPoint::GetFoundRatio()
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	return static_cast<float>(nfound_) / nvisible_;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
	// Retrieve all observed descriptors
	std::vector<cv::Mat> vDescriptors;

	std::map<KeyFrame*, size_t> observations;

	{
		std::unique_lock<std::mutex> lock1(mutexFeatures_);
		if (bad_)
			return;
		observations = mObservations;
	}

	if (observations.empty())
		return;

	vDescriptors.reserve(observations.size());

	for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
	{
		KeyFrame* pKF = mit->first;

		if (!pKF->isBad())
			vDescriptors.push_back(pKF->descriptorsL.row(mit->second));
	}

	if (vDescriptors.empty())
		return;

	// Compute distances between them
	const size_t N = vDescriptors.size();

	std::vector<std::vector<int>> Distances(N, std::vector<int>(N, 0));
	for (size_t i = 0; i < N; i++)
	{
		Distances[i][i] = 0;
		for (size_t j = i + 1; j < N; j++)
		{
			int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
			Distances[i][j] = distij;
			Distances[j][i] = distij;
		}
	}

	// Take the descriptor with least median distance to the rest
	int BestMedian = INT_MAX;
	int BestIdx = 0;
	for (size_t i = 0; i < N; i++)
	{
		std::vector<int> vDists(Distances[i]);
		std::sort(vDists.begin(), vDists.end());
		int median = vDists[0.5*(N - 1)];

		if (median < BestMedian)
		{
			BestMedian = median;
			BestIdx = i;
		}
	}

	{
		std::unique_lock<std::mutex> lock(mutexFeatures_);
		mDescriptor = vDescriptors[BestIdx].clone();
	}
}

cv::Mat MapPoint::GetDescriptor()
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	if (mObservations.count(pKF))
		return mObservations[pKF];
	else
		return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
	std::unique_lock<std::mutex> lock(mutexFeatures_);
	return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
	std::map<KeyFrame*, size_t> observations;
	KeyFrame* pRefKF;
	cv::Mat Pos;
	{
		std::unique_lock<std::mutex> lock1(mutexFeatures_);
		std::unique_lock<std::mutex> lock2(mutexPos_);
		if (bad_)
			return;
		observations = mObservations;
		pRefKF = referenceKF_;
		Pos = Xw_.clone();
	}

	if (observations.empty())
		return;

	cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
	int n = 0;
	for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
	{
		KeyFrame* pKF = mit->first;
		cv::Mat Owi = pKF->GetCameraCenter();
		cv::Mat normali = Xw_ - Owi;
		normal = normal + normali / cv::norm(normali);
		n++;
	}

	cv::Mat PC = Pos - pRefKF->GetCameraCenter();
	const float dist = cv::norm(PC);
	const int level = pRefKF->keypointsUn[observations[pRefKF]].octave;
	const float levelScaleFactor = pRefKF->pyramid.scaleFactors[level];
	const int nLevels = pRefKF->pyramid.nlevels;

	{
		std::unique_lock<std::mutex> lock3(mutexPos_);
		maxDistance_ = dist*levelScaleFactor;
		minDistance_ = maxDistance_ / pRefKF->pyramid.scaleFactors[nLevels - 1];
		normal_ = normal / n;
	}
}

float MapPoint::GetMinDistanceInvariance()
{
	std::unique_lock<std::mutex> lock(mutexPos_);
	return 0.8f*minDistance_;
}

float MapPoint::GetMaxDistanceInvariance()
{
	std::unique_lock<std::mutex> lock(mutexPos_);
	return 1.2f*maxDistance_;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
	float ratio;
	{
		std::unique_lock<std::mutex> lock(mutexPos_);
		ratio = maxDistance_ / currentDist;
	}

	int nScale = ceil(log(ratio) / pKF->pyramid.logScaleFactor);
	if (nScale < 0)
		nScale = 0;
	else if (nScale >= pKF->pyramid.nlevels)
		nScale = pKF->pyramid.nlevels - 1;

	return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
	float ratio;
	{
		std::unique_lock<std::mutex> lock(mutexPos_);
		ratio = maxDistance_ / currentDist;
	}

	int nScale = ceil(log(ratio) / pF->pyramid.logScaleFactor);
	if (nScale < 0)
		nScale = 0;
	else if (nScale >= pF->pyramid.nlevels)
		nScale = pF->pyramid.nlevels - 1;

	return nScale;
}



} //namespace ORB_SLAM
