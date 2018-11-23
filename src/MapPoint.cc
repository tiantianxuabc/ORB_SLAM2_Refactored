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

#define LOCK_MUTEX_POINT_CREATION() std::unique_lock<std::mutex> lock1(map_->mutexPointCreation);
#define LOCK_MUTEX_POSITION()       std::unique_lock<std::mutex> lock2(mutexPos_);
#define LOCK_MUTEX_FEATURES()       std::unique_lock<std::mutex> lock3(mutexFeatures_);
#define LOCK_MUTEX_GLOBAL()         std::unique_lock<std::mutex> lock3(globalMutex_);

namespace ORB_SLAM2
{

MapPoint::mappointid_t MapPoint::nextId = 0;
std::mutex MapPoint::globalMutex_;

MapPoint::MapPoint(const cv::Mat& Xw, KeyFrame* referenceKF, Map* map) :
	firstKFid(referenceKF->id), firstFrame(referenceKF->frameId), nobservations_(0), trackReferenceForFrame(0),
	lastFrameSeen(0), BALocalForKF(0), fuseCandidateForKF(0), loopPointForKF(0), correctedByKF(0),
	correctedReference(0), BAGlobalForKF(0), referenceKF_(referenceKF), nvisible_(1), nfound_(1), bad_(false),
	replaced_(nullptr), minDistance_(0), maxDistance_(0), map_(map)
{
	Xw.copyTo(Xw_);
	normal_ = cv::Mat::zeros(3, 1, CV_32F);

	// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	LOCK_MUTEX_POINT_CREATION();
	id = nextId++;
}

MapPoint::MapPoint(const cv::Mat& Xw, Map* map, Frame* frame, const int &idx) :
	firstKFid(-1), firstFrame(frame->id), nobservations_(0), trackReferenceForFrame(0), lastFrameSeen(0),
	BALocalForKF(0), fuseCandidateForKF(0), loopPointForKF(0), correctedByKF(0),
	correctedReference(0), BAGlobalForKF(0), referenceKF_(nullptr), nvisible_(1),
	nfound_(1), bad_(false), replaced_(nullptr), map_(map)
{
	Xw.copyTo(Xw_);
	cv::Mat Ow = frame->GetCameraCenter();
	normal_ = Xw_ - Ow;
	normal_ = normal_ / cv::norm(normal_);

	cv::Mat PC = Xw - Ow;
	const float dist = cv::norm(PC);
	const int level = frame->keypointsUn[idxF].octave;
	const float levelScaleFactor = frame->pyramid.scaleFactors[level];
	const int nLevels = frame->pyramid.nlevels;

	maxDistance_ = dist*levelScaleFactor;
	minDistance_ = maxDistance_ / frame->pyramid.scaleFactors[nLevels - 1];

	frame->descriptorsL.row(idxF).copyTo(mDescriptor);

	// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	LOCK_MUTEX_POINT_CREATION();
	id = nextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
	LOCK_MUTEX_GLOBAL();
	LOCK_MUTEX_POSITION();
	Pos.copyTo(Xw_);
}

cv::Mat MapPoint::GetWorldPos()
{
	LOCK_MUTEX_POSITION();
	return Xw_.clone();
}

cv::Mat MapPoint::GetNormal()
{
	LOCK_MUTEX_POSITION();
	return normal_.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
	LOCK_MUTEX_FEATURES();
	return referenceKF_;
}

void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
	LOCK_MUTEX_FEATURES();
	if (mObservations.count(pKF))
		return;
	mObservations[pKF] = idx;

	if (pKF->uright[idx] >= 0)
		nobservations_ += 2;
	else
		nobservations_++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
	bool bBad = false;
	{
		LOCK_MUTEX_FEATURES();
		if (mObservations.count(pKF))
		{
			int idx = mObservations[pKF];
			if (pKF->uright[idx] >= 0)
				nobservations_ -= 2;
			else
				nobservations_--;

			mObservations.erase(pKF);

			if (referenceKF_ == pKF)
				referenceKF_ = mObservations.begin()->first;

			// If only 2 observations or less, discard point
			if (nobservations_ <= 2)
				bBad = true;
		}
	}

	if (bBad)
		SetBadFlag();
}

std::map<KeyFrame*, size_t> MapPoint::GetObservations()
{
	LOCK_MUTEX_FEATURES();
	return mObservations;
}

int MapPoint::Observations()
{
	LOCK_MUTEX_FEATURES();
	return nobservations_;
}

void MapPoint::SetBadFlag()
{
	std::map<KeyFrame*, size_t> obs;
	{
		LOCK_MUTEX_FEATURES();
		LOCK_MUTEX_POSITION();
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
	LOCK_MUTEX_FEATURES();
	LOCK_MUTEX_POSITION();
	return replaced_;
}

void MapPoint::Replace(MapPoint* pMP)
{
	if (pMP->id == this->id)
		return;

	int nvisible, nfound;
	std::map<KeyFrame*, size_t> obs;
	{
		LOCK_MUTEX_FEATURES();
		LOCK_MUTEX_POSITION();
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
	LOCK_MUTEX_FEATURES();
	LOCK_MUTEX_POSITION();
	return bad_;
}

void MapPoint::IncreaseVisible(int n)
{
	LOCK_MUTEX_FEATURES();
	nvisible_ += n;
}

void MapPoint::IncreaseFound(int n)
{
	LOCK_MUTEX_FEATURES();
	nfound_ += n;
}

float MapPoint::GetFoundRatio()
{
	LOCK_MUTEX_FEATURES();
	return static_cast<float>(nfound_) / nvisible_;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
	// Retrieve all observed descriptors
	std::vector<cv::Mat> vDescriptors;

	std::map<KeyFrame*, size_t> observations;

	{
		LOCK_MUTEX_FEATURES();
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
		LOCK_MUTEX_FEATURES();
		mDescriptor = vDescriptors[BestIdx].clone();
	}
}

cv::Mat MapPoint::GetDescriptor()
{
	LOCK_MUTEX_FEATURES();
	return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
	LOCK_MUTEX_FEATURES();
	if (mObservations.count(pKF))
		return mObservations[pKF];
	else
		return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
	LOCK_MUTEX_FEATURES();
	return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
	std::map<KeyFrame*, size_t> observations;
	KeyFrame* pRefKF;
	cv::Mat Pos;
	{
		LOCK_MUTEX_FEATURES();
		LOCK_MUTEX_POSITION();
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
		LOCK_MUTEX_POSITION();
		maxDistance_ = dist*levelScaleFactor;
		minDistance_ = maxDistance_ / pRefKF->pyramid.scaleFactors[nLevels - 1];
		normal_ = normal / n;
	}
}

float MapPoint::GetMinDistanceInvariance()
{
	LOCK_MUTEX_POSITION();
	return 0.8f*minDistance_;
}

float MapPoint::GetMaxDistanceInvariance()
{
	LOCK_MUTEX_POSITION();
	return 1.2f*maxDistance_;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
	float ratio;
	{
		LOCK_MUTEX_POSITION();
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
		LOCK_MUTEX_POSITION();
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
