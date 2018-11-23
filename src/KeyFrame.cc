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

#include "KeyFrame.h"

#include <mutex>

#include "Converter.h"
#include "ORBmatcher.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"

#define LOCK_MUTEX_POSE()        std::unique_lock<std::mutex> lock1(mutexPose_);
#define LOCK_MUTEX_CONNECTIONS() std::unique_lock<std::mutex> lock2(mutexConnections_);
#define LOCK_MUTEX_FEATURES()    std::unique_lock<std::mutex> lock3(mutexFeatures_);

namespace ORB_SLAM2
{

auto GetR = CameraPose::GetR;
auto Gett = CameraPose::Gett;

long unsigned int KeyFrame::nextId = 0;

KeyFrame::KeyFrame(Frame& frame, Map* map, KeyFrameDatabase* keyframeDB) :
	frameId(frame.id), timestamp(frame.timestamp), grid(frame.grid),
	trackReferenceForFrame(0), fuseTargetForKF(0), BALocalForKF(0), BAFixedForKF(0),
	loopQuery(0), loopWords(0), relocQuery(0), relocWords(0), BAGlobalForKF(0),
	camera(frame.camera), thDepth(frame.thDepth), N(frame.N), keypointsL(frame.keypointsL), keypointsUn(frame.keypointsUn),
	uright(frame.uright), depth(frame.depth), descriptorsL(frame.descriptorsL.clone()),
	bowVector(frame.bowVector), featureVector(frame.featureVector), pyramid(frame.pyramid), imageBounds(frame.imageBounds),
	mappoints_(frame.mappoints), keyFrameDB_(keyframeDB),
	voc_(frame.voc), firstConnection_(true), parent_(NULL), notErase_(false),
	toBeErased_(false), bad_(false), halfBaseline_(frame.camera.baseline / 2), map_(map)
{
	id = nextId++;
	SetPose(frame.pose.Tcw);
}

void KeyFrame::ComputeBoW()
{
	if (!bowVector.empty() && !featureVector.empty())
		return;

	// Feature vector associate features with nodes in the 4th level (from leaves up)
	// We assume the vocabulary tree has 6 levels, change the 4 otherwise
	voc_->transform(Converter::toDescriptorVector(descriptorsL), bowVector, featureVector, 4);
}

void KeyFrame::SetPose(const cv::Mat& Tcw_)
{
	LOCK_MUTEX_POSE();
	Tcw_.copyTo(Tcw);
	cv::Mat Rcw = GetR(Tcw);
	cv::Mat tcw = Gett(Tcw);
	cv::Mat Rwc = Rcw.t();
	Ow = -Rwc * tcw;

	Twc = cv::Mat::eye(4, 4, Tcw.type());
	Rwc.copyTo(GetR(Twc));
	Ow.copyTo(Gett(Twc));
	cv::Mat center = (cv::Mat_<float>(4, 1) << halfBaseline_, 0, 0, 1);
	Cw = Twc * center;
}

cv::Mat KeyFrame::GetPose()
{
	LOCK_MUTEX_POSE();
	return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
	LOCK_MUTEX_POSE();
	return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
	LOCK_MUTEX_POSE();
	return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
	LOCK_MUTEX_POSE();
	return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
	LOCK_MUTEX_POSE();
	return GetR(Tcw).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
	LOCK_MUTEX_POSE();
	return Gett(Tcw).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
	{
		LOCK_MUTEX_CONNECTIONS();
		if (!connectedKeyFrameWeights_.count(pKF))
			connectedKeyFrameWeights_[pKF] = weight;
		else if (connectedKeyFrameWeights_[pKF] != weight)
			connectedKeyFrameWeights_[pKF] = weight;
		else
			return;
	}

	UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
	LOCK_MUTEX_CONNECTIONS();
	vector<pair<int, KeyFrame*> > vPairs;
	vPairs.reserve(connectedKeyFrameWeights_.size());
	for (map<KeyFrame*, int>::iterator mit = connectedKeyFrameWeights_.begin(), mend = connectedKeyFrameWeights_.end(); mit != mend; mit++)
		vPairs.push_back(make_pair(mit->second, mit->first));

	sort(vPairs.begin(), vPairs.end());
	list<KeyFrame*> lKFs;
	list<int> lWs;
	for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
	{
		lKFs.push_front(vPairs[i].second);
		lWs.push_front(vPairs[i].first);
	}

	orderedConnectedKeyFrames_ = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
	orderedWeights_ = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
	LOCK_MUTEX_CONNECTIONS();
	set<KeyFrame*> s;
	for (map<KeyFrame*, int>::iterator mit = connectedKeyFrameWeights_.begin(); mit != connectedKeyFrameWeights_.end(); mit++)
		s.insert(mit->first);
	return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
	LOCK_MUTEX_CONNECTIONS();
	return orderedConnectedKeyFrames_;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
	LOCK_MUTEX_CONNECTIONS();
	if ((int)orderedConnectedKeyFrames_.size() < N)
		return orderedConnectedKeyFrames_;
	else
		return vector<KeyFrame*>(orderedConnectedKeyFrames_.begin(), orderedConnectedKeyFrames_.begin() + N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
	LOCK_MUTEX_CONNECTIONS();

	if (orderedConnectedKeyFrames_.empty())
		return vector<KeyFrame*>();

	vector<int>::iterator it = upper_bound(orderedWeights_.begin(), orderedWeights_.end(), w, KeyFrame::weightComp);
	if (it == orderedWeights_.end())
		return vector<KeyFrame*>();
	else
	{
		int n = it - orderedWeights_.begin();
		return vector<KeyFrame*>(orderedConnectedKeyFrames_.begin(), orderedConnectedKeyFrames_.begin() + n);
	}
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
	LOCK_MUTEX_CONNECTIONS();
	if (connectedKeyFrameWeights_.count(pKF))
		return connectedKeyFrameWeights_[pKF];
	else
		return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
	LOCK_MUTEX_FEATURES();
	mappoints_[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
	LOCK_MUTEX_FEATURES();
	mappoints_[idx] = static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
	int idx = pMP->GetIndexInKeyFrame(this);
	if (idx >= 0)
		mappoints_[idx] = static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
	mappoints_[idx] = pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
	LOCK_MUTEX_FEATURES();
	set<MapPoint*> s;
	for (size_t i = 0, iend = mappoints_.size(); i < iend; i++)
	{
		if (!mappoints_[i])
			continue;
		MapPoint* pMP = mappoints_[i];
		if (!pMP->isBad())
			s.insert(pMP);
	}
	return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
	LOCK_MUTEX_FEATURES();

	int nPoints = 0;
	const bool bCheckObs = minObs > 0;
	for (int i = 0; i < N; i++)
	{
		MapPoint* pMP = mappoints_[i];
		if (pMP)
		{
			if (!pMP->isBad())
			{
				if (bCheckObs)
				{
					if (mappoints_[i]->Observations() >= minObs)
						nPoints++;
				}
				else
					nPoints++;
			}
		}
	}

	return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
	LOCK_MUTEX_FEATURES();
	return mappoints_;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
	LOCK_MUTEX_FEATURES();
	return mappoints_[idx];
}

void KeyFrame::UpdateConnections()
{
	map<KeyFrame*, int> KFcounter;

	vector<MapPoint*> vpMP;

	{
		LOCK_MUTEX_FEATURES();
		vpMP = mappoints_;
	}

	//For all map points in keyframe check in which other keyframes are they seen
	//Increase counter for those keyframes
	for (vector<MapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
	{
		MapPoint* pMP = *vit;

		if (!pMP)
			continue;

		if (pMP->isBad())
			continue;

		map<KeyFrame*, size_t> observations = pMP->GetObservations();

		for (map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			if (mit->first->id == id)
				continue;
			KFcounter[mit->first]++;
		}
	}

	// This should not happen
	if (KFcounter.empty())
		return;

	//If the counter is greater than threshold add connection
	//In case no keyframe counter is over threshold add the one with maximum counter
	int nmax = 0;
	KeyFrame* pKFmax = NULL;
	int th = 15;

	vector<pair<int, KeyFrame*> > vPairs;
	vPairs.reserve(KFcounter.size());
	for (map<KeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
	{
		if (mit->second > nmax)
		{
			nmax = mit->second;
			pKFmax = mit->first;
		}
		if (mit->second >= th)
		{
			vPairs.push_back(make_pair(mit->second, mit->first));
			(mit->first)->AddConnection(this, mit->second);
		}
	}

	if (vPairs.empty())
	{
		vPairs.push_back(make_pair(nmax, pKFmax));
		pKFmax->AddConnection(this, nmax);
	}

	sort(vPairs.begin(), vPairs.end());
	list<KeyFrame*> lKFs;
	list<int> lWs;
	for (size_t i = 0; i < vPairs.size(); i++)
	{
		lKFs.push_front(vPairs[i].second);
		lWs.push_front(vPairs[i].first);
	}

	{
		LOCK_MUTEX_CONNECTIONS();

		// mspConnectedKeyFrames = spConnectedKeyFrames;
		connectedKeyFrameWeights_ = KFcounter;
		orderedConnectedKeyFrames_ = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
		orderedWeights_ = vector<int>(lWs.begin(), lWs.end());

		if (firstConnection_ && id != 0)
		{
			parent_ = orderedConnectedKeyFrames_.front();
			parent_->AddChild(this);
			firstConnection_ = false;
		}

	}
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
	LOCK_MUTEX_CONNECTIONS();
	children_.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
	LOCK_MUTEX_CONNECTIONS();
	children_.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
	LOCK_MUTEX_CONNECTIONS();
	parent_ = pKF;
	pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
	LOCK_MUTEX_CONNECTIONS();
	return children_;
}

KeyFrame* KeyFrame::GetParent()
{
	LOCK_MUTEX_CONNECTIONS();
	return parent_;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
	LOCK_MUTEX_CONNECTIONS();
	return children_.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
	LOCK_MUTEX_CONNECTIONS();
	notErase_ = true;
	loopEdges_.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
	LOCK_MUTEX_CONNECTIONS();
	return loopEdges_;
}

void KeyFrame::SetNotErase()
{
	LOCK_MUTEX_CONNECTIONS();
	notErase_ = true;
}

void KeyFrame::SetErase()
{
	{
		LOCK_MUTEX_CONNECTIONS();
		if (loopEdges_.empty())
		{
			notErase_ = false;
		}
	}

	if (toBeErased_)
	{
		SetBadFlag();
	}
}

void KeyFrame::SetBadFlag()
{
	{
		LOCK_MUTEX_CONNECTIONS();
		if (id == 0)
			return;
		else if (notErase_)
		{
			toBeErased_ = true;
			return;
		}
	}

	for (map<KeyFrame*, int>::iterator mit = connectedKeyFrameWeights_.begin(), mend = connectedKeyFrameWeights_.end(); mit != mend; mit++)
		mit->first->EraseConnection(this);

	for (size_t i = 0; i < mappoints_.size(); i++)
		if (mappoints_[i])
			mappoints_[i]->EraseObservation(this);
	{
		LOCK_MUTEX_CONNECTIONS();
		LOCK_MUTEX_FEATURES();

		connectedKeyFrameWeights_.clear();
		orderedConnectedKeyFrames_.clear();

		// Update Spanning Tree
		set<KeyFrame*> sParentCandidates;
		sParentCandidates.insert(parent_);

		// Assign at each iteration one children with a parent (the pair with highest covisibility weight)
		// Include that children as new parent candidate for the rest
		while (!children_.empty())
		{
			bool bContinue = false;

			int max = -1;
			KeyFrame* pC;
			KeyFrame* pP;

			for (set<KeyFrame*>::iterator sit = children_.begin(), send = children_.end(); sit != send; sit++)
			{
				KeyFrame* pKF = *sit;
				if (pKF->isBad())
					continue;

				// Check if a parent candidate is connected to the keyframe
				vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
				for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
				{
					for (set<KeyFrame*>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
					{
						if (vpConnected[i]->id == (*spcit)->id)
						{
							int w = pKF->GetWeight(vpConnected[i]);
							if (w > max)
							{
								pC = pKF;
								pP = vpConnected[i];
								max = w;
								bContinue = true;
							}
						}
					}
				}
			}

			if (bContinue)
			{
				pC->ChangeParent(pP);
				sParentCandidates.insert(pC);
				children_.erase(pC);
			}
			else
				break;
		}

		// If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
		if (!children_.empty())
			for (set<KeyFrame*>::iterator sit = children_.begin(); sit != children_.end(); sit++)
			{
				(*sit)->ChangeParent(parent_);
			}

		parent_->EraseChild(this);
		Tcp = Tcw*parent_->GetPoseInverse();
		bad_ = true;
	}


	map_->EraseKeyFrame(this);
	keyFrameDB_->erase(this);
}

bool KeyFrame::isBad()
{
	LOCK_MUTEX_CONNECTIONS();
	return bad_;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
	bool bUpdate = false;
	{
		LOCK_MUTEX_CONNECTIONS();
		if (connectedKeyFrameWeights_.count(pKF))
		{
			connectedKeyFrameWeights_.erase(pKF);
			bUpdate = true;
		}
	}

	if (bUpdate)
		UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
	return grid.GetFeaturesInArea(x, y, r);
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
	return imageBounds.Contains(x, y);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
	const float z = depth[i];
	const float invfx = 1.f / camera.fx;
	const float invfy = 1.f / camera.fy;
	const float cx = camera.cx;
	const float cy = camera.cy;
	if (z > 0)
	{
		const float u = keypointsL[i].pt.x;
		const float v = keypointsL[i].pt.y;
		const float x = (u - cx)*z*invfx;
		const float y = (v - cy)*z*invfy;
		cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

		LOCK_MUTEX_POSE();
		return GetR(Twc) * x3Dc + Gett(Twc);
	}
	else
		return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
	vector<MapPoint*> vpMapPoints;
	cv::Mat Tcw_;
	{
		LOCK_MUTEX_FEATURES();
		LOCK_MUTEX_POSE();
		vpMapPoints = mappoints_;
		Tcw_ = Tcw.clone();
	}

	vector<float> vDepths;
	vDepths.reserve(N);
	cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
	Rcw2 = Rcw2.t();
	float zcw = Tcw_.at<float>(2, 3);
	for (int i = 0; i < N; i++)
	{
		if (mappoints_[i])
		{
			MapPoint* pMP = mappoints_[i];
			cv::Mat x3Dw = pMP->GetWorldPos();
			float z = Rcw2.dot(x3Dw) + zcw;
			vDepths.push_back(z);
		}
	}

	sort(vDepths.begin(), vDepths.end());

	return vDepths[(vDepths.size() - 1) / q];
}

} //namespace ORB_SLAM
