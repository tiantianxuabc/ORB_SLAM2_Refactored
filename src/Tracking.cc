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


#include "Tracking.h"

#include <iostream>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBmatcher.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "PnPsolver.h"
#include "Usleep.h"

using namespace std;

namespace ORB_SLAM2
{

namespace Optimizer
{

int PoseOptimization(Frame* pFrame);
void GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5, bool *pbStopFlag = NULL,
	const unsigned long nLoopKF = 0, const bool bRobust = true);

}

TrackPoint::TrackPoint(const Frame& frame, bool lost)
	: pReferenceKF(frame.mpReferenceKF), timestamp(frame.mTimeStamp), lost(lost)
{
	Tcr = frame.mTcw * frame.mpReferenceKF->GetPoseInverse();
}

struct TrackerParameters
{
	//New KeyFrame rules (according to fps)
	int minFrames;
	int maxFrames;

	// Threshold close/far points
	// Points seen as close by the stereo/RGBD sensor are considered reliable
	// and inserted from just one frame. Far points requiere a match in two keyframes.
	float thDepth;

	TrackerParameters(int minFrames, int maxFrames, float thDepth)
		: minFrames(minFrames), maxFrames(maxFrames), thDepth(thDepth) {}
};

struct LocalMap
{
	LocalMap(Map* pMap) : mpMap(pMap) {}

	void Update(Frame& mCurrentFrame)
	{
		// This is for visualization
		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		// Update
		UpdateLocalKeyFrames(mCurrentFrame);
		UpdateLocalPoints(mCurrentFrame);
	}

	void UpdateLocalKeyFrames(Frame& mCurrentFrame)
	{
		// Each map point vote for the keyframes in which it has been observed
		map<KeyFrame*, int> keyframeCounter;
		for (int i = 0; i < mCurrentFrame.N; i++)
		{
			if (mCurrentFrame.mvpMapPoints[i])
			{
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
				if (!pMP->isBad())
				{
					const map<KeyFrame*, size_t> observations = pMP->GetObservations();
					for (map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
						keyframeCounter[it->first]++;
				}
				else
				{
					mCurrentFrame.mvpMapPoints[i] = NULL;
				}
			}
		}

		if (keyframeCounter.empty())
			return;

		int max = 0;
		KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

		mvpLocalKeyFrames.clear();
		mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			KeyFrame* pKF = it->first;

			if (pKF->isBad())
				continue;

			if (it->second > max)
			{
				max = it->second;
				pKFmax = pKF;
			}

			mvpLocalKeyFrames.push_back(it->first);
			pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
		}


		// Include also some not-already-included keyframes that are neighbors to already-included keyframes
		for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
		{
			// Limit the number of keyframes
			if (mvpLocalKeyFrames.size() > 80)
				break;

			KeyFrame* pKF = *itKF;

			const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

			for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
			{
				KeyFrame* pNeighKF = *itNeighKF;
				if (!pNeighKF->isBad())
				{
					if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
					{
						mvpLocalKeyFrames.push_back(pNeighKF);
						pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
						break;
					}
				}
			}

			const set<KeyFrame*> spChilds = pKF->GetChilds();
			for (set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				KeyFrame* pChildKF = *sit;
				if (!pChildKF->isBad())
				{
					if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
					{
						mvpLocalKeyFrames.push_back(pChildKF);
						pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
						break;
					}
				}
			}

			KeyFrame* pParent = pKF->GetParent();
			if (pParent)
			{
				if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
				{
					mvpLocalKeyFrames.push_back(pParent);
					pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
					break;
				}
			}

		}

		if (pKFmax)
		{
			mpReferenceKF = pKFmax;
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		}
	}

	void UpdateLocalPoints(Frame& mCurrentFrame)
	{
		mvpLocalMapPoints.clear();

		for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
		{
			KeyFrame* pKF = *itKF;
			const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

			for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				MapPoint* pMP = *itMP;
				if (!pMP)
					continue;
				if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
					continue;
				if (!pMP->isBad())
				{
					mvpLocalMapPoints.push_back(pMP);
					pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
				}
			}
		}
	}

	KeyFrame* mpReferenceKF;
	std::vector<KeyFrame*> mvpLocalKeyFrames;
	std::vector<MapPoint*> mvpLocalMapPoints;
	Map* mpMap;
};

static int DiscardOutliers(Frame& mCurrentFrame)
{
	int nmatchesMap = 0;
	for (int i = 0; i < mCurrentFrame.N; i++)
	{
		if (mCurrentFrame.mvpMapPoints[i])
		{
			if (mCurrentFrame.mvbOutlier[i])
			{
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
				mCurrentFrame.mvbOutlier[i] = false;
				pMP->mbTrackInView = false;
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
			}
			else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
				nmatchesMap++;
		}
	}
	return nmatchesMap;
}

static void UpdateLastFramePose(Frame& mLastFrame, const TrackPoint& LastTrackPoint)
{
	// Update pose according to reference keyframe
	KeyFrame* referenceKF = mLastFrame.mpReferenceKF;
	cv::Mat Tlr = LastTrackPoint.Tcr;
	mLastFrame.SetPose(Tlr * referenceKF->GetPose());
}

bool TrackWithMotionModel(Frame& currFrame, Frame& lastFrame, const cv::Mat& velocity,
	int minInliers, int sensor, bool* fewMatches = nullptr)
{
	ORBmatcher matcher(0.9f, true);

	currFrame.SetPose(velocity * lastFrame.mTcw);

	// Project points seen in previous frame
	const float threshold = sensor == System::STEREO ? 7.f : 15.f;
	const int minMatches = 20;
	int nmatches = 0;
	{
		std::fill(std::begin(currFrame.mvpMapPoints), std::end(currFrame.mvpMapPoints), nullptr);
		nmatches = matcher.SearchByProjection(currFrame, lastFrame, threshold, sensor == System::MONOCULAR);
	}
	if (nmatches < minMatches)
	{
		// If few matches, uses a wider window search
		std::fill(std::begin(currFrame.mvpMapPoints), std::end(currFrame.mvpMapPoints), nullptr);
		nmatches = matcher.SearchByProjection(currFrame, lastFrame, 2 * threshold, sensor == System::MONOCULAR);
	}

	if (nmatches < minMatches)
		return false;

	// Optimize frame pose with all matches
	Optimizer::PoseOptimization(&currFrame);

	// Discard outliers
	const int ninliers = DiscardOutliers(currFrame);

	if (fewMatches)
		*fewMatches = ninliers < 10;

	return ninliers >= minInliers;
}

static bool TrackReferenceKeyFrame(Frame& currFrame, KeyFrame* referenceKF, Frame& lastFrame, int minInliers = 10)
{
	// Compute Bag of Words vector
	currFrame.ComputeBoW();

	// We perform first an ORB matching with the reference keyframe
	// If enough matches are found we setup a PnP solver
	ORBmatcher matcher(0.7f, true);
	vector<MapPoint*> mappoints;

	const int minMatches = 15;
	const int nmatches = matcher.SearchByBoW(referenceKF, currFrame, mappoints);

	if (nmatches < minMatches)
		return false;

	currFrame.mvpMapPoints = mappoints;
	currFrame.SetPose(lastFrame.mTcw);

	Optimizer::PoseOptimization(&currFrame);

	// Discard outliers
	const int ninliers = DiscardOutliers(currFrame);

	return ninliers >= minInliers;
}

class NewKeyFrameCondition
{
public:

	using Parameters = TrackerParameters;

	NewKeyFrameCondition(Map* pMap, const LocalMap& LocalMap, const Parameters& param, int sensor)
		: mpMap(pMap), mLocalMap(LocalMap), param_(param), mSensor(sensor) {}

	bool Satisfy(const Frame& mCurrentFrame, LocalMapping* mpLocalMapper, int mnMatchesInliers, int lastRelocFrameId, int lastKeyFrameId) const
	{
		// If Local Mapping is freezed by a Loop Closure do not insert keyframes
		if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
			return false;

		const int nKFs = mpMap->KeyFramesInMap();

		// Do not insert keyframes if not enough frames have passed from last relocalisation
		if ((int)mCurrentFrame.mnId<lastRelocFrameId + param_.maxFrames && nKFs>param_.maxFrames)
			return false;

		// Tracked MapPoints in the reference keyframe
		int nMinObs = 3;
		if (nKFs <= 2)
			nMinObs = 2;
		int nRefMatches = mLocalMap.mpReferenceKF->TrackedMapPoints(nMinObs);

		// Local Mapping accept keyframes?
		bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

		// Check how many "close" points are being tracked and how many could be potentially created.
		int nNonTrackedClose = 0;
		int nTrackedClose = 0;
		if (mSensor != System::MONOCULAR)
		{
			for (int i = 0; i < mCurrentFrame.N; i++)
			{
				if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < param_.thDepth)
				{
					if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
						nTrackedClose++;
					else
						nNonTrackedClose++;
				}
			}
		}

		bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

		// Thresholds
		float thRefRatio = 0.75f;
		if (nKFs < 2)
			thRefRatio = 0.4f;

		if (mSensor == System::MONOCULAR)
			thRefRatio = 0.9f;

		// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
		const bool c1a = (int)mCurrentFrame.mnId >= lastKeyFrameId + param_.maxFrames;
		// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
		const bool c1b = ((int)mCurrentFrame.mnId >= lastKeyFrameId + param_.minFrames && bLocalMappingIdle);
		//Condition 1c: tracking is weak
		const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches*0.25 || bNeedToInsertClose);
		// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
		const bool c2 = ((mnMatchesInliers < nRefMatches*thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

		if ((c1a || c1b || c1c) && c2)
		{
			// If the mapping accepts keyframes, insert keyframe.
			// Otherwise send a signal to interrupt BA
			if (bLocalMappingIdle)
			{
				return true;
			}
			else
			{
				mpLocalMapper->InterruptBA();
				if (mSensor != System::MONOCULAR)
				{
					if (mpLocalMapper->KeyframesInQueue() < 3)
						return true;
					else
						return false;
				}
				else
					return false;
			}
		}
		else
			return false;
	}

private:
	Map* mpMap;
	const LocalMap& mLocalMap;
	Parameters param_;
	int mSensor;
};

static void ConvertToGray(const cv::Mat& src, cv::Mat& dst, bool RGB)
{
	static const int codes[] = { cv::COLOR_RGB2GRAY, cv::COLOR_BGR2GRAY, cv::COLOR_RGBA2GRAY, cv::COLOR_BGRA2GRAY };

	const int ch = src.channels();
	CV_Assert(ch == 1 || ch == 3 || ch == 4);

	if (ch == 1)
	{
		dst = src;
		return;
	}

	const int idx = ((ch == 3 ? 0 : 1) << 1) + (RGB ? 0 : 1);
	cv::cvtColor(src, dst, codes[idx]);
}

static void SearchLocalPoints(const LocalMap& mLocalMap, Frame& mCurrentFrame, float th)
{
	// Do not search map points already matched
	for (vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
	{
		MapPoint* pMP = *vit;
		if (pMP)
		{
			if (pMP->isBad())
			{
				*vit = static_cast<MapPoint*>(NULL);
			}
			else
			{
				pMP->IncreaseVisible();
				pMP->mnLastFrameSeen = mCurrentFrame.mnId;
				pMP->mbTrackInView = false;
			}
		}
	}

	int nToMatch = 0;

	// Project points in frame and check its visibility
	auto& mvpLocalMapPoints = mLocalMap.mvpLocalMapPoints;
	for (auto vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
	{
		MapPoint* pMP = *vit;
		if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
			continue;
		if (pMP->isBad())
			continue;
		// Project (this fills MapPoint variables for matching)
		if (mCurrentFrame.isInFrustum(pMP, 0.5))
		{
			pMP->IncreaseVisible();
			nToMatch++;
		}
	}

	if (nToMatch > 0)
	{
		ORBmatcher matcher(0.8f);
		matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
	}
}

static int TrackLocalMap(LocalMap& mLocalMap, Frame& mCurrentFrame, float th, bool mbLocalizationMode, bool stereo)
{
	// We have an estimation of the camera pose and some map points tracked in the frame.
	// We retrieve the local map and try to find matches to points in the local map.

	mLocalMap.Update(mCurrentFrame);

	SearchLocalPoints(mLocalMap, mCurrentFrame, th);

	// Optimize Pose
	Optimizer::PoseOptimization(&mCurrentFrame);
	int mnMatchesInliers = 0;

	// Update MapPoints Statistics
	for (int i = 0; i < mCurrentFrame.N; i++)
	{
		if (mCurrentFrame.mvpMapPoints[i])
		{
			if (!mCurrentFrame.mvbOutlier[i])
			{
				mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
				if (!mbLocalizationMode)
				{
					if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
						mnMatchesInliers++;
				}
				else
					mnMatchesInliers++;
			}
			else if (stereo)
				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

		}
	}

	return mnMatchesInliers;
}

void CreateMapPoints(Frame& mCurrentFrame, KeyFrame* pKF, Map* mpMap, float thDepth)
{
	mCurrentFrame.UpdatePoseMatrices();

	// We sort points by the measured depth by the stereo/RGBD sensor.
	// We create all those MapPoints whose depth < param_.thDepth.
	// If there are less than 100 close points we create the 100 closest.
	vector<pair<float, int> > vDepthIdx;
	vDepthIdx.reserve(mCurrentFrame.N);
	for (int i = 0; i < mCurrentFrame.N; i++)
	{
		float z = mCurrentFrame.mvDepth[i];
		if (z > 0)
		{
			vDepthIdx.push_back(make_pair(z, i));
		}
	}

	if (!vDepthIdx.empty())
	{
		sort(vDepthIdx.begin(), vDepthIdx.end());

		int nPoints = 0;
		for (size_t j = 0; j < vDepthIdx.size(); j++)
		{
			int i = vDepthIdx[j].second;

			bool bCreateNew = false;

			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
			if (!pMP)
				bCreateNew = true;
			else if (pMP->Observations() < 1)
			{
				bCreateNew = true;
				mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
			}

			if (bCreateNew)
			{
				cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
				MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
				pNewMP->AddObservation(pKF, i);
				pKF->AddMapPoint(pNewMP, i);
				pNewMP->ComputeDistinctiveDescriptors();
				pNewMP->UpdateNormalAndDepth();
				mpMap->AddMapPoint(pNewMP);

				mCurrentFrame.mvpMapPoints[i] = pNewMP;
				nPoints++;
			}
			else
			{
				nPoints++;
			}

			if (vDepthIdx[j].first > thDepth && nPoints > 100)
				break;
		}
	}
}

static void CreateMapPointsVO(Frame& LastFrame, list<MapPoint*>& mlpTemporalPoints, Map* mpMap, float thDepth)
{
	// Create "visual odometry" MapPoints
	// We sort points according to their measured depth by the stereo/RGB-D sensor
	vector<pair<float, int> > vDepthIdx;
	vDepthIdx.reserve(LastFrame.N);
	for (int i = 0; i < LastFrame.N; i++)
	{
		float z = LastFrame.mvDepth[i];
		if (z > 0)
		{
			vDepthIdx.push_back(make_pair(z, i));
		}
	}

	if (vDepthIdx.empty())
		return;

	sort(vDepthIdx.begin(), vDepthIdx.end());

	// We insert all close points (depth<param_.thDepth)
	// If less than 100 close points, we insert the 100 closest ones.
	int nPoints = 0;
	for (size_t j = 0; j < vDepthIdx.size(); j++)
	{
		int i = vDepthIdx[j].second;

		bool bCreateNew = false;

		MapPoint* pMP = LastFrame.mvpMapPoints[i];
		if (!pMP)
			bCreateNew = true;
		else if (pMP->Observations() < 1)
		{
			bCreateNew = true;
		}

		if (bCreateNew)
		{
			cv::Mat x3D = LastFrame.UnprojectStereo(i);
			MapPoint* pNewMP = new MapPoint(x3D, mpMap, &LastFrame, i);

			LastFrame.mvpMapPoints[i] = pNewMP;

			mlpTemporalPoints.push_back(pNewMP);
			nPoints++;
		}
		else
		{
			nPoints++;
		}

		if (vDepthIdx[j].first > thDepth && nPoints > 100)
			break;
	}
}

class Relocalizer
{
public:

	Relocalizer() : mnLastRelocFrameId(0) {}

	bool Relocalize(Frame& mCurrentFrame, KeyFrameDatabase* mpKeyFrameDB)
	{
		// Compute Bag of Words Vector
		mCurrentFrame.ComputeBoW();

		// Relocalization is performed when tracking is lost
		// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
		vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

		if (vpCandidateKFs.empty())
			return false;

		const int nKFs = static_cast<int>(vpCandidateKFs.size());

		// We perform first an ORB matching with each candidate
		// If enough matches are found we setup a PnP solver
		ORBmatcher matcher(0.75, true);

		vector<PnPsolver*> vpPnPsolvers;
		vpPnPsolvers.resize(nKFs);

		vector<vector<MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nKFs);

		vector<bool> vbDiscarded;
		vbDiscarded.resize(nKFs);

		int nCandidates = 0;

		for (int i = 0; i < nKFs; i++)
		{
			KeyFrame* pKF = vpCandidateKFs[i];
			if (pKF->isBad())
				vbDiscarded[i] = true;
			else
			{
				int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
				if (nmatches < 15)
				{
					vbDiscarded[i] = true;
					continue;
				}
				else
				{
					PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
					pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5f, 5.991f);
					vpPnPsolvers[i] = pSolver;
					nCandidates++;
				}
			}
		}

		// Alternatively perform some iterations of P4P RANSAC
		// Until we found a camera pose supported by enough inliers
		bool bMatch = false;
		ORBmatcher matcher2(0.9f, true);

		while (nCandidates > 0 && !bMatch)
		{
			for (int i = 0; i < nKFs; i++)
			{
				if (vbDiscarded[i])
					continue;

				// Perform 5 Ransac Iterations
				vector<bool> vbInliers;
				int nInliers;
				bool bNoMore;

				PnPsolver* pSolver = vpPnPsolvers[i];
				cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

				// If Ransac reachs max. iterations discard keyframe
				if (bNoMore)
				{
					vbDiscarded[i] = true;
					nCandidates--;
				}

				// If a Camera Pose is computed, optimize
				if (!Tcw.empty())
				{
					Tcw.copyTo(mCurrentFrame.mTcw);

					set<MapPoint*> sFound;

					const int np = static_cast<int>(vbInliers.size());

					for (int j = 0; j < np; j++)
					{
						if (vbInliers[j])
						{
							mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
							sFound.insert(vvpMapPointMatches[i][j]);
						}
						else
							mCurrentFrame.mvpMapPoints[j] = NULL;
					}

					int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

					if (nGood < 10)
						continue;

					for (int io = 0; io < mCurrentFrame.N; io++)
						if (mCurrentFrame.mvbOutlier[io])
							mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

					// If few inliers, search by projection in a coarse window and optimize again
					if (nGood < 50)
					{
						int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

						if (nadditional + nGood >= 50)
						{
							nGood = Optimizer::PoseOptimization(&mCurrentFrame);

							// If many inliers but still not enough, search by projection again in a narrower window
							// the camera has been already optimized with many points
							if (nGood > 30 && nGood < 50)
							{
								sFound.clear();
								for (int ip = 0; ip < mCurrentFrame.N; ip++)
									if (mCurrentFrame.mvpMapPoints[ip])
										sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
								nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

								// Final optimization
								if (nGood + nadditional >= 50)
								{
									nGood = Optimizer::PoseOptimization(&mCurrentFrame);

									for (int io = 0; io < mCurrentFrame.N; io++)
										if (mCurrentFrame.mvbOutlier[io])
											mCurrentFrame.mvpMapPoints[io] = NULL;
								}
							}
						}
					}


					// If the pose is supported by enough inliers stop ransacs and continue
					if (nGood >= 50)
					{
						bMatch = true;
						break;
					}
				}
			}
		}

		if (!bMatch)
		{
			return false;
		}
		else
		{
			mnLastRelocFrameId = mCurrentFrame.mnId;
			return true;
		}
	}

	int GetLastRelocFrameId() const
	{
		return mnLastRelocFrameId;
	}

private:

	int mnLastRelocFrameId;
};

struct CameraParams
{
	float fx;                 //!< focal length x (pixel)
	float fy;                 //!< focal length y (pixel)
	float cx;                 //!< principal point x (pixel)
	float cy;                 //!< principal point y (pixel)
	float bf;                 //!< stereo baseline times fx
	float baseline;

	CameraParams()
	{
		fx = 1.f;
		fy = 1.f;
		cx = 0.f;
		cy = 0.f;
		bf = 1.f;
		baseline = 1.f;
	}

	cv::Mat1f Mat() const
	{
		cv::Mat1f K = (cv::Mat1f(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
		return K;
	}
};

static CameraParams ReadCameraParams(const cv::FileStorage& fs)
{
	CameraParams param;
	param.fx = fs["Camera.fx"];
	param.fy = fs["Camera.fy"];
	param.cx = fs["Camera.cx"];
	param.cy = fs["Camera.cy"];
	param.bf = fs["Camera.bf"];
	param.baseline = param.bf / param.fx;
	return param;
}

static cv::Mat1f ReadDistCoeffs(const cv::FileStorage& fs)
{
	const float k1 = fs["Camera.k1"];
	const float k2 = fs["Camera.k2"];
	const float p1 = fs["Camera.p1"];
	const float p2 = fs["Camera.p2"];
	const float k3 = fs["Camera.k3"];
	cv::Mat1f distCoeffs = k3 == 0 ? (cv::Mat1f(4, 1) << k1, k2, p1, p2) : (cv::Mat1f(5, 1) << k1, k2, p1, p2, k3);
	return distCoeffs;
}

static float ReadFps(const cv::FileStorage& fs)
{
	const float fps = fs["Camera.fps"];
	return fps == 0 ? 30 : fps;
}

struct ORBExtractorParams
{
	int nfeatures;
	float scaleFactor;
	int nlevels;
	int initTh;
	int minTh;
};

static ORBExtractorParams ReadExtractorParams(const cv::FileStorage& fs)
{
	ORBExtractorParams param;
	param.nfeatures = fs["ORBextractor.nFeatures"];
	param.scaleFactor = fs["ORBextractor.scaleFactor"];
	param.nlevels = fs["ORBextractor.nLevels"];
	param.initTh = fs["ORBextractor.iniThFAST"];
	param.minTh = fs["ORBextractor.minThFAST"];
	return param;
}

static float ReadDepthFactor(const cv::FileStorage& fs)
{
	const float factor = fs["DepthMapFactor"];
	return fabs(factor) < 1e-5 ? 1 : 1.f / factor;
}

static void PrintSettings(const CameraParams& camera, const cv::Mat1f& distCoeffs,
	float fps, bool rgb, const ORBExtractorParams& param, float thDepth, int sensor)
{
	cout << endl << "Camera Parameters: " << endl;
	cout << "- fx: " << camera.fx << endl;
	cout << "- fy: " << camera.fy << endl;
	cout << "- cx: " << camera.cx << endl;
	cout << "- cy: " << camera.cy << endl;
	cout << "- k1: " << distCoeffs(0) << endl;
	cout << "- k2: " << distCoeffs(1) << endl;
	if (distCoeffs.rows == 5)
		cout << "- k3: " << distCoeffs(4) << endl;
	cout << "- p1: " << distCoeffs(2) << endl;
	cout << "- p2: " << distCoeffs(3) << endl;
	cout << "- fps: " << fps << endl;

	cout << "- color order: " << (rgb ? "RGB" : "BGR") << " (ignored if grayscale)" << endl;

	cout << endl << "ORB Extractor Parameters: " << endl;
	cout << "- Number of Features: " << param.nfeatures << endl;
	cout << "- Scale Levels: " << param.nlevels << endl;
	cout << "- Scale Factor: " << param.scaleFactor << endl;
	cout << "- Initial Fast Threshold: " << param.initTh << endl;
	cout << "- Minimum Fast Threshold: " << param.minTh << endl;

	if (sensor == System::STEREO || sensor == System::RGBD)
		cout << endl << "Depth Threshold (Close/Far Points): " << thDepth << endl;
}

class InitialTracker
{

public:

	InitialTracker(Map* map, KeyFrameDatabase* keyFrameDB, LocalMap& localMap, Relocalizer& relocalizer,
		const Trajectory& trajectory, int sensor, float thDepth)
		: sensor_(sensor), fewMatches_(false), keyFrameDB_(keyFrameDB), localMap_(localMap), map_(map),
		relocalizer_(relocalizer), trajectory_(trajectory), thDepth_(thDepth)
	{
	}

	bool TrackNormal(Frame& currFrame, Frame& lastFrame, const cv::Mat& velocity, int state)
	{
		// Local Mapping is activated. This is the normal behaviour, unless
		// you explicitly activate the "only tracking" mode.

		if (state != Tracking::STATE_OK)
			return relocalizer_.Relocalize(currFrame, keyFrameDB_);

		const int minInliers = 10;

		// Local Mapping might have changed some MapPoints tracked in last frame
		for (int i = 0; i < lastFrame.N; i++)
		{
			MapPoint* mappoint = lastFrame.mvpMapPoints[i];
			MapPoint* replaced = mappoint ? mappoint->GetReplaced() : nullptr;
			if (replaced)
				lastFrame.mvpMapPoints[i] = replaced;
		}

		bool success = false;
		const bool withMotionModel = !velocity.empty() && (int)currFrame.mnId >= relocalizer_.GetLastRelocFrameId() + 2;
		if (withMotionModel)
		{
			UpdateLastFramePose(lastFrame, trajectory_.back());
			success = TrackWithMotionModel(currFrame, lastFrame, velocity, minInliers, sensor_);
		}
		if (!withMotionModel || (withMotionModel && !success))
		{
			success = TrackReferenceKeyFrame(currFrame, localMap_.mpReferenceKF, lastFrame);
		}

		return success;
	}

	bool TrackLocalization(Frame& currFrame, Frame& lastFrame, const cv::Mat& velocity, int state, int lastKeyFrameId)
	{
		// Localization Mode: Local Mapping is deactivated

		if (state != Tracking::STATE_OK)
			return relocalizer_.Relocalize(currFrame, keyFrameDB_);

		const int minInliers = 21;
		const bool createPoints = sensor_ != System::MONOCULAR && lastFrame.mnId != lastKeyFrameId;
		bool success = false;

		if (!fewMatches_)
		{
			// In last frame we tracked enough MapPoints in the map

			if (!velocity.empty())
			{
				UpdateLastFramePose(lastFrame, trajectory_.back());
				if (createPoints)
					CreateMapPointsVO(lastFrame, tempPoints_, map_, thDepth_);

				success = TrackWithMotionModel(currFrame, lastFrame, velocity, minInliers, sensor_, &fewMatches_);
			}
			else
			{
				success = TrackReferenceKeyFrame(currFrame, localMap_.mpReferenceKF, lastFrame);
			}
		}
		else
		{
			// In last frame we tracked mainly "visual odometry" points.

			// We compute two camera poses, one from motion model and one doing relocalization.
			// If relocalization is sucessfull we choose that solution, otherwise we retain
			// the "visual odometry" solution.

			bool bOKMM = false;
			bool bOKReloc = false;
			vector<MapPoint*> vpMPsMM;
			vector<bool> vbOutMM;
			cv::Mat TcwMM;
			if (!velocity.empty())
			{
				UpdateLastFramePose(lastFrame, trajectory_.back());
				if (createPoints)
					CreateMapPointsVO(lastFrame, tempPoints_, map_, thDepth_);

				bOKMM = TrackWithMotionModel(currFrame, lastFrame, velocity, minInliers, sensor_, &fewMatches_);
				vpMPsMM = currFrame.mvpMapPoints;
				vbOutMM = currFrame.mvbOutlier;
				TcwMM = currFrame.mTcw.clone();
			}
			bOKReloc = relocalizer_.Relocalize(currFrame, keyFrameDB_);

			if (bOKMM && !bOKReloc)
			{
				currFrame.SetPose(TcwMM);
				currFrame.mvpMapPoints = vpMPsMM;
				currFrame.mvbOutlier = vbOutMM;

				if (fewMatches_)
				{
					for (int i = 0; i < currFrame.N; i++)
					{
						if (currFrame.mvpMapPoints[i] && !currFrame.mvbOutlier[i])
						{
							currFrame.mvpMapPoints[i]->IncreaseFound();
						}
					}
				}
			}
			else if (bOKReloc)
			{
				fewMatches_ = false;
			}

			success = bOKReloc || bOKMM;
		}

		return success;
	}

	bool FewMatches() const
	{
		return fewMatches_;
	}

	void DeleteTemporalMapPoints()
	{
		for (MapPoint* mappoint : tempPoints_)
			delete mappoint;
		tempPoints_.clear();
	}

private:

	// Input sensor
	int sensor_;

	// In case of performing only localization, this flag is true when there are no matches to
	// points in the map. Still tracking will continue if there are enough matches with temporal points.
	// In that case we are doing visual odometry. The system will try to do relocalization to recover
	// "zero-drift" localization to the map.
	bool fewMatches_;

	//BoW
	KeyFrameDatabase* keyFrameDB_;

	//Local Map
	LocalMap& localMap_;

	//Map
	Map* map_;

	//Last Frame, KeyFrame and Relocalisation Info
	Relocalizer& relocalizer_;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	const Trajectory& trajectory_;

	list<MapPoint*> tempPoints_;

	float thDepth_;
};

class TrackerCore
{

public:

	using Parameters = TrackerParameters;

	TrackerCore(Tracking* tracking, System* system, FrameDrawer* frameDrawer, MapDrawer* mapDrawer, Map* map,
		KeyFrameDatabase* keyFrameDB, int sensor, const Parameters& param)
		: state_(STATE_NO_IMAGES), sensor_(sensor), localization_(false), keyFrameDB_(keyFrameDB),
		initializer_(nullptr), tracking_(tracking), system_(system), frameDrawer_(frameDrawer), mapDrawer_(mapDrawer),
		map_(map), localMap_(map), newKeyFrameCondition_(map, localMap_, param, sensor),
		trackerIni_(map, keyFrameDB, localMap_, relocalizer_, trajectory_, sensor, param.thDepth), param_(param)
	{
	}

	// Map initialization for stereo and RGB-D
	void StereoInitialization(Frame& currFrame)
	{
		if (currFrame.N <= 500)
			return;

		// Set Frame pose to the origin
		currFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

		// Create KeyFrame
		KeyFrame* keyframe = new KeyFrame(currFrame, map_, keyFrameDB_);

		// Insert KeyFrame in the map
		map_->AddKeyFrame(keyframe);

		// Create MapPoints and asscoiate to KeyFrame
		for (int i = 0; i < currFrame.N; i++)
		{
			const float Z = currFrame.mvDepth[i];
			if (Z <= 0.f)
				continue;

			cv::Mat Xw = currFrame.UnprojectStereo(i);
			MapPoint* mappoint = new MapPoint(Xw, keyframe, map_);
			mappoint->AddObservation(keyframe, i);
			mappoint->ComputeDistinctiveDescriptors();
			mappoint->UpdateNormalAndDepth();

			keyframe->AddMapPoint(mappoint, i);
			map_->AddMapPoint(mappoint);

			currFrame.mvpMapPoints[i] = mappoint;
		}

		cout << "New map created with " << map_->MapPointsInMap() << " points" << endl;

		localMapper_->InsertKeyFrame(keyframe);

		lastFrame_ = Frame(currFrame);
		lastKeyFrame_ = keyframe;
		CV_Assert(lastKeyFrame_->mnFrameId == currFrame.mnId);

		localMap_.mvpLocalKeyFrames.push_back(keyframe);
		localMap_.mvpLocalMapPoints = map_->GetAllMapPoints();
		localMap_.mpReferenceKF = keyframe;
		currFrame.mpReferenceKF = keyframe;

		map_->SetReferenceMapPoints(localMap_.mvpLocalMapPoints);

		map_->mvpKeyFrameOrigins.push_back(keyframe);

		mapDrawer_->SetCurrentCameraPose(currFrame.mTcw);

		state_ = STATE_OK;
	}

	// Map initialization for monocular
	void MonocularInitialization(Frame& mCurrentFrame)
	{

		if (!initializer_)
		{
			// Set Reference Frame
			if (mCurrentFrame.mvKeys.size() > 100)
			{
				mInitialFrame = Frame(mCurrentFrame);
				lastFrame_ = Frame(mCurrentFrame);
				mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
				for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
					mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

				if (initializer_)
					delete initializer_;

				initializer_ = new Initializer(mCurrentFrame, 1.0, 200);

				fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

				return;
			}
		}
		else
		{
			// Try to initialize
			if ((int)mCurrentFrame.mvKeys.size() <= 100)
			{
				delete initializer_;
				initializer_ = static_cast<Initializer*>(NULL);
				fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
				return;
			}

			// Find correspondences
			ORBmatcher matcher(0.9f, true);
			int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

			// Check if there are enough correspondences
			if (nmatches < 100)
			{
				delete initializer_;
				initializer_ = static_cast<Initializer*>(NULL);
				return;
			}

			cv::Mat Rcw; // Current Camera Rotation
			cv::Mat tcw; // Current Camera Translation
			vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

			if (initializer_->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
			{
				for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
				{
					if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
					{
						mvIniMatches[i] = -1;
						nmatches--;
					}
				}

				// Set Frame Poses
				mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
				cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
				Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
				tcw.copyTo(Tcw.rowRange(0, 3).col(3));
				mCurrentFrame.SetPose(Tcw);

				CreateInitialMapMonocular(mCurrentFrame);
			}
		}
	}

	void CreateInitialMapMonocular(Frame& mCurrentFrame)
	{
		// Create KeyFrames
		KeyFrame* pKFini = new KeyFrame(mInitialFrame, map_, keyFrameDB_);
		KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, map_, keyFrameDB_);


		pKFini->ComputeBoW();
		pKFcur->ComputeBoW();

		// Insert KFs in the map
		map_->AddKeyFrame(pKFini);
		map_->AddKeyFrame(pKFcur);

		// Create MapPoints and asscoiate to keyframes
		for (size_t i = 0; i < mvIniMatches.size(); i++)
		{
			if (mvIniMatches[i] < 0)
				continue;

			//Create MapPoint.
			cv::Mat worldPos(mvIniP3D[i]);

			MapPoint* pMP = new MapPoint(worldPos, pKFcur, map_);

			pKFini->AddMapPoint(pMP, i);
			pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

			pMP->AddObservation(pKFini, i);
			pMP->AddObservation(pKFcur, mvIniMatches[i]);

			pMP->ComputeDistinctiveDescriptors();
			pMP->UpdateNormalAndDepth();

			//Fill Current Frame structure
			mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
			mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

			//Add to Map
			map_->AddMapPoint(pMP);
		}

		// Update Connections
		pKFini->UpdateConnections();
		pKFcur->UpdateConnections();

		// Bundle Adjustment
		cout << "New Map created with " << map_->MapPointsInMap() << " points" << endl;

		Optimizer::GlobalBundleAdjustemnt(map_, 20);

		// Set median depth to 1
		float medianDepth = pKFini->ComputeSceneMedianDepth(2);
		float invMedianDepth = 1.0f / medianDepth;

		if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
		{
			cout << "Wrong initialization, reseting..." << endl;
			tracking_->Reset();
			return;
		}

		// Scale initial baseline
		cv::Mat Tc2w = pKFcur->GetPose();
		Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
		pKFcur->SetPose(Tc2w);

		// Scale points
		vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
		for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
		{
			if (vpAllMapPoints[iMP])
			{
				MapPoint* pMP = vpAllMapPoints[iMP];
				pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
			}
		}

		localMapper_->InsertKeyFrame(pKFini);
		localMapper_->InsertKeyFrame(pKFcur);

		mCurrentFrame.SetPose(pKFcur->GetPose());
		lastKeyFrame_ = pKFcur;
		CV_Assert(lastKeyFrame_->mnFrameId == mCurrentFrame.mnId);

		localMap_.mvpLocalKeyFrames.push_back(pKFcur);
		localMap_.mvpLocalKeyFrames.push_back(pKFini);
		localMap_.mvpLocalMapPoints = map_->GetAllMapPoints();
		localMap_.mpReferenceKF = pKFcur;
		mCurrentFrame.mpReferenceKF = pKFcur;

		lastFrame_ = Frame(mCurrentFrame);

		map_->SetReferenceMapPoints(localMap_.mvpLocalMapPoints);

		mapDrawer_->SetCurrentCameraPose(pKFcur->GetPose());

		map_->mvpKeyFrameOrigins.push_back(pKFini);

		state_ = STATE_OK;
	}

	void Initialization(Frame& currFrame, int sensor)
	{
		if (sensor == System::STEREO || sensor == System::RGBD)
		{
			StereoInitialization(currFrame);
		}
		else
		{
			MonocularInitialization(currFrame);
		}
	}

	// Main tracking function. It is independent of the input sensor.
	void Update(Frame& currFrame)
	{
		if (state_ == STATE_NO_IMAGES)
			state_ = STATE_NOT_INITIALIZED;

		lastProcessedState_ = state_;

		// Get Map Mutex -> Map cannot be changed
		unique_lock<mutex> lock(map_->mMutexMapUpdate);

		// Initialize Tracker if not initialized.
		if (state_ == STATE_NOT_INITIALIZED)
		{
			Initialization(currFrame, sensor_);

			frameDrawer_->Update(tracking_);

			if (state_ == STATE_OK)
				trajectory_.push_back(TrackPoint(currFrame, false));

			return;
		}

		// System is initialized. Track Frame.
		bool success = false;

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		if (!localization_)
			success = trackerIni_.TrackNormal(currFrame, lastFrame_, velocity_, state_);
		else
			success = trackerIni_.TrackLocalization(currFrame, lastFrame_, velocity_, state_, lastKeyFrame_->mnFrameId);

		currFrame.mpReferenceKF = localMap_.mpReferenceKF;

		// If we have an initial estimation of the camera pose and matching. Track the local map.
		// [In Localization Mode]
		// mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
		// a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
		// the camera we will use the local map again.
		if (success && (!localization_ || (localization_ && !trackerIni_.FewMatches())))
		{
			// If the camera has been relocalised recently, perform a coarser search
			const bool relocalizedRecently = (int)currFrame.mnId < relocalizer_.GetLastRelocFrameId() + 2;
			const float th = relocalizedRecently ? 5.f : (sensor_ == System::RGBD ? 3.f : 1.f);

			matchesInliers_ = TrackLocalMap(localMap_, currFrame, th, localization_, sensor_ == System::STEREO);

			// Decide if the tracking was succesful
			// More restrictive if there was a relocalization recently
			const int minInliers = ((int)currFrame.mnId < relocalizer_.GetLastRelocFrameId() + param_.maxFrames) ? 50 : 30;
			success = matchesInliers_ >= minInliers;
		}

		state_ = success ? STATE_OK : STATE_LOST;

		// Update drawer
		frameDrawer_->Update(tracking_);

		// If tracking were good, check if we insert a keyframe
		if (success)
		{
			// Update motion model
			if (!lastFrame_.mTcw.empty())
			{
				cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
				lastFrame_.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
				lastFrame_.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
				velocity_ = currFrame.mTcw * LastTwc;
			}
			else
				velocity_ = cv::Mat();

			mapDrawer_->SetCurrentCameraPose(currFrame.mTcw);

			// Clean VO matches
			for (int i = 0; i < currFrame.N; i++)
			{
				MapPoint* mappoint = currFrame.mvpMapPoints[i];
				if (mappoint && mappoint->Observations() < 1)
				{
					currFrame.mvbOutlier[i] = false;
					currFrame.mvpMapPoints[i] = nullptr;
				}
			}

			// Delete temporal MapPoints
			trackerIni_.DeleteTemporalMapPoints();

			// Check if we need to insert a new keyframe
			if (!localization_ && newKeyFrameCondition_.Satisfy(currFrame, localMapper_, matchesInliers_,
				relocalizer_.GetLastRelocFrameId(), lastKeyFrame_->mnFrameId))
			{
				if (localMapper_->SetNotStop(true))
				{
					KeyFrame* keyframe = new KeyFrame(currFrame, map_, keyFrameDB_);
					localMap_.mpReferenceKF = keyframe;
					currFrame.mpReferenceKF = keyframe;

					if (sensor_ != System::MONOCULAR)
						CreateMapPoints(currFrame, keyframe, map_, param_.thDepth);

					localMapper_->InsertKeyFrame(keyframe);
					localMapper_->SetNotStop(false);
					lastKeyFrame_ = keyframe;
					CV_Assert(lastKeyFrame_->mnFrameId == currFrame.mnId);
				}
			}

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame.
			for (int i = 0; i < currFrame.N; i++)
			{
				if (currFrame.mvpMapPoints[i] && currFrame.mvbOutlier[i])
					currFrame.mvpMapPoints[i] = nullptr;
			}
		}

		// Reset if the camera get lost soon after initialization
		if (state_ == STATE_LOST)
		{
			if (map_->KeyFramesInMap() <= 5)
			{
				cout << "Track lost soon after initialisation, reseting..." << endl;
				system_->Reset();
				return;
			}
		}

		CV_Assert(currFrame.mpReferenceKF);

		lastFrame_ = Frame(currFrame);

		// Store frame pose information to retrieve the complete camera trajectory afterwards.
		CV_Assert(currFrame.mpReferenceKF == localMap_.mpReferenceKF);
		const bool lost = state_ == STATE_LOST;
		if (!currFrame.mTcw.empty())
		{
			trajectory_.push_back(TrackPoint(currFrame, lost));
		}
		else
		{
			// This can happen if tracking is lost
			trajectory_.push_back(trajectory_.back());
			trajectory_.back().lost = lost;
		}
	}

	void Clear()
	{
		state_ = STATE_NO_IMAGES;

		if (initializer_)
		{
			delete initializer_;
			initializer_ = nullptr;
		}

		trajectory_.clear();
	}

	void SetLocalMapper(LocalMapping *pLocalMapper)
	{
		localMapper_ = pLocalMapper;
	}

	void SetLoopClosing(LoopClosing *pLoopClosing)
	{
		mpLoopClosing = pLoopClosing;
	}

	void InformOnlyTracking(const bool &flag)
	{
		localization_ = flag;
	}

	int GetState() const
	{
		return state_;
	}

	int GetLastProcessedState() const
	{
		return lastProcessedState_;
	}

	const Frame& GetInitialFrame() const
	{
		return mInitialFrame;
	}

	const std::vector<int>& GetIniMatches() const
	{
		return mvIniMatches;
	}

	const Trajectory& GetTrajectory() const
	{
		return trajectory_;
	}

	bool OnlyTracking() const
	{
		return localization_;
	}

private:

	// Tracking states
	enum State
	{
		STATE_NOT_READY = Tracking::STATE_NOT_READY,
		STATE_NO_IMAGES = Tracking::STATE_NO_IMAGES,
		STATE_NOT_INITIALIZED = Tracking::STATE_NOT_INITIALIZED,
		STATE_OK = Tracking::STATE_OK,
		STATE_LOST = Tracking::STATE_LOST
	};

	Tracking* tracking_;

	State state_;
	State lastProcessedState_;

	// Input sensor
	int sensor_;

	// Initialization Variables (Monocular)
	std::vector<int> mvIniLastMatches;
	std::vector<int> mvIniMatches;
	std::vector<cv::Point2f> mvbPrevMatched;
	std::vector<cv::Point3f> mvIniP3D;
	Frame mInitialFrame;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	Trajectory trajectory_;

	// True if local mapping is deactivated and we are performing only localization
	bool localization_;

	//Other Thread Pointers
	LocalMapping* localMapper_;
	LoopClosing* mpLoopClosing;

	//BoW
	KeyFrameDatabase* keyFrameDB_;

	// Initalization (only for monocular)
	Initializer* initializer_;

	//Local Map
	LocalMap localMap_;

	// System
	System* system_;

	//Drawers
	FrameDrawer* frameDrawer_;
	MapDrawer* mapDrawer_;

	//Map
	Map* map_;

	// Parameters
	TrackerParameters param_;

	//Current matches in frame
	int matchesInliers_;

	//Last Frame, KeyFrame and Relocalisation Info
	Frame lastFrame_;
	KeyFrame* lastKeyFrame_;
	Relocalizer relocalizer_;

	//Motion Model
	cv::Mat velocity_;

	InitialTracker trackerIni_;

	NewKeyFrameCondition newKeyFrameCondition_;
};

class TrackingImpl : public Tracking
{

public:

	TrackingImpl(System* system, ORBVocabulary* voc, FrameDrawer* frameDrawer, MapDrawer* mapDrawer,
		Map* map, KeyFrameDatabase* keyframeDB, const string& settingsFile, int sensor)
		: voc_(voc), keyframeDB_(keyframeDB), viewer_(nullptr), map_(map)
	{
		cv::FileStorage settings(settingsFile, cv::FileStorage::READ);

		// Load camera parameters from settings file
		camera_ = ReadCameraParams(settings);
		distCoeffs_ = ReadDistCoeffs(settings);

		// Load fps
		const float fps = ReadFps(settings);

		// Max/Min Frames to insert keyframes and to check relocalisation
		const int minFrames = 0;
		const int maxFrames = static_cast<int>(fps);

		// Load color
		RGB_ = static_cast<int>(settings["Camera.RGB"]) != 0;

		// Load ORB parameters
		const ORBExtractorParams extractorParams = ReadExtractorParams(settings);

		// Load depth threshold
		const float thDepth = settings["ThDepth"];
		thDepth_ = camera_.baseline * thDepth;

		// Load depth factor
		depthFactor_ = sensor == System::RGBD ? ReadDepthFactor(settings) : 1.f;

		// Print settings
		PrintSettings(camera_, distCoeffs_, fps, RGB_, extractorParams, thDepth_, sensor);

		// Initialize ORB extractors
		const int nfeatures = extractorParams.nfeatures;
		const float scaleFactor = extractorParams.scaleFactor;
		const int nlevels = extractorParams.nlevels;
		const int initTh = extractorParams.initTh;
		const int minTh = extractorParams.minTh;

		extractorL_ = std::make_unique<ORBextractor>(nfeatures, scaleFactor, nlevels, initTh, minTh);
		extractorR_ = std::make_unique<ORBextractor>(nfeatures, scaleFactor, nlevels, initTh, minTh);
		extractorIni_ = std::make_unique<ORBextractor>(2 * nfeatures, scaleFactor, nlevels, initTh, minTh);

		// Initialize tracker core
		tracker_ = std::make_unique<TrackerCore>(this, system, frameDrawer, mapDrawer, map, keyframeDB, sensor,
			TrackerCore::Parameters(minFrames, maxFrames, thDepth_));
	}

	// Preprocess the input and call Track(). Extract features and performs stereo matching.
	cv::Mat GrabImageStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) override
	{
		ConvertToGray(imageL, imageL_, RGB_);
		ConvertToGray(imageR, imageR_, RGB_);

		currFrame_ = Frame(imageL_, imageR_, timestamp, extractorL_.get(), extractorR_.get(), voc_,
			camera_.Mat(), distCoeffs_, camera_.bf, thDepth_);

		tracker_->Update(currFrame_);

		return currFrame_.mTcw.clone();
	}

	cv::Mat GrabImageRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) override
	{
		ConvertToGray(image, imageL_, RGB_);

		depth.convertTo(depth_, CV_32F, depthFactor_);

		currFrame_ = Frame(imageL_, depth_, timestamp, extractorL_.get(), voc_,
			camera_.Mat(), distCoeffs_, camera_.bf, thDepth_);

		tracker_->Update(currFrame_);

		return currFrame_.mTcw.clone();
	}

	cv::Mat GrabImageMonocular(const cv::Mat& image, double timestamp) override
	{
		ConvertToGray(image, imageL_, RGB_);

		const int state = tracker_->GetState();
		const bool init = state == STATE_NOT_INITIALIZED || state == STATE_NO_IMAGES;

		ORBextractor* pORBextractor = init ? extractorIni_.get() : extractorL_.get();

		currFrame_ = Frame(imageL_, timestamp, pORBextractor, voc_,
			camera_.Mat(), distCoeffs_, camera_.bf, thDepth_);

		tracker_->Update(currFrame_);

		return currFrame_.mTcw.clone();
	}

	void SetLocalMapper(LocalMapping* localMapper) override
	{
		localMapper_ = localMapper;
		tracker_->SetLocalMapper(localMapper);
	}

	void SetLoopClosing(LoopClosing* loopClosing) override
	{
		loopClosing_ = loopClosing;
		tracker_->SetLoopClosing(loopClosing);
	}

	void SetViewer(Viewer* viewer) override
	{
		viewer_ = viewer;
	}

	// Load new settings
	// The focal lenght should be similar or scale prediction will fail when projecting points
	// TODO: Modify MapPoint::PredictScale to take into account focal lenght
	void ChangeCalibration(const string& settingsFile) override
	{
		cv::FileStorage settings(settingsFile, cv::FileStorage::READ);
		camera_ = ReadCameraParams(settings);
		distCoeffs_ = ReadDistCoeffs(settings);
		Frame::mbInitialComputations = true;
	}

	void Reset() override
	{
		cout << "System Reseting" << endl;
		if (viewer_)
		{
			viewer_->RequestStop();
			while (!viewer_->isStopped())
				usleep(3000);
		}

		// Reset Local Mapping
		cout << "Reseting Local Mapper...";
		localMapper_->RequestReset();
		cout << " done" << endl;

		// Reset Loop Closing
		cout << "Reseting Loop Closing...";
		loopClosing_->RequestReset();
		cout << " done" << endl;

		// Clear BoW Database
		cout << "Reseting Database...";
		keyframeDB_->clear();
		cout << " done" << endl;

		// Clear Map (this erase MapPoints and KeyFrames)
		map_->clear();

		KeyFrame::nNextId = 0;
		Frame::nNextId = 0;

		tracker_->Clear();

		if (viewer_)
			viewer_->Release();
	}

	// Use this function if you have deactivated local mapping and you only want to localize the camera.
	void InformOnlyTracking(bool flag) override
	{
		tracker_->InformOnlyTracking(flag);
	}

	int GetState() const override
	{
		return tracker_->GetState();
	}

	int GetLastProcessedState() const override
	{
		return tracker_->GetLastProcessedState();
	}

	const Frame& GetCurrentFrame() const override
	{
		return currFrame_;
	}

	const Frame& GetInitialFrame() const override
	{
		return tracker_->GetInitialFrame();
	}

	cv::Mat GetImGray() const override
	{
		return imageL_;
	}

	const std::vector<int>& GetIniMatches() const override
	{
		return tracker_->GetIniMatches();
	}

	const Trajectory& GetTrajectory() const override
	{
		return tracker_->GetTrajectory();
	}

	bool OnlyTracking() const override
	{
		return tracker_->OnlyTracking();
	}

private:

	// Current Frame
	Frame currFrame_;
	cv::Mat imageL_;
	cv::Mat imageR_;
	cv::Mat depth_;

	//Other Thread Pointers
	LocalMapping* localMapper_;
	LoopClosing* loopClosing_;

	// ORB
	std::unique_ptr<ORBextractor> extractorL_;
	std::unique_ptr<ORBextractor> extractorR_;
	std::unique_ptr<ORBextractor> extractorIni_;

	// BoW
	ORBVocabulary* voc_;
	KeyFrameDatabase* keyframeDB_;

	// Drawers
	Viewer* viewer_;

	//Map
	Map* map_;

	// Calibration matrix
	CameraParams camera_;
	cv::Mat1f distCoeffs_;

	// For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
	float depthFactor_;

	// Color order (true RGB, false BGR, ignored if grayscale)
	bool RGB_;

	std::unique_ptr<TrackerCore> tracker_;
	float thDepth_;
};

std::shared_ptr<Tracking> Tracking::Create(System* system, ORBVocabulary* voc, FrameDrawer* frameDrawer,
	MapDrawer* mapDrawer, Map* map, KeyFrameDatabase* keyframeDB, const string& settingsFile, int sensor)
{
	return std::make_shared<TrackingImpl>(system, voc, frameDrawer, mapDrawer, map, keyframeDB, settingsFile, sensor);
}

} //namespace ORB_SLAM
