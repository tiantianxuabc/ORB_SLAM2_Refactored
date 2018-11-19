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

#include "LocalMapping.h"

#include "Tracking.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Usleep.h"
#include "KeyFrame.h"
#include "Map.h"

#include<mutex>

#define LOCK_MUTEX_NEW_KF()    std::unique_lock<std::mutex> lock1(mutexNewKFs_);
#define LOCK_MUTEX_RESET()     std::unique_lock<std::mutex> lock2(mutexReset_);
#define LOCK_MUTEX_FINISH()    std::unique_lock<std::mutex> lock3(mutexFinish_);
#define LOCK_MUTEX_STOP()      std::unique_lock<std::mutex> lock4(mutexStop_);
#define LOCK_MUTEX_ACCEPT_KF() std::unique_lock<std::mutex> lock5(mutexAccept_);

namespace ORB_SLAM2
{

static cv::Mat SkewSymmetricMatrix(const cv::Mat1f& v)
{
	return (cv::Mat1f(3, 3) << 
		    0, -v(2),  v(1),
		 v(2),     0, -v(0),
		-v(1),  v(0),    0);
}

static cv::Mat ComputeF12(KeyFrame* keyframe1, KeyFrame* keyframe2)
{
	const cv::Mat R1w = keyframe1->GetRotation();
	const cv::Mat t1w = keyframe1->GetTranslation();
	const cv::Mat R2w = keyframe2->GetRotation();
	const cv::Mat t2w = keyframe2->GetTranslation();

	const cv::Mat R12 = R1w * R2w.t();
	const cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

	const cv::Mat t12x = SkewSymmetricMatrix(t12);

	const cv::Mat& K1 = keyframe1->camera.Mat();
	const cv::Mat& K2 = keyframe2->camera.Mat();

	return K1.t().inv() * t12x * R12 * K2.inv();
}

class LocalMappingImpl : public LocalMapping
{
public:

	LocalMappingImpl::LocalMappingImpl(Map* map, bool monocular) :
		monocular_(monocular), resetRequested_(false), finishRequested_(false), finished_(true), map_(map),
		abortBA_(false), stopped_(false), stopRequested_(false), notStop_(false), acceptKeyFrames_(true)
	{
	}

	void SetLoopCloser(LoopClosing* loopCloser) override
	{
		loopCloser_ = loopCloser;
	}

	void SetTracker(Tracking* tracker) override
	{
		tracker_ = tracker;
	}

	// Main function
	void Run() override
	{
		finished_ = false;

		while (1)
		{
			// Tracking will see that Local Mapping is busy
			SetAcceptKeyFrames(false);

			// Check if there are keyframes in the queue
			if (CheckNewKeyFrames())
			{
				// BoW conversion and insertion in Map
				ProcessNewKeyFrame();

				// Check recent MapPoints
				MapPointCulling();

				// Triangulate new MapPoints
				CreateNewMapPoints();

				if (!CheckNewKeyFrames())
				{
					// Find more matches in neighbor keyframes and fuse point duplications
					SearchInNeighbors();
				}

				abortBA_ = false;

				if (!CheckNewKeyFrames() && !stopRequested())
				{
					// Local BA
					if (map_->KeyFramesInMap() > 2)
						Optimizer::LocalBundleAdjustment(currKeyFrame_, &abortBA_, map_);

					// Check redundant local Keyframes
					KeyFrameCulling();
				}

				loopCloser_->InsertKeyFrame(currKeyFrame_);
			}
			else if (Stop())
			{
				// Safe area to stop
				while (isStopped() && !CheckFinish())
				{
					usleep(3000);
				}
				if (CheckFinish())
					break;
			}

			ResetIfRequested();

			// Tracking will see that Local Mapping is busy
			SetAcceptKeyFrames(true);

			if (CheckFinish())
				break;

			usleep(3000);
		}

		SetFinish();
	}

	void InsertKeyFrame(KeyFrame* keyframe) override
	{
		LOCK_MUTEX_NEW_KF();
		newKeyFrames_.push_back(keyframe);
		abortBA_ = true;
	}

	// Thread Synch
	void RequestStop() override
	{
		LOCK_MUTEX_STOP();
		stopRequested_ = true;
		LOCK_MUTEX_NEW_KF();
		abortBA_ = true;
	}

	void RequestReset() override
	{
		{
			LOCK_MUTEX_RESET();
			resetRequested_ = true;
		}

		while (1)
		{
			{
				LOCK_MUTEX_RESET();
				if (!resetRequested_)
					break;
			}
			usleep(3000);
		}
	}

	bool Stop() override
	{
		LOCK_MUTEX_STOP();
		if (stopRequested_ && !notStop_)
		{
			stopped_ = true;
			cout << "Local Mapping STOP" << endl;
			return true;
		}

		return false;
	}

	void Release() override
	{
		LOCK_MUTEX_STOP();
		LOCK_MUTEX_FINISH();

		if (finished_)
			return;

		stopped_ = false;
		stopRequested_ = false;
		for (list<KeyFrame*>::iterator lit = newKeyFrames_.begin(), lend = newKeyFrames_.end(); lit != lend; lit++)
			delete *lit;
		newKeyFrames_.clear();

		cout << "Local Mapping RELEASE" << endl;
	}

	bool isStopped() const override
	{
		LOCK_MUTEX_STOP();
		return stopped_;
	}

	bool stopRequested() const override
	{
		LOCK_MUTEX_STOP();
		return stopRequested_;
	}

	bool AcceptKeyFrames() const override
	{
		LOCK_MUTEX_ACCEPT_KF();
		return acceptKeyFrames_;
	}

	void SetAcceptKeyFrames(bool flag) override
	{
		LOCK_MUTEX_ACCEPT_KF();
		acceptKeyFrames_ = flag;
	}

	bool SetNotStop(bool flag) override
	{
		LOCK_MUTEX_STOP();

		if (flag && stopped_)
			return false;

		notStop_ = flag;

		return true;
	}

	void InterruptBA() override
	{
		abortBA_ = true;
	}

	void RequestFinish() override
	{
		LOCK_MUTEX_FINISH();
		finishRequested_ = true;
	}

	bool isFinished() const override
	{
		LOCK_MUTEX_FINISH();
		return finished_;
	}

	int KeyframesInQueue() const override
	{
		LOCK_MUTEX_NEW_KF();
		return static_cast<int>(newKeyFrames_.size());
	}

private:

	bool CheckNewKeyFrames()
	{
		LOCK_MUTEX_NEW_KF();
		return(!newKeyFrames_.empty());
	}

	void ProcessNewKeyFrame()
	{
		{
			LOCK_MUTEX_NEW_KF();
			currKeyFrame_ = newKeyFrames_.front();
			newKeyFrames_.pop_front();
		}

		// Compute Bags of Words structures
		currKeyFrame_->ComputeBoW();

		// Associate MapPoints to the new keyframe and update normal and descriptor
		const std::vector<MapPoint*> mapopints = currKeyFrame_->GetMapPointMatches();
		for (size_t i = 0; i < mapopints.size(); i++)
		{
			MapPoint* mappoint = mapopints[i];
			if (!mappoint || mappoint->isBad())
				continue;

			if (!mappoint->IsInKeyFrame(currKeyFrame_))
			{
				mappoint->AddObservation(currKeyFrame_, i);
				mappoint->UpdateNormalAndDepth();
				mappoint->ComputeDistinctiveDescriptors();
			}
			else // this can only happen for new stereo points inserted by the Tracking
			{
				recentAddedMapPoints_.push_back(mappoint);
			}
		}

		// Update links in the Covisibility Graph
		currKeyFrame_->UpdateConnections();

		// Insert Keyframe in Map
		map_->AddKeyFrame(currKeyFrame_);
	}

	void MapPointCulling()
	{
		// Check Recent Added MapPoints
		const int currKFId = static_cast<int>(currKeyFrame_->mnId);

		const int minObservation = monocular_ ? 2 : 3;

		for (auto it = std::begin(recentAddedMapPoints_); it != std::end(recentAddedMapPoints_);)
		{
			MapPoint* mappoint = *it;
			const int firstKFId = mappoint->mnFirstKFid;
			if (mappoint->isBad())
			{
				it = recentAddedMapPoints_.erase(it);
			}
			else if (mappoint->GetFoundRatio() < 0.25f)
			{
				mappoint->SetBadFlag();
				it = recentAddedMapPoints_.erase(it);
			}
			else if ((currKFId - firstKFId) >= 2 && mappoint->Observations() <= minObservation)
			{
				mappoint->SetBadFlag();
				it = recentAddedMapPoints_.erase(it);
			}
			else if ((currKFId - firstKFId) >= 3)
			{
				it = recentAddedMapPoints_.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	void CreateNewMapPoints()
	{
		KeyFrame* keyframe1 = currKeyFrame_;

		// Retrieve neighbor keyframes in covisibility graph
		const int nneighbors = monocular_ ? 20 : 10;
		const std::vector<KeyFrame*> neighborKFs = keyframe1->GetBestCovisibilityKeyFrames(nneighbors);

		ORBmatcher matcher(0.6f, false);

		const cv::Mat Rcw1 = keyframe1->GetRotation();
		const cv::Mat Rwc1 = Rcw1.t();
		const cv::Mat tcw1 = keyframe1->GetTranslation();
		const cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));
		const cv::Mat Ow1 = keyframe1->GetCameraCenter();

		const float fx1 = keyframe1->camera.fx;
		const float fy1 = keyframe1->camera.fy;
		const float cx1 = keyframe1->camera.cx;
		const float cy1 = keyframe1->camera.cy;
		const float invfx1 = 1.f / fx1;
		const float invfy1 = 1.f / fy1;

		const float ratioFactor = 1.5f * keyframe1->pyramid.scaleFactor;

		int nnew = 0;

		// Search matches with epipolar restriction and triangulate
		for (size_t i = 0; i < neighborKFs.size(); i++)
		{
			if (i > 0 && CheckNewKeyFrames())
				return;

			KeyFrame* keyframe2 = neighborKFs[i];

			// Check first that baseline is not too short
			const cv::Mat Ow2 = keyframe2->GetCameraCenter();
			const float baseline = static_cast<float>(cv::norm(Ow2 - Ow1));

			if (!monocular_)
			{
				if (baseline < keyframe2->camera.baseline)
					continue;
			}
			else
			{
				const float medianDepthKF2 = keyframe2->ComputeSceneMedianDepth(2);
				const float ratioBaselineDepth = baseline / medianDepthKF2;

				if (ratioBaselineDepth < 0.01f)
					continue;
			}

			// Compute Fundamental Matrix
			const cv::Mat F12 = ComputeF12(keyframe1, keyframe2);

			// Search matches that fullfil epipolar constraint
			std::vector<std::pair<size_t, size_t> > matchIndices;
			matcher.SearchForTriangulation(keyframe1, keyframe2, F12, matchIndices, false);

			const cv::Mat Rcw2 = keyframe2->GetRotation();
			const cv::Mat Rwc2 = Rcw2.t();
			const cv::Mat tcw2 = keyframe2->GetTranslation();
			const cv::Mat Tcw2(3, 4, CV_32F);
			Rcw2.copyTo(Tcw2.colRange(0, 3));
			tcw2.copyTo(Tcw2.col(3));

			const float fx2 = keyframe2->camera.fx;
			const float fy2 = keyframe2->camera.fy;
			const float cx2 = keyframe2->camera.cx;
			const float cy2 = keyframe2->camera.cy;
			const float invfx2 = 1.f / fx2;
			const float invfy2 = 1.f / fy2;

			// Triangulate each match
			const int nmatches = static_cast<int>(matchIndices.size());
			for (int ikp = 0; ikp < nmatches; ikp++)
			{
				const int idx1 = static_cast<int>(matchIndices[ikp].first);
				const int idx2 = static_cast<int>(matchIndices[ikp].second);

				const cv::KeyPoint& keypoint1 = keyframe1->mvKeysUn[idx1];
				const float ur1 = keyframe1->mvuRight[idx1];
				const bool stereo1 = ur1 >= 0;

				const cv::KeyPoint& keypoint2 = keyframe2->mvKeysUn[idx2];
				const float ur2 = keyframe2->mvuRight[idx2];
				const bool stereo2 = ur2 >= 0;

				// Check parallax between rays
				cv::Mat xn1 = (cv::Mat1f(3, 1) << (keypoint1.pt.x - cx1) * invfx1, (keypoint1.pt.y - cy1) * invfy1, 1.f);
				cv::Mat xn2 = (cv::Mat1f(3, 1) << (keypoint2.pt.x - cx2) * invfx2, (keypoint2.pt.y - cy2) * invfy2, 1.f);

				const cv::Mat ray1 = Rwc1 * xn1;
				const cv::Mat ray2 = Rwc2 * xn2;
				const float cosParallaxRays = static_cast<float>(ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2)));

				float cosParallaxStereo = cosParallaxRays + 1;
				float cosParallaxStereo1 = cosParallaxStereo;
				float cosParallaxStereo2 = cosParallaxStereo;

				if (stereo1)
					cosParallaxStereo1 = cos(2 * atan2(keyframe1->camera.baseline / 2, keyframe1->mvDepth[idx1]));
				else if (stereo2)
					cosParallaxStereo2 = cos(2 * atan2(keyframe2->camera.baseline / 2, keyframe2->mvDepth[idx2]));

				cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

				cv::Mat x3D;
				if (cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (stereo1 || stereo2 || cosParallaxRays < 0.9998))
				{
					// Linear Triangulation Method
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
					A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
					A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

					cv::Mat w, u, vt;
					cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

					x3D = vt.row(3).t();

					if (x3D.at<float>(3) == 0)
						continue;

					// Euclidean coordinates
					x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

				}
				else if (stereo1 && cosParallaxStereo1 < cosParallaxStereo2)
				{
					x3D = keyframe1->UnprojectStereo(idx1);
				}
				else if (stereo2 && cosParallaxStereo2 < cosParallaxStereo1)
				{
					x3D = keyframe2->UnprojectStereo(idx2);
				}
				else
					continue; //No stereo and very low parallax

				cv::Mat x3Dt = x3D.t();

				//Check triangulation in front of cameras
				const float z1 = static_cast<float>(Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2));
				if (z1 <= 0)
					continue;

				const float z2 = static_cast<float>(Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2));
				if (z2 <= 0)
					continue;

				//Check reprojection error in first keyframe
				const float sigmaSquare1 = keyframe1->pyramid.sigmaSq[keypoint1.octave];
				const float x1 = static_cast<float>(Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0));
				const float y1 = static_cast<float>(Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1));
				const float invz1 = 1.f / z1;

				if (!stereo1)
				{
					float u1 = fx1*x1*invz1 + cx1;
					float v1 = fy1*y1*invz1 + cy1;
					float errX1 = u1 - keypoint1.pt.x;
					float errY1 = v1 - keypoint1.pt.y;
					if ((errX1*errX1 + errY1*errY1) > 5.991*sigmaSquare1)
						continue;
				}
				else
				{
					float u1 = fx1*x1*invz1 + cx1;
					float u1_r = u1 - keyframe1->camera.bf*invz1;
					float v1 = fy1*y1*invz1 + cy1;
					float errX1 = u1 - keypoint1.pt.x;
					float errY1 = v1 - keypoint1.pt.y;
					float errX1_r = u1_r - ur1;
					if ((errX1*errX1 + errY1*errY1 + errX1_r*errX1_r) > 7.8*sigmaSquare1)
						continue;
				}

				//Check reprojection error in second keyframe
				const float sigmaSquare2 = keyframe2->pyramid.sigmaSq[keypoint2.octave];
				const float x2 = static_cast<float>(Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0));
				const float y2 = static_cast<float>(Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1));
				const float invz2 = 1.f / z2;
				if (!stereo2)
				{
					float u2 = fx2*x2*invz2 + cx2;
					float v2 = fy2*y2*invz2 + cy2;
					float errX2 = u2 - keypoint2.pt.x;
					float errY2 = v2 - keypoint2.pt.y;
					if ((errX2*errX2 + errY2*errY2) > 5.991*sigmaSquare2)
						continue;
				}
				else
				{
					float u2 = fx2*x2*invz2 + cx2;
					float u2_r = u2 - keyframe1->camera.bf*invz2;
					float v2 = fy2*y2*invz2 + cy2;
					float errX2 = u2 - keypoint2.pt.x;
					float errY2 = v2 - keypoint2.pt.y;
					float errX2_r = u2_r - ur2;
					if ((errX2*errX2 + errY2*errY2 + errX2_r*errX2_r) > 7.8*sigmaSquare2)
						continue;
				}

				//Check scale consistency
				cv::Mat normal1 = x3D - Ow1;
				float dist1 = static_cast<float>(cv::norm(normal1));

				cv::Mat normal2 = x3D - Ow2;
				float dist2 = static_cast<float>(cv::norm(normal2));

				if (dist1 == 0 || dist2 == 0)
					continue;

				const float ratioDist = dist2 / dist1;
				const float ratioOctave = keyframe1->pyramid.scaleFactors[keypoint1.octave] / keyframe2->pyramid.scaleFactors[keypoint2.octave];

				if (ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
					continue;

				// Triangulation is succesfull
				MapPoint* pMP = new MapPoint(x3D, keyframe1, map_);

				pMP->AddObservation(keyframe1, idx1);
				pMP->AddObservation(keyframe2, idx2);

				keyframe1->AddMapPoint(pMP, idx1);
				keyframe2->AddMapPoint(pMP, idx2);

				pMP->ComputeDistinctiveDescriptors();

				pMP->UpdateNormalAndDepth();

				map_->AddMapPoint(pMP);
				recentAddedMapPoints_.push_back(pMP);

				nnew++;
			}
		}
	}

	void SearchInNeighbors()
	{
		// Retrieve neighbor keyframes
		const int nneighbors = monocular_ ? 20 : 10;
		std::vector<KeyFrame*> targetKFs;
		for (KeyFrame* neighborKF : currKeyFrame_->GetBestCovisibilityKeyFrames(nneighbors))
		{
			if (neighborKF->isBad() || neighborKF->mnFuseTargetForKF == currKeyFrame_->mnId)
				continue;

			targetKFs.push_back(neighborKF);
			neighborKF->mnFuseTargetForKF = currKeyFrame_->mnId;

			// Extend to some second neighbors
			for (KeyFrame* secondKF : neighborKF->GetBestCovisibilityKeyFrames(5))
			{
				if (secondKF->isBad() || secondKF->mnFuseTargetForKF == currKeyFrame_->mnId || secondKF->mnId == currKeyFrame_->mnId)
					continue;
				targetKFs.push_back(secondKF);
			}
		}
		
		// Search matches by projection from current KF in target KFs
		ORBmatcher matcher;
		std::vector<MapPoint*> mappoints = currKeyFrame_->GetMapPointMatches();
		for (KeyFrame* targetKF : targetKFs)
			matcher.Fuse(targetKF, mappoints);

		// Search matches by projection from target KFs in current KF
		std::vector<MapPoint*> fuseCandidates;
		fuseCandidates.reserve(targetKFs.size() * mappoints.size());

		for (KeyFrame* targetKF : targetKFs)
		{
			for (MapPoint* mappoint : targetKF->GetMapPointMatches())
			{
				if (!mappoint || mappoint->isBad() || mappoint->mnFuseCandidateForKF == currKeyFrame_->mnId)
					continue;

				mappoint->mnFuseCandidateForKF = currKeyFrame_->mnId;
				fuseCandidates.push_back(mappoint);
			}
		}

		matcher.Fuse(currKeyFrame_, fuseCandidates);

		// Update points
		mappoints = currKeyFrame_->GetMapPointMatches();
		for (MapPoint* mappoint : mappoints)
		{
			if (!mappoint || mappoint->isBad())
				continue;

			mappoint->ComputeDistinctiveDescriptors();
			mappoint->UpdateNormalAndDepth();
		}

		// Update connections in covisibility graph
		currKeyFrame_->UpdateConnections();
	}

	void KeyFrameCulling()
	{
		// Check redundant keyframes (only local keyframes)
		// A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
		// in at least other 3 keyframes (in the same or finer scale)
		// We only consider close stereo points
		vector<KeyFrame*> vpLocalKeyFrames = currKeyFrame_->GetVectorCovisibleKeyFrames();

		for (vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
		{
			KeyFrame* pKF = *vit;
			if (pKF->mnId == 0)
				continue;
			const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

			int nObs = 3;
			const int thObs = nObs;
			int nRedundantObservations = 0;
			int nMPs = 0;
			for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isBad())
					{
						if (!monocular_)
						{
							if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
								continue;
						}

						nMPs++;
						if (pMP->Observations() > thObs)
						{
							const int scaleLevel = pKF->mvKeysUn[i].octave;
							const map<KeyFrame*, size_t> observations = pMP->GetObservations();
							int nObs = 0;
							for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
							{
								KeyFrame* pKFi = mit->first;
								if (pKFi == pKF)
									continue;
								const int scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

								if (scaleLeveli <= scaleLevel + 1)
								{
									nObs++;
									if (nObs >= thObs)
										break;
								}
							}
							if (nObs >= thObs)
							{
								nRedundantObservations++;
							}
						}
					}
				}
			}

			if (nRedundantObservations > 0.9*nMPs)
				pKF->SetBadFlag();
		}
	}

	void ResetIfRequested()
	{
		LOCK_MUTEX_RESET();
		if (resetRequested_)
		{
			newKeyFrames_.clear();
			recentAddedMapPoints_.clear();
			resetRequested_ = false;
		}
	}

	bool CheckFinish()
	{
		LOCK_MUTEX_FINISH();
		return finishRequested_;
	}

	void SetFinish()
	{
		LOCK_MUTEX_FINISH();
		finished_ = true;
		LOCK_MUTEX_STOP();
		stopped_ = true;
	}

	bool monocular_;
	bool resetRequested_;
	bool finishRequested_;
	bool finished_;

	Map* map_;

	LoopClosing* loopCloser_;
	Tracking* tracker_;

	std::list<KeyFrame*> newKeyFrames_;

	KeyFrame* currKeyFrame_;

	std::list<MapPoint*> recentAddedMapPoints_;

	bool abortBA_;
	bool stopped_;
	bool stopRequested_;
	bool notStop_;
	bool acceptKeyFrames_;

	mutable std::mutex mutexNewKFs_;
	mutable std::mutex mutexReset_;
	mutable std::mutex mutexFinish_;
	mutable std::mutex mutexStop_;
	mutable std::mutex mutexAccept_;
};

std::shared_ptr<LocalMapping> LocalMapping::Create(Map* map, bool monocular)
{
	return std::make_shared<LocalMappingImpl>(map, monocular);
}

} //namespace ORB_SLAM
