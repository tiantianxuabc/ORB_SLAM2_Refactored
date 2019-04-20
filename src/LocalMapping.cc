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

#include "LocalMapping.h"

#include <mutex>

#include "Tracking.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Usleep.h"
#include "KeyFrame.h"
#include "Map.h"
#include "Optimizer.h"
#include "CameraProjection.h"

#define LOCK_MUTEX_NEW_KF()    std::unique_lock<std::mutex> lock1(mutexNewKFs_);
#define LOCK_MUTEX_RESET()     std::unique_lock<std::mutex> lock2(mutexReset_);
#define LOCK_MUTEX_FINISH()    std::unique_lock<std::mutex> lock3(mutexFinish_);
#define LOCK_MUTEX_STOP()      std::unique_lock<std::mutex> lock4(mutexStop_);
#define LOCK_MUTEX_ACCEPT_KF() std::unique_lock<std::mutex> lock5(mutexAccept_);

namespace ORB_SLAM2
{

static inline cv::Matx33f SkewSymmetricMatrix(const Vec3D& v)
{
	const float x = v(0);
	const float y = v(1);
	const float z = v(2);

	return cv::Matx33f(
		0, -z, y,
		z, 0, -x,
		-y, x, 0);
}

static cv::Mat ComputeF12(KeyFrame* keyframe1, KeyFrame* keyframe2)
{
	const auto R1w = keyframe1->GetPose().R();
	const auto t1w = keyframe1->GetPose().t();
	const auto R2w = keyframe2->GetPose().R();
	const auto t2w = keyframe2->GetPose().t();

	const auto R12 = R1w * R2w.t();
	const auto t12 = -R1w * R2w.t() * t2w + t1w;

	const auto t12x = SkewSymmetricMatrix(t12);

	const auto K1 = cv::Matx33f(keyframe1->camera.Mat());
	const auto K2 = cv::Matx33f(keyframe2->camera.Mat());

	return cv::Mat(K1.t().inv() * t12x * R12 * K2.inv());
}

class LocalMappingImpl : public LocalMapping
{
public:

	LocalMappingImpl(Map* map, bool monocular, float thDepth) :
		monocular_(monocular), resetRequested_(false), finishRequested_(false), finished_(true), map_(map),
		abortBA_(false), stopped_(false), stopRequested_(false), notStop_(false), acceptKeyFrames_(true), thDepth_(thDepth)
	{
	}

	void SetTracker(Tracking* tracker) override
	{
		tracker_ = tracker;
	}

	void SetLoopCloser(LoopClosing* loopCloser) override
	{
		loopCloser_ = loopCloser;
	}

	void Update()
	{
		KeyFrame* currKeyFrame_;
		{
			LOCK_MUTEX_NEW_KF();
			currKeyFrame_ = newKeyFrames_.front();
			newKeyFrames_.pop_front();
		}

		// BoW conversion and insertion in Map
		ProcessNewKeyFrame(currKeyFrame_);

		// Check recent MapPoints
		MapPointCulling(currKeyFrame_);

		// Triangulate new MapPoints
		CreateNewMapPoints(currKeyFrame_);

		if (!CheckNewKeyFrames())
		{
			// Find more matches in neighbor keyframes and fuse point duplications
			SearchInNeighbors(currKeyFrame_);
		}

		abortBA_ = false;

		if (!CheckNewKeyFrames() && !stopRequested())
		{
			// Local BA
			if (map_->KeyFramesInMap() > 2)
				Optimizer::LocalBundleAdjustment(currKeyFrame_, &abortBA_, map_);

			// Check redundant local Keyframes
			KeyFrameCulling(currKeyFrame_);
		}

		loopCloser_->InsertKeyFrame(currKeyFrame_);
	}

	// Main function
	void Run() override
	{
		finished_ = false;

		while (true)
		{
			// Tracking will see that Local Mapping is busy
			SetAcceptKeyFrames(false);

			// Check if there are keyframes in the queue
			if (CheckNewKeyFrames())
			{
				Update();
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

		while (true)
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
			std::cout << "Local Mapping STOP" << std::endl;
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
		for (KeyFrame* keyframe : newKeyFrames_)
			delete keyframe;
		newKeyFrames_.clear();

		std::cout << "Local Mapping RELEASE" << std::endl;
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

	void ProcessNewKeyFrame(KeyFrame* currKeyFrame_)
	{
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

	void MapPointCulling(KeyFrame* currKeyFrame_)
	{
		// Check Recent Added MapPoints
		const int currKFId = static_cast<int>(currKeyFrame_->id);

		const int minObservation = monocular_ ? 2 : 3;

		for (auto it = std::begin(recentAddedMapPoints_); it != std::end(recentAddedMapPoints_);)
		{
			MapPoint* mappoint = *it;
			const int firstKFId = mappoint->firstKFid;
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

	static inline float CosAngle(const Vec3D& v1, const Vec3D& v2)
	{ 
		return static_cast<float>(v1.dot(v2) / (cv::norm(v1) * cv::norm(v2)));
	}

	static inline float Parallax(float baseline, float Z) { return 2.f * atan2f(0.5f * baseline, Z); }
	static inline float NormSq(float x, float y) { return x * x + y * y; }
	static inline float NormSq(float x, float y, float z) { return x * x + y * y + z * z; }

	void CreateNewMapPoints(KeyFrame* currKeyFrame_)
	{
		KeyFrame* keyframe1 = currKeyFrame_;

		// Retrieve neighbor keyframes in covisibility graph
		const int nneighbors = monocular_ ? 20 : 10;
		const std::vector<KeyFrame*> neighborKFs = keyframe1->GetBestCovisibilityKeyFrames(nneighbors);

		ORBmatcher matcher(0.6f, false);

		const CameraPose pose1 = keyframe1->GetPose();
		const CameraProjection proj1(pose1, keyframe1->camera);
		const CameraUnProjection unproj1(pose1, keyframe1->camera);
		const Point3D Ow1 = keyframe1->GetCameraCenter();
		const cv::Mat Tcw1 = pose1.Mat();

		const float ratioFactor = 1.5f * keyframe1->pyramid.scaleFactor;

		// Search matches with epipolar restriction and triangulate
		for (size_t i = 0; i < neighborKFs.size(); i++)
		{
			if (i > 0 && CheckNewKeyFrames())
				return;

			KeyFrame* keyframe2 = neighborKFs[i];

			// Check first that baseline is not too short
			const Point3D Ow2 = keyframe2->GetCameraCenter();
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

			const CameraPose pose2 = keyframe2->GetPose();
			const CameraProjection proj2(pose2, keyframe2->camera);
			const CameraUnProjection unproj2(pose2, keyframe2->camera);
			const cv::Mat Tcw2 = pose2.Mat();

			// Triangulate each match
			for (const auto& matchIdx : matchIndices)
			{
				const int idx1 = static_cast<int>(matchIdx.first);
				const int idx2 = static_cast<int>(matchIdx.second);

				const cv::KeyPoint& keypoint1 = keyframe1->keypointsUn[idx1];
				const cv::KeyPoint& keypoint2 = keyframe2->keypointsUn[idx2];
				const float ur1 = keyframe1->uright[idx1];
				const float ur2 = keyframe2->uright[idx2];
				const float Z1 = keyframe1->depth[idx1];
				const float Z2 = keyframe2->depth[idx2];
				const bool stereo1 = ur1 >= 0;
				const bool stereo2 = ur2 >= 0;

				// Check parallax between rays
				const Vec3D xn1 = unproj1.uvZToCamera(keypoint1.pt.x, keypoint1.pt.y, 1.f);
				const Vec3D xn2 = unproj2.uvZToCamera(keypoint2.pt.x, keypoint2.pt.y, 1.f);

				const Vec3D ray1 = unproj1.Rwc * xn1;
				const Vec3D ray2 = unproj2.Rwc * xn2;
				const float cosParallaxRays = CosAngle(ray1, ray2);

				float cosParallaxStereo = cosParallaxRays + 1;
				float cosParallaxStereo1 = cosParallaxStereo;
				float cosParallaxStereo2 = cosParallaxStereo;

				if (stereo1)
					cosParallaxStereo1 = cosf(Parallax(keyframe1->camera.baseline, Z1));
				else if (stereo2)
					cosParallaxStereo2 = cosf(Parallax(keyframe2->camera.baseline, Z2));

				cosParallaxStereo = std::min(cosParallaxStereo1, cosParallaxStereo2);

				Point3D Xw;
				if (cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (stereo1 || stereo2 || cosParallaxRays < 0.9998))
				{
					// Linear Triangulation Method
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = xn1(0) * Tcw1.row(2) - Tcw1.row(0);
					A.row(1) = xn1(1) * Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = xn2(0) * Tcw2.row(2) - Tcw2.row(0);
					A.row(3) = xn2(1) * Tcw2.row(2) - Tcw2.row(1);

					cv::Mat w, u, vt;
					cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

					cv::Mat1f v = vt.row(3).t();

					if (v(3) == 0)
						continue;

					// Euclidean coordinates
					const double denom = 1. / v(3);
					Xw = denom * Point3D(v(0), v(1), v(2));

				}
				else if (stereo1 && cosParallaxStereo1 < cosParallaxStereo2)
				{
					Xw = unproj1.uvZToWorld(keypoint1.pt, Z1);
				}
				else if (stereo2 && cosParallaxStereo2 < cosParallaxStereo1)
				{
					Xw = unproj2.uvZToWorld(keypoint2.pt, Z2);
				}
				else
					continue; //No stereo and very low parallax

				// Check triangulation in front of cameras
				const Point3D Xc1 = proj1.WorldToCamera(Xw);
				const Point3D Xc2 = proj2.WorldToCamera(Xw);
				if (Xc1(2) <= 0 || Xc2(2) <= 0)
					continue;

				// Check reprojection error in first keyframe
				const float sigmaSq1 = keyframe1->pyramid.sigmaSq[keypoint1.octave];
				const Point2D pt1 = proj1.CameraToImage(Xc1);
				const Point2D diff1 = pt1 - keypoint1.pt;
				if (!stereo1)
				{
					if (NormSq(diff1.x, diff1.y) > 5.991 * sigmaSq1)
						continue;
				}
				else
				{
					const float d1 = proj1.DepthToDisparity(Xc1(2));
					const float diff1z = (pt1.x - d1) - ur1;
					if (NormSq(diff1.x, diff1.y, diff1z) > 7.8 * sigmaSq1)
						continue;
				}

				// Check reprojection error in second keyframe
				const float sigmaSq2 = keyframe2->pyramid.sigmaSq[keypoint2.octave];
				const Point2D pt2 = proj2.CameraToImage(Xc2);
				const Point2D diff2 = pt2 - keypoint2.pt;
				if (!stereo2)
				{
					if (NormSq(diff2.x, diff2.y) > 5.991 * sigmaSq2)
						continue;
				}
				else
				{
					const float d2 = proj2.DepthToDisparity(Xc2(2));
					const float diff2z = (pt2.x - d2) - ur2;
					if (NormSq(diff2.x, diff2.y, diff2z) > 7.8 * sigmaSq2)
						continue;
				}

				//Check scale consistency
				const Vec3D normal1 = Xw - Ow1;
				const float dist1 = static_cast<float>(cv::norm(normal1));

				const Vec3D normal2 = Xw - Ow2;
				const float dist2 = static_cast<float>(cv::norm(normal2));

				if (dist1 == 0 || dist2 == 0)
					continue;

				const float ratioDist = dist2 / dist1;
				const float scale1 = keyframe1->pyramid.scaleFactors[keypoint1.octave];
				const float scale2 = keyframe2->pyramid.scaleFactors[keypoint2.octave];
				const float ratioOctave = scale1 / scale2;

				if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
					continue;

				// Triangulation is succesfull
				MapPoint* mappoint = new MapPoint(Xw, keyframe1, map_);

				mappoint->AddObservation(keyframe1, idx1);
				mappoint->AddObservation(keyframe2, idx2);

				keyframe1->AddMapPoint(mappoint, idx1);
				keyframe2->AddMapPoint(mappoint, idx2);

				mappoint->ComputeDistinctiveDescriptors();
				mappoint->UpdateNormalAndDepth();

				map_->AddMapPoint(mappoint);
				recentAddedMapPoints_.push_back(mappoint);
			}
		}
	}

	void SearchInNeighbors(KeyFrame* currKeyFrame_)
	{
		// Retrieve neighbor keyframes
		const int nneighbors = monocular_ ? 20 : 10;
		std::vector<KeyFrame*> targetKFs;
		for (KeyFrame* neighborKF : currKeyFrame_->GetBestCovisibilityKeyFrames(nneighbors))
		{
			if (neighborKF->isBad() || neighborKF->fuseTargetForKF == currKeyFrame_->id)
				continue;

			targetKFs.push_back(neighborKF);
			neighborKF->fuseTargetForKF = currKeyFrame_->id;

			// Extend to some second neighbors
			for (KeyFrame* secondKF : neighborKF->GetBestCovisibilityKeyFrames(5))
			{
				if (secondKF->isBad() || secondKF->fuseTargetForKF == currKeyFrame_->id || secondKF->id == currKeyFrame_->id)
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
				if (!mappoint || mappoint->isBad() || mappoint->fuseCandidateForKF == currKeyFrame_->id)
					continue;

				mappoint->fuseCandidateForKF = currKeyFrame_->id;
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

	void KeyFrameCulling(KeyFrame* currKeyFrame_)
	{
		// Check redundant keyframes (only local keyframes)
		// A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
		// in at least other 3 keyframes (in the same or finer scale)
		// We only consider close stereo points
		const int minObservations = 3;
		for (KeyFrame* targetKF : currKeyFrame_->GetVectorCovisibleKeyFrames())
		{
			if (targetKF->id == 0)
				continue;

			const std::vector<MapPoint*> mappoints = targetKF->GetMapPointMatches();

			int nredundant = 0;
			int npoints = 0;

			for (size_t i1 = 0; i1 < mappoints.size(); i1++)
			{
				MapPoint* mappoint = mappoints[i1];
				if (!mappoint || mappoint->isBad())
					continue;

				if (!monocular_)
				{
					if (targetKF->depth[i1] > thDepth_ || targetKF->depth[i1] < 0)
						continue;
				}

				npoints++;
				if (mappoint->Observations() > minObservations)
				{
					const int targetScale = targetKF->keypointsUn[i1].octave;
					int nobservations = 0;
					for (const auto& observation : mappoint->GetObservations())
					{
						const KeyFrame* otherKF = observation.first;
						const size_t i2 = observation.second;
						if (otherKF == targetKF)
							continue;

						const int otherScale = otherKF->keypointsUn[i2].octave;

						if (otherScale <= targetScale + 1)
						{
							nobservations++;
							if (nobservations >= minObservations)
								break;
						}
					}
					if (nobservations >= minObservations)
					{
						nredundant++;
					}
				}
			}

			if (nredundant > 0.9 * npoints)
				targetKF->SetBadFlag();
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
	std::list<MapPoint*> recentAddedMapPoints_;

	bool abortBA_;
	bool stopped_;
	bool stopRequested_;
	bool notStop_;
	bool acceptKeyFrames_;

	float thDepth_;

	mutable std::mutex mutexNewKFs_;
	mutable std::mutex mutexReset_;
	mutable std::mutex mutexFinish_;
	mutable std::mutex mutexStop_;
	mutable std::mutex mutexAccept_;
};

LocalMapping::Pointer LocalMapping::Create(Map* map, bool monocular, float thDepth)
{
	return std::make_unique<LocalMappingImpl>(map, monocular, thDepth);
}

LocalMapping::~LocalMapping() {}

} //namespace ORB_SLAM
