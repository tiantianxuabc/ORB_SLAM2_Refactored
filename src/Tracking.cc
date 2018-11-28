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
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBmatcher.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "System.h"
#include "PnPsolver.h"
#include "Usleep.h"
#include "CameraParameters.h"
#include "Optimizer.h"

namespace ORB_SLAM2
{

TrackPoint::TrackPoint(const Frame& frame, bool lost)
	: referenceKF(frame.referenceKF), timestamp(frame.timestamp), lost(lost)
{
	Tcr = frame.pose * frame.referenceKF->GetPose().Inverse();
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
	LocalMap(Map* map) : map_(map) {}

	void Update(Frame& currFrame)
	{
		// This is for visualization
		map_->SetReferenceMapPoints(mappoints);

		// Update
		UpdateLocalKeyFrames(currFrame);
		UpdateLocalPoints(currFrame);
	}

	void UpdateLocalKeyFrames(Frame& currFrame)
	{
		// Each map point vote for the keyframes in which it has been observed
		std::map<KeyFrame*, int> keyframeCounter;
		for (int i = 0; i < currFrame.N; i++)
		{
			if (!currFrame.mappoints[i])
				continue;

			MapPoint* mappoint = currFrame.mappoints[i];
			if (!mappoint->isBad())
			{
				for (const auto& observations : mappoint->GetObservations())
					keyframeCounter[observations.first]++;
			}
			else
			{
				currFrame.mappoints[i] = nullptr;
			}
		}

		if (keyframeCounter.empty())
			return;

		int maxCount = 0;
		KeyFrame* maxKeyFrame = nullptr;

		keyframes.clear();
		keyframes.reserve(3 * keyframeCounter.size());

		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (const auto& v : keyframeCounter)
		{
			KeyFrame* keyframe = v.first;
			const int count = v.second;

			if (keyframe->isBad())
				continue;

			if (count > maxCount)
			{
				maxCount = count;
				maxKeyFrame = keyframe;
			}

			keyframes.push_back(keyframe);
			keyframe->trackReferenceForFrame = currFrame.id;
		}

		// Include also some not-already-included keyframes that are neighbors to already-included keyframes
		for (KeyFrame* keyframe : keyframes)
		{
			// Limit the number of keyframes
			if (keyframes.size() > 80)
				break;

			for (KeyFrame* neighborKF : keyframe->GetBestCovisibilityKeyFrames(10))
			{
				if (!neighborKF->isBad() && neighborKF->trackReferenceForFrame != currFrame.id)
				{
					keyframes.push_back(neighborKF);
					neighborKF->trackReferenceForFrame = currFrame.id;
					break;
				}
			}

			for (KeyFrame* childKF : keyframe->GetChildren())
			{
				if (!childKF->isBad() && childKF->trackReferenceForFrame != currFrame.id)
				{
					keyframes.push_back(childKF);
					childKF->trackReferenceForFrame = currFrame.id;
					break;
				}
			}

			KeyFrame* parentKF = keyframe->GetParent();
			if (parentKF)
			{
				if (parentKF->trackReferenceForFrame != currFrame.id)
				{
					keyframes.push_back(parentKF);
					parentKF->trackReferenceForFrame = currFrame.id;
					break;
				}
			}

		}

		if (maxKeyFrame)
		{
			referenceKF = maxKeyFrame;
			currFrame.referenceKF = referenceKF;
		}
	}

	void UpdateLocalPoints(Frame& currFrame)
	{
		mappoints.clear();
		for (KeyFrame* keyframe : keyframes)
		{
			for (MapPoint* mappoint : keyframe->GetMapPointMatches())
			{
				if (!mappoint || mappoint->trackReferenceForFrame == currFrame.id || mappoint->isBad())
					continue;

				mappoints.push_back(mappoint);
				mappoint->trackReferenceForFrame = currFrame.id;
			}
		}
	}

	KeyFrame* referenceKF;
	std::vector<KeyFrame*> keyframes;
	std::vector<MapPoint*> mappoints;
	Map* map_;
};

static int DiscardOutliers(Frame& currFrame)
{
	int ninliers = 0;
	for (int i = 0; i < currFrame.N; i++)
	{
		if (!currFrame.mappoints[i])
			continue;

		if (currFrame.outlier[i])
		{
			MapPoint* mappoint = currFrame.mappoints[i];

			currFrame.mappoints[i] = nullptr;
			currFrame.outlier[i] = false;

			mappoint->trackInView = false;
			mappoint->lastFrameSeen = currFrame.id;
		}
		else if (currFrame.mappoints[i]->Observations() > 0)
		{
			ninliers++;
		}
	}
	return ninliers;
}

static void UpdateLastFramePose(Frame& lastFrame, const TrackPoint& lastTrackPoint)
{
	// Update pose according to reference keyframe
	KeyFrame* referenceKF = lastFrame.referenceKF;
	lastFrame.SetPose(lastTrackPoint.Tcr * CameraPose(referenceKF->GetPose()));
}

bool TrackWithMotionModel(Frame& currFrame, Frame& lastFrame, const cv::Mat& velocity,
	int minInliers, int sensor, bool* fewMatches = nullptr)
{
	ORBmatcher matcher(0.9f, true);

	currFrame.SetPose(CameraPose(velocity) * lastFrame.pose);

	// Project points seen in previous frame
	const float threshold = sensor == System::STEREO ? 7.f : 15.f;
	const int minMatches = 20;
	int nmatches = 0;
	{
		std::fill(std::begin(currFrame.mappoints), std::end(currFrame.mappoints), nullptr);
		nmatches = matcher.SearchByProjection(currFrame, lastFrame, threshold, sensor == System::MONOCULAR);
	}
	if (nmatches < minMatches)
	{
		// If few matches, uses a wider window search
		std::fill(std::begin(currFrame.mappoints), std::end(currFrame.mappoints), nullptr);
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

	currFrame.mappoints = mappoints;
	currFrame.SetPose(lastFrame.pose);

	Optimizer::PoseOptimization(&currFrame);

	// Discard outliers
	const int ninliers = DiscardOutliers(currFrame);

	return ninliers >= minInliers;
}

class NewKeyFrameCondition
{
public:

	using Parameters = TrackerParameters;

	NewKeyFrameCondition(Map* map, const LocalMap& LocalMap, const Parameters& param, int sensor)
		: map_(map), localMap_(LocalMap), param_(param), sensor_(sensor) {}

	bool Satisfy(const Frame& currFrame, LocalMapping* localMapper, int matchesInliers,
		int lastRelocFrameId, int lastKeyFrameId) const
	{
		// If Local Mapping is freezed by a Loop Closure do not insert keyframes
		if (localMapper->isStopped() || localMapper->stopRequested())
			return false;

		const size_t nkeyframes = map_->KeyFramesInMap();

		// Do not insert keyframes if not enough frames have passed from last relocalisation
		if (currFrame.PassedFrom(lastRelocFrameId) < param_.maxFrames && nkeyframes > param_.maxFrames)
			return false;

		// Tracked MapPoints in the reference keyframe
		const int minObservations = nkeyframes <= 2 ? 2 : 3;
		const int refMatches = localMap_.referenceKF->TrackedMapPoints(minObservations);

		// Local Mapping accept keyframes?
		const bool acceptKeyFrames = localMapper->AcceptKeyFrames();

		// Check how many "close" points are being tracked and how many could be potentially created.
		enum { TRACKED = 0, NON_TRACKED = 1 };
		int count[2] = { 0, 0 };
		if (sensor_ != System::MONOCULAR)
		{
			for (int i = 0; i < currFrame.N; i++)
			{
				if (currFrame.depth[i] > 0 && currFrame.depth[i] < param_.thDepth)
				{
					const bool tracked = currFrame.mappoints[i] && !currFrame.outlier[i];
					const int idx = tracked ? TRACKED : NON_TRACKED;
					count[idx]++;
				}
			}
		}

		const bool needToInsertClose = (count[TRACKED] < 100) && (count[NON_TRACKED] > 70);

		// Thresholds
		const float refRatio = sensor_ == System::MONOCULAR ? 0.9f : (nkeyframes < 2 ? 0.4f : 0.75f);

		// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
		const bool c1a = currFrame.PassedFrom(lastKeyFrameId) >= param_.maxFrames;
		// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
		const bool c1b = currFrame.PassedFrom(lastKeyFrameId) >= param_.minFrames && acceptKeyFrames;
		//Condition 1c: tracking is weak
		const bool c1c = sensor_ != System::MONOCULAR && (matchesInliers < refMatches * 0.25 || needToInsertClose);
		// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
		const bool c2 = ((matchesInliers < refMatches * refRatio || needToInsertClose) && matchesInliers > 15);

		if ((c1a || c1b || c1c) && c2)
		{
			// If the mapping accepts keyframes, insert keyframe.
			// Otherwise send a signal to interrupt BA
			if (acceptKeyFrames)
				return true;

			localMapper->InterruptBA();

			if (sensor_ != System::MONOCULAR)
				return localMapper->KeyframesInQueue() < 3;
		}

		return false;
	}

private:
	Map* map_;
	const LocalMap& localMap_;
	Parameters param_;
	int sensor_;
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

// Check if a MapPoint is in the frustum of the camera
// and fill variables of the MapPoint to be used by the tracking
static bool IsInFrustum(const Frame& frame, MapPoint* mappoint, float minViewingCos)
{
	mappoint->trackInView = false;

	const CameraParams& camera = frame.camera;
	const CameraPose& pose = frame.pose;
	const auto Ow = frame.GetCameraCenter();

	// 3D in absolute coordinates
	const Point3D Xw = mappoint->GetWorldPos();

	// 3D in camera coordinates
	const auto Rcw = pose.R();
	const auto tcw = pose.t();
	const Point3D Xc = Rcw * Xw + tcw;
	const float PcX = Xc(0);
	const float PcY = Xc(1);
	const float PcZ = Xc(2);

	// Check positive depth
	if (PcZ < 0.f)
		return false;

	// Project in image and check it is not outside
	const float invZ = 1.f / PcZ;
	const float u = camera.fx * PcX * invZ + camera.cx;
	const float v = camera.fy * PcY * invZ + camera.cy;

	if (!frame.imageBounds.Contains(u, v))
		return false;

	// Check distance is in the scale invariance region of the MapPoint
	const float maxDistance = mappoint->GetMaxDistanceInvariance();
	const float minDistance = mappoint->GetMinDistanceInvariance();
	const Vec3D PO = Xw - Ow;
	const float dist = static_cast<float>(cv::norm(PO));

	if (dist < minDistance || dist > maxDistance)
		return false;

	// Check viewing angle
	const Vec3D Pn = mappoint->GetNormal();

	const float viewCos = static_cast<float>(PO.dot(Pn) / dist);

	if (viewCos < minViewingCos)
		return false;

	// Predict scale in the image
	const int scale = mappoint->PredictScale(dist, (Frame*)&frame);

	// Data used by the tracking
	mappoint->trackInView = true;
	mappoint->trackProjX = u;
	mappoint->trackProjXR = u - camera.bf * invZ;
	mappoint->trackProjY = v;
	mappoint->trackScaleLevel = scale;
	mappoint->trackViewCos = viewCos;

	return true;
}

static void SearchLocalPoints(const LocalMap& localMap, Frame& currFrame, float th)
{
	// Do not search map points already matched
	for (MapPoint* mappoint : currFrame.mappoints)
	{
		if (!mappoint)
			continue;

		if (mappoint->isBad())
		{
			mappoint = nullptr;
		}
		else
		{
			mappoint->IncreaseVisible();
			mappoint->lastFrameSeen = currFrame.id;
			mappoint->trackInView = false;
		}
	}

	int nToMatch = 0;

	// Project points in frame and check its visibility
	for (MapPoint* mappoint : localMap.mappoints)
	{
		if (mappoint->lastFrameSeen == currFrame.id || mappoint->isBad())
			continue;

		// Project (this fills MapPoint variables for matching)
		if (IsInFrustum(currFrame, mappoint, 0.5f))
		{
			mappoint->IncreaseVisible();
			nToMatch++;
		}
	}

	if (nToMatch > 0)
	{
		ORBmatcher matcher(0.8f);
		matcher.SearchByProjection(currFrame, localMap.mappoints, th);
	}
}

static int TrackLocalMap(LocalMap& localMap, Frame& currFrame, float th, bool localization, bool stereo)
{
	// We have an estimation of the camera pose and some map points tracked in the frame.
	// We retrieve the local map and try to find matches to points in the local map.

	localMap.Update(currFrame);

	SearchLocalPoints(localMap, currFrame, th);

	// Optimize Pose
	Optimizer::PoseOptimization(&currFrame);
	int ninliers = 0;

	// Update MapPoints Statistics
	for (int i = 0; i < currFrame.N; i++)
	{
		if (!currFrame.mappoints[i])
			continue;

		if (!currFrame.outlier[i])
		{
			currFrame.mappoints[i]->IncreaseFound();
			if (localization || (!localization && currFrame.mappoints[i]->Observations() > 0))
				ninliers++;
		}
		else if (stereo)
		{
			currFrame.mappoints[i] = nullptr;
		}
	}

	return ninliers;
}

void CreateMapPoints(Frame& currFrame, KeyFrame* keyframe, Map* map, float thDepth)
{
	// We sort points by the measured depth by the stereo/RGBD sensor.
	// We create all those MapPoints whose depth < param_.thDepth.
	// If there are less than 100 close points we create the 100 closest.
	std::vector<std::pair<float, int> > depthIndices;
	depthIndices.reserve(currFrame.N);
	for (int i = 0; i < currFrame.N; i++)
	{
		const float Z = currFrame.depth[i];
		if (Z > 0)
			depthIndices.push_back(std::make_pair(Z, i));
	}

	if (depthIndices.empty())
		return;

	std::sort(std::begin(depthIndices), std::end(depthIndices));

	int npoints = 0;
	for (const auto& v : depthIndices)
	{
		const float Z = v.first;
		const int i = v.second;

		bool create = false;

		MapPoint* mappoint = currFrame.mappoints[i];
		if (!mappoint)
		{
			create = true;
		}
		else if (mappoint->Observations() < 1)
		{
			create = true;
			currFrame.mappoints[i] = nullptr;
		}

		if (create)
		{
			const Point3D Xw = currFrame.UnprojectStereo(i);

			MapPoint* newpoint = new MapPoint(Xw, keyframe, map);
			newpoint->AddObservation(keyframe, i);
			newpoint->ComputeDistinctiveDescriptors();
			newpoint->UpdateNormalAndDepth();

			keyframe->AddMapPoint(newpoint, i);
			map->AddMapPoint(newpoint);
			currFrame.mappoints[i] = newpoint;
		}

		npoints++;

		if (Z > thDepth && npoints > 100)
			break;
	}
}

static void CreateMapPointsVO(Frame& lastFrame, list<MapPoint*>& tempPoints, Map* map, float thDepth)
{
	// Create "visual odometry" MapPoints
	// We sort points according to their measured depth by the stereo/RGB-D sensor
	std::vector<std::pair<float, int>> depthIndices;
	depthIndices.reserve(lastFrame.N);
	for (int i = 0; i < lastFrame.N; i++)
	{
		const float Z = lastFrame.depth[i];
		if (Z > 0)
			depthIndices.push_back(std::make_pair(Z, i));
	}

	if (depthIndices.empty())
		return;

	std::sort(std::begin(depthIndices), std::end(depthIndices));

	// We insert all close points (depth<param_.thDepth)
	// If less than 100 close points, we insert the 100 closest ones.
	int npoints = 0;
	for (const auto& v : depthIndices)
	{
		const float Z = v.first;
		const int i = v.second;

		MapPoint* mappoint = lastFrame.mappoints[i];
		if (!mappoint || mappoint->Observations() < 1)
		{
			const Point3D Xw = lastFrame.UnprojectStereo(i);
			MapPoint* newpoint = new MapPoint(Xw, map, &lastFrame, i);

			lastFrame.mappoints[i] = newpoint;
			tempPoints.push_back(newpoint);
		}

		npoints++;

		if (Z > thDepth && npoints > 100)
			break;
	}
}

class Relocalizer
{
public:

	Relocalizer(KeyFrameDatabase* keyFrameDB) : keyFrameDB_(keyFrameDB), lastRelocFrameId_(0) {}

	bool Relocalize(Frame& currFrame)
	{
		// Compute Bag of Words Vector
		currFrame.ComputeBoW();

		// Relocalization is performed when tracking is lost
		// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
		std::vector<KeyFrame*> candidateKFs = keyFrameDB_->DetectRelocalizationCandidates(&currFrame);

		if (candidateKFs.empty())
			return false;

		const int nkeyframes = static_cast<int>(candidateKFs.size());

		// We perform first an ORB matching with each candidate
		// If enough matches are found we setup a PnP solver
		ORBmatcher matcher(0.75f, true);

		std::vector<PnPsolver*> PnPsolvers;
		PnPsolvers.resize(nkeyframes);

		std::vector<std::vector<MapPoint*>> vmatches;
		vmatches.resize(nkeyframes);

		std::vector<bool> discarded;
		discarded.resize(nkeyframes);

		int ncandidates = 0;

		for (int i = 0; i < nkeyframes; i++)
		{
			KeyFrame* keyframe = candidateKFs[i];
			if (keyframe->isBad())
			{
				discarded[i] = true;
			}
			else
			{
				const int nmatches = matcher.SearchByBoW(keyframe, currFrame, vmatches[i]);
				if (nmatches < 15)
				{
					discarded[i] = true;
					continue;
				}
				else
				{
					PnPsolver* solver = new PnPsolver(currFrame, vmatches[i]);
					solver->SetRansacParameters(0.99, 10, 300, 4, 0.5f, 5.991f);
					PnPsolvers[i] = solver;
					ncandidates++;
				}
			}
		}

		// Alternatively perform some iterations of P4P RANSAC
		// Until we found a camera pose supported by enough inliers
		bool found = false;
		ORBmatcher matcher2(0.9f, true);

		while (ncandidates > 0 && !found)
		{
			for (int i = 0; i < nkeyframes; i++)
			{
				if (discarded[i])
					continue;

				// Perform 5 Ransac Iterations
				std::vector<bool> isInlier;
				int nInliers;
				bool terminate;

				PnPsolver* solver = PnPsolvers[i];
				const cv::Mat Tcw = solver->iterate(5, terminate, isInlier, nInliers);

				// If Ransac reachs max. iterations discard keyframe
				if (terminate)
				{
					discarded[i] = true;
					ncandidates--;
				}

				// If a Camera Pose is computed, optimize
				if (!Tcw.empty())
				{
					currFrame.SetPose(CameraPose(Tcw));

					std::set<MapPoint*> foundPoints;

					const int np = static_cast<int>(isInlier.size());

					for (int j = 0; j < np; j++)
					{
						if (isInlier[j])
						{
							currFrame.mappoints[j] = vmatches[i][j];
							foundPoints.insert(vmatches[i][j]);
						}
						else
							currFrame.mappoints[j] = nullptr;
					}

					int ngood = Optimizer::PoseOptimization(&currFrame);

					if (ngood < 10)
						continue;

					for (int io = 0; io < currFrame.N; io++)
						if (currFrame.outlier[io])
							currFrame.mappoints[io] = nullptr;

					// If few inliers, search by projection in a coarse window and optimize again
					if (ngood < 50)
					{
						int nadditional = matcher2.SearchByProjection(currFrame, candidateKFs[i], foundPoints, 10, 100);

						if (nadditional + ngood >= 50)
						{
							ngood = Optimizer::PoseOptimization(&currFrame);

							// If many inliers but still not enough, search by projection again in a narrower window
							// the camera has been already optimized with many points
							if (ngood > 30 && ngood < 50)
							{
								foundPoints.clear();
								for (int ip = 0; ip < currFrame.N; ip++)
									if (currFrame.mappoints[ip])
										foundPoints.insert(currFrame.mappoints[ip]);
								nadditional = matcher2.SearchByProjection(currFrame, candidateKFs[i], foundPoints, 3, 64);

								// Final optimization
								if (ngood + nadditional >= 50)
								{
									ngood = Optimizer::PoseOptimization(&currFrame);

									for (int io = 0; io < currFrame.N; io++)
										if (currFrame.outlier[io])
											currFrame.mappoints[io] = nullptr;
								}
							}
						}
					}


					// If the pose is supported by enough inliers stop ransacs and continue
					if (ngood >= 50)
					{
						found = true;
						break;
					}
				}
			}
		}

		if (!found)
		{
			return false;
		}
		else
		{
			lastRelocFrameId_ = currFrame.id;
			return true;
		}
	}

	frameid_t GetLastRelocFrameId() const
	{
		return lastRelocFrameId_;
	}

private:

	KeyFrameDatabase* keyFrameDB_;
	frameid_t lastRelocFrameId_;
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

static ORBextractor::Parameters ReadExtractorParams(const cv::FileStorage& fs)
{
	ORBextractor::Parameters param;
	param.nfeatures = fs["ORBextractor.nFeatures"];
	param.scaleFactor = fs["ORBextractor.scaleFactor"];
	param.nlevels = fs["ORBextractor.nLevels"];
	param.iniThFAST = fs["ORBextractor.iniThFAST"];
	param.minThFAST = fs["ORBextractor.minThFAST"];
	return param;
}

static float ReadDepthFactor(const cv::FileStorage& fs)
{
	const float factor = fs["DepthMapFactor"];
	return fabs(factor) < 1e-5 ? 1 : 1.f / factor;
}

static void PrintSettings(const CameraParams& camera, const cv::Mat1f& distCoeffs,
	float fps, bool rgb, const ORBextractor::Parameters& param, float thDepth, int sensor)
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
	cout << "- Initial Fast Threshold: " << param.iniThFAST << endl;
	cout << "- Minimum Fast Threshold: " << param.minThFAST << endl;

	if (sensor == System::STEREO || sensor == System::RGBD)
		cout << endl << "Depth Threshold (Close/Far Points): " << thDepth << endl;
}

class InitialPoseEstimator
{

public:

	InitialPoseEstimator(Map* map, LocalMap& localMap, Relocalizer& relocalizer, const Trajectory& trajectory,
		int sensor, float thDepth)
		: sensor_(sensor), fewMatches_(false), localMap_(localMap), map_(map),
		relocalizer_(relocalizer), trajectory_(trajectory), thDepth_(thDepth)
	{
	}

	bool Estimate(Frame& currFrame, Frame& lastFrame, const cv::Mat& velocity)
	{
		// Local Mapping is activated. This is the normal behaviour, unless
		// you explicitly activate the "only tracking" mode.

		const int minInliers = 10;

		// Local Mapping might have changed some MapPoints tracked in last frame
		for (int i = 0; i < lastFrame.N; i++)
		{
			MapPoint* mappoint = lastFrame.mappoints[i];
			MapPoint* replaced = mappoint ? mappoint->GetReplaced() : nullptr;
			if (replaced)
				lastFrame.mappoints[i] = replaced;
		}

		bool success = false;
		const bool withMotionModel = !velocity.empty() && currFrame.PassedFrom(relocalizer_.GetLastRelocFrameId()) >= 2;
		if (withMotionModel)
		{
			UpdateLastFramePose(lastFrame, trajectory_.back());
			success = TrackWithMotionModel(currFrame, lastFrame, velocity, minInliers, sensor_);
		}
		if (!withMotionModel || (withMotionModel && !success))
		{
			success = TrackReferenceKeyFrame(currFrame, localMap_.referenceKF, lastFrame);
		}

		return success;
	}

	bool EstimateLocalization(Frame& currFrame, Frame& lastFrame, const cv::Mat& velocity, int lastKeyFrameId)
	{
		// Localization Mode: Local Mapping is deactivated

		const int minInliers = 21;
		const bool createPoints = sensor_ != System::MONOCULAR && lastFrame.id != lastKeyFrameId;
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
				success = TrackReferenceKeyFrame(currFrame, localMap_.referenceKF, lastFrame);
			}
		}
		else
		{
			// In last frame we tracked mainly "visual odometry" points.

			// We compute two camera poses, one from motion model and one doing relocalization.
			// If relocalization is sucessfull we choose that solution, otherwise we retain
			// the "visual odometry" solution.

			// compute camera pose from motion model
			bool successMM = false;
			std::vector<MapPoint*> mappointsMM;
			std::vector<bool> outlierMM;
			CameraPose poseMM;
			if (!velocity.empty())
			{
				UpdateLastFramePose(lastFrame, trajectory_.back());
				if (createPoints)
					CreateMapPointsVO(lastFrame, tempPoints_, map_, thDepth_);

				successMM = TrackWithMotionModel(currFrame, lastFrame, velocity, minInliers, sensor_, &fewMatches_);
				mappointsMM = currFrame.mappoints;
				outlierMM = currFrame.outlier;
				poseMM = currFrame.pose;
			}

			// compute camera pose from relocalization
			const bool successReloc = relocalizer_.Relocalize(currFrame);

			if (successReloc)
			{
				// If relocalization is sucessfull we choose that solution
				fewMatches_ = false;
			}
			else if (successMM)
			{
				// otherwise we retain the "visual odometry" solution.
				currFrame.SetPose(poseMM);
				currFrame.mappoints = mappointsMM;
				currFrame.outlier = outlierMM;

				if (fewMatches_)
				{
					for (int i = 0; i < currFrame.N; i++)
						if (currFrame.mappoints[i] && !currFrame.outlier[i])
							currFrame.mappoints[i]->IncreaseFound();
				}
			}

			success = successReloc || successMM;
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

	//Local Map
	LocalMap& localMap_;

	//Map
	Map* map_;

	//Last Frame, KeyFrame and Relocalisation Info
	Relocalizer& relocalizer_;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	const Trajectory& trajectory_;

	std::list<MapPoint*> tempPoints_;

	float thDepth_;
};

class TrackerCore
{

public:

	using Parameters = TrackerParameters;

	TrackerCore(Tracking* tracking, System* system, Map* map, KeyFrameDatabase* keyFrameDB,
		int sensor, const Parameters& param)
		: state_(STATE_NO_IMAGES), sensor_(sensor), localization_(false), keyFrameDB_(keyFrameDB),
		initializer_(nullptr), tracking_(tracking), system_(system), map_(map), localMap_(map),
		newKeyFrameCondition_(map, localMap_, param, sensor), relocalizer_(keyFrameDB),
		initPose_(map, localMap_, relocalizer_, trajectory_, sensor, param.thDepth), param_(param)
	{
	}

	// Map initialization for stereo and RGB-D
	void StereoInitialization(Frame& currFrame)
	{
		if (currFrame.N <= 500)
			return;

		// Set Frame pose to the origin
		currFrame.SetPose(CameraPose::Origin());

		// Create KeyFrame
		KeyFrame* keyframe = new KeyFrame(currFrame, map_, keyFrameDB_);

		// Insert KeyFrame in the map
		map_->AddKeyFrame(keyframe);

		// Create MapPoints and asscoiate to KeyFrame
		for (int i = 0; i < currFrame.N; i++)
		{
			const float Z = currFrame.depth[i];
			if (Z <= 0.f)
				continue;

			const Point3D Xw = currFrame.UnprojectStereo(i);
			MapPoint* mappoint = new MapPoint(Xw, keyframe, map_);
			mappoint->AddObservation(keyframe, i);
			mappoint->ComputeDistinctiveDescriptors();
			mappoint->UpdateNormalAndDepth();

			keyframe->AddMapPoint(mappoint, i);
			map_->AddMapPoint(mappoint);

			currFrame.mappoints[i] = mappoint;
		}

		cout << "New map created with " << map_->MapPointsInMap() << " points" << endl;

		localMapper_->InsertKeyFrame(keyframe);

		lastFrame_ = Frame(currFrame);
		lastKeyFrame_ = keyframe;
		CV_Assert(lastKeyFrame_->frameId == currFrame.id);

		localMap_.keyframes.push_back(keyframe);
		localMap_.mappoints = map_->GetAllMapPoints();
		localMap_.referenceKF = keyframe;
		currFrame.referenceKF = keyframe;

		map_->SetReferenceMapPoints(localMap_.mappoints);

		map_->keyFrameOrigins.push_back(keyframe);

		if (viewer_)
			viewer_->SetCurrentCameraPose(currFrame.pose);

		state_ = STATE_OK;
	}

	// Map initialization for monocular
	void MonocularInitialization(Frame& currFrame)
	{
		if (!initializer_)
		{
			// Set Reference Frame
			if (currFrame.keypointsL.size() > 100)
			{
				initFrame_ = Frame(currFrame);
				lastFrame_ = Frame(currFrame);
				prevMatched_.resize(currFrame.keypointsUn.size());
				for (size_t i = 0; i < currFrame.keypointsUn.size(); i++)
					prevMatched_[i] = currFrame.keypointsUn[i].pt;

				if (initializer_)
					delete initializer_;

				initializer_ = new Initializer(currFrame, 1.0, 200);

				fill(iniMatches_.begin(), iniMatches_.end(), -1);

				return;
			}
		}
		else
		{
			// Try to initialize
			if ((int)currFrame.keypointsL.size() <= 100)
			{
				delete initializer_;
				initializer_ = static_cast<Initializer*>(NULL);
				fill(iniMatches_.begin(), iniMatches_.end(), -1);
				return;
			}

			// Find correspondences
			ORBmatcher matcher(0.9f, true);
			int nmatches = matcher.SearchForInitialization(initFrame_, currFrame, prevMatched_, iniMatches_, 100);

			// Check if there are enough correspondences
			if (nmatches < 100)
			{
				delete initializer_;
				initializer_ = static_cast<Initializer*>(NULL);
				return;
			}

			cv::Mat Rcw; // Current Camera Rotation
			cv::Mat tcw; // Current Camera Translation
			vector<bool> triangulated; // Triangulated Correspondences (mvIniMatches)

			if (initializer_->Initialize(currFrame, iniMatches_, Rcw, tcw, mvIniP3D, triangulated))
			{
				for (size_t i = 0, iend = iniMatches_.size(); i < iend; i++)
				{
					if (iniMatches_[i] >= 0 && !triangulated[i])
					{
						iniMatches_[i] = -1;
						nmatches--;
					}
				}

				// Set Frame Poses
				initFrame_.SetPose(CameraPose::Origin());
				cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
				Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
				tcw.copyTo(Tcw.rowRange(0, 3).col(3));
				currFrame.SetPose(cv::Mat1f(Tcw));

				CreateInitialMapMonocular(currFrame);
			}
		}
	}

	void CreateInitialMapMonocular(Frame& currFrame)
	{
		// Create KeyFrames
		KeyFrame* pKFini = new KeyFrame(initFrame_, map_, keyFrameDB_);
		KeyFrame* pKFcur = new KeyFrame(currFrame, map_, keyFrameDB_);

		pKFini->ComputeBoW();
		pKFcur->ComputeBoW();

		// Insert KFs in the map
		map_->AddKeyFrame(pKFini);
		map_->AddKeyFrame(pKFcur);

		// Create MapPoints and asscoiate to keyframes
		for (size_t i = 0; i < iniMatches_.size(); i++)
		{
			if (iniMatches_[i] < 0)
				continue;

			//Create MapPoint.
			cv::Mat worldPos(mvIniP3D[i]);

			MapPoint* pMP = new MapPoint(worldPos, pKFcur, map_);

			pKFini->AddMapPoint(pMP, i);
			pKFcur->AddMapPoint(pMP, iniMatches_[i]);

			pMP->AddObservation(pKFini, i);
			pMP->AddObservation(pKFcur, iniMatches_[i]);

			pMP->ComputeDistinctiveDescriptors();
			pMP->UpdateNormalAndDepth();

			//Fill Current Frame structure
			currFrame.mappoints[iniMatches_[i]] = pMP;
			currFrame.outlier[iniMatches_[i]] = false;

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
		pKFcur->SetPose(CameraPose(Tc2w));

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

		currFrame.SetPose(cv::Mat1f(pKFcur->GetPose()));
		lastKeyFrame_ = pKFcur;
		CV_Assert(lastKeyFrame_->frameId == currFrame.id);

		localMap_.keyframes.push_back(pKFcur);
		localMap_.keyframes.push_back(pKFini);
		localMap_.mappoints = map_->GetAllMapPoints();
		localMap_.referenceKF = pKFcur;
		currFrame.referenceKF = pKFcur;

		lastFrame_ = Frame(currFrame);

		map_->SetReferenceMapPoints(localMap_.mappoints);

		if (viewer_)
			viewer_->SetCurrentCameraPose(pKFcur->GetPose());

		map_->keyFrameOrigins.push_back(pKFini);

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
		unique_lock<mutex> lock(map_->mutexMapUpdate);

		// Initialize Tracker if not initialized.
		if (state_ == STATE_NOT_INITIALIZED)
		{
			Initialization(currFrame, sensor_);

			if (viewer_)
				viewer_->UpdateFrame(tracking_);

			if (state_ == STATE_OK)
				trajectory_.push_back(TrackPoint(currFrame, false));

			return;
		}

		// System is initialized. Track Frame.
		bool success = false;

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		if (state_ != STATE_OK)
		{
			success = relocalizer_.Relocalize(currFrame);
		}
		else if (localization_)
		{
			success = initPose_.EstimateLocalization(currFrame, lastFrame_, velocity_, lastKeyFrame_->frameId);
		}
		else
		{
			success = initPose_.Estimate(currFrame, lastFrame_, velocity_);
		}

		currFrame.referenceKF = localMap_.referenceKF;

		// If we have an initial estimation of the camera pose and matching. Track the local map.
		// [In Localization Mode]
		// mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
		// a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
		// the camera we will use the local map again.
		if (success && (!localization_ || (localization_ && !initPose_.FewMatches())))
		{
			const int lastRelocFrameId = relocalizer_.GetLastRelocFrameId();
			// If the camera has been relocalised recently, perform a coarser search
			const bool relocalizedRecently = currFrame.PassedFrom(lastRelocFrameId) < 2;
			const float th = relocalizedRecently ? 5.f : (sensor_ == System::RGBD ? 3.f : 1.f);

			matchesInliers_ = TrackLocalMap(localMap_, currFrame, th, localization_, sensor_ == System::STEREO);

			// Decide if the tracking was succesful
			// More restrictive if there was a relocalization recently
			const int minInliers = currFrame.PassedFrom(lastRelocFrameId) < param_.maxFrames ? 50 : 30;
			success = matchesInliers_ >= minInliers;
		}

		state_ = success ? STATE_OK : STATE_LOST;

		// Update drawer
		if (viewer_)
			viewer_->UpdateFrame(tracking_);

		// If tracking were good, check if we insert a keyframe
		if (success)
		{
			// Update motion model
			velocity_ = !lastFrame_.pose.Empty() ? currFrame.pose * lastFrame_.pose.Inverse() : cv::Mat();
			
			if (viewer_)
				viewer_->SetCurrentCameraPose(currFrame.pose);

			// Clean VO matches
			for (int i = 0; i < currFrame.N; i++)
			{
				MapPoint* mappoint = currFrame.mappoints[i];
				if (mappoint && mappoint->Observations() < 1)
				{
					currFrame.outlier[i] = false;
					currFrame.mappoints[i] = nullptr;
				}
			}

			// Delete temporal MapPoints
			initPose_.DeleteTemporalMapPoints();

			// Check if we need to insert a new keyframe
			if (!localization_ && newKeyFrameCondition_.Satisfy(currFrame, localMapper_, matchesInliers_,
				relocalizer_.GetLastRelocFrameId(), lastKeyFrame_->frameId))
			{
				if (localMapper_->SetNotStop(true))
				{
					KeyFrame* keyframe = new KeyFrame(currFrame, map_, keyFrameDB_);
					localMap_.referenceKF = keyframe;
					currFrame.referenceKF = keyframe;

					if (sensor_ != System::MONOCULAR)
						CreateMapPoints(currFrame, keyframe, map_, param_.thDepth);

					localMapper_->InsertKeyFrame(keyframe);
					localMapper_->SetNotStop(false);
					lastKeyFrame_ = keyframe;
					CV_Assert(lastKeyFrame_->frameId == currFrame.id);
				}
			}

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame.
			for (int i = 0; i < currFrame.N; i++)
			{
				if (currFrame.mappoints[i] && currFrame.outlier[i])
					currFrame.mappoints[i] = nullptr;
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

		CV_Assert(currFrame.referenceKF);

		lastFrame_ = Frame(currFrame);

		// Store frame pose information to retrieve the complete camera trajectory afterwards.
		CV_Assert(currFrame.referenceKF == localMap_.referenceKF);
		const bool lost = state_ == STATE_LOST;
		if (!currFrame.pose.Empty())
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
		loopClosing_ = pLoopClosing;
	}

	void SetViewer(Viewer* viewer)
	{
		viewer_ = viewer;
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
		return initFrame_;
	}

	const std::vector<int>& GetIniMatches() const
	{
		return iniMatches_;
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
	std::vector<int> iniLastMatches_;
	std::vector<int> iniMatches_;
	std::vector<cv::Point2f> prevMatched_;
	std::vector<cv::Point3f> mvIniP3D;
	Frame initFrame_;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	Trajectory trajectory_;

	// True if local mapping is deactivated and we are performing only localization
	bool localization_;

	//Other Thread Pointers
	LocalMapping* localMapper_;
	LoopClosing* loopClosing_;

	//BoW
	KeyFrameDatabase* keyFrameDB_;

	// Initalization (only for monocular)
	Initializer* initializer_;

	//Local Map
	LocalMap localMap_;

	// System
	System* system_;

	//Drawers
	Viewer* viewer_;

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

	InitialPoseEstimator initPose_;

	NewKeyFrameCondition newKeyFrameCondition_;
};

class TrackingImpl : public Tracking
{

public:

	TrackingImpl(System* system, ORBVocabulary* voc, Map* map, KeyFrameDatabase* keyframeDB,
		const string& settingsFile, int sensor)
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
		ORBextractor::Parameters extractorParams = ReadExtractorParams(settings);

		// Load depth threshold
		const float thDepth = settings["ThDepth"];
		thDepth_ = camera_.baseline * thDepth;

		// Load depth factor
		depthFactor_ = sensor == System::RGBD ? ReadDepthFactor(settings) : 1.f;

		// Print settings
		PrintSettings(camera_, distCoeffs_, fps, RGB_, extractorParams, thDepth_, sensor);

		// Initialize ORB extractors
		extractorL_ = std::make_unique<ORBextractor>(extractorParams);
		extractorR_ = std::make_unique<ORBextractor>(extractorParams);

		if (sensor == System::MONOCULAR)
		{
			extractorParams.nfeatures *= 2;
			extractorIni_ = std::make_unique<ORBextractor>(extractorParams);
		}

		// Initialize tracker core
		tracker_ = std::make_unique<TrackerCore>(this, system, map, keyframeDB, sensor,
			TrackerCore::Parameters(minFrames, maxFrames, thDepth_));
	}

	// Preprocess the input and call Track(). Extract features and performs stereo matching.
	cv::Mat GrabImageStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) override
	{
		ConvertToGray(imageL, imageL_, RGB_);
		ConvertToGray(imageR, imageR_, RGB_);

		currFrame_ = Frame(imageL_, imageR_, timestamp, extractorL_.get(), extractorR_.get(), voc_,
			camera_, distCoeffs_, thDepth_);

		tracker_->Update(currFrame_);

		return currFrame_.pose;
	}

	cv::Mat GrabImageRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) override
	{
		ConvertToGray(image, imageL_, RGB_);

		depth.convertTo(depth_, CV_32F, depthFactor_);

		currFrame_ = Frame(imageL_, depth_, timestamp, extractorL_.get(), voc_, camera_, distCoeffs_, thDepth_);

		tracker_->Update(currFrame_);

		return currFrame_.pose;
	}

	cv::Mat GrabImageMonocular(const cv::Mat& image, double timestamp) override
	{
		ConvertToGray(image, imageL_, RGB_);

		const int state = tracker_->GetState();
		const bool init = state == STATE_NOT_INITIALIZED || state == STATE_NO_IMAGES;

		ORBextractor* pORBextractor = init ? extractorIni_.get() : extractorL_.get();

		currFrame_ = Frame(imageL_, timestamp, pORBextractor, voc_, camera_, distCoeffs_, thDepth_);

		tracker_->Update(currFrame_);

		return currFrame_.pose;
	}

	void SetLocalMapper(const std::shared_ptr<LocalMapping>& localMapper) override
	{
		localMapper_ = localMapper;
		tracker_->SetLocalMapper(localMapper.get());
	}

	void SetLoopClosing(const std::shared_ptr<LoopClosing>& loopClosing) override
	{
		loopClosing_ = loopClosing;
		tracker_->SetLoopClosing(loopClosing.get());
	}

	void SetViewer(Viewer* viewer) override
	{
		viewer_ = viewer;
		tracker_->SetViewer(viewer);
	}

	// Load new settings
	// The focal lenght should be similar or scale prediction will fail when projecting points
	// TODO: Modify MapPoint::PredictScale to take into account focal lenght
	void ChangeCalibration(const string& settingsFile) override
	{
		cv::FileStorage settings(settingsFile, cv::FileStorage::READ);
		camera_ = ReadCameraParams(settings);
		distCoeffs_ = ReadDistCoeffs(settings);
		Frame::initialComputation = true;
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
		map_->Clear();

		KeyFrame::nextId = 0;
		Frame::nextId = 0;

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
	std::shared_ptr<LocalMapping> localMapper_;
	std::shared_ptr<LoopClosing> loopClosing_;

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

std::shared_ptr<Tracking> Tracking::Create(System* system, ORBVocabulary* voc, Map* map,
	KeyFrameDatabase* keyframeDB, const string& settingsFile, int sensor)
{
	return std::make_shared<TrackingImpl>(system, voc, map, keyframeDB, settingsFile, sensor);
}

} //namespace ORB_SLAM
