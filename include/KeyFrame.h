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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <mutex>

#include <Thirdparty/DBoW2/DBoW2/BowVector.h>
#include <Thirdparty/DBoW2/DBoW2/FeatureVector.h>

#include "FrameId.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "CameraParameters.h"
#include "CameraPose.h"

namespace ORB_SLAM2
{

class Map;
class MapPoint;
class KeyFrameDatabase;

class KeyFrame
{
public:

	KeyFrame(const Frame& frame, Map* map, KeyFrameDatabase* keyframeDB);

	// Pose functions
	void SetPose(const CameraPose& pose);
	CameraPose GetPose() const;
	Point3D GetCameraCenter() const;

	// Bag of Words Representation
	void ComputeBoW();

	// Covisibility graph functions
	void AddConnection(KeyFrame* keyframe, int weight);
	void EraseConnection(KeyFrame* keyframe);
	void UpdateConnections();
	void UpdateBestCovisibles();
	std::set<KeyFrame *> GetConnectedKeyFrames() const;
	std::vector<KeyFrame* > GetVectorCovisibleKeyFrames() const;
	std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(int N) const;
	std::vector<KeyFrame*> GetCovisiblesByWeight(int w) const;
	int GetWeight(KeyFrame* keyframe) const;

	// Spanning tree functions
	void AddChild(KeyFrame* keyframe);
	void EraseChild(KeyFrame* keyframe);
	void ChangeParent(KeyFrame* keyframe);
	std::set<KeyFrame*> GetChildren() const;
	KeyFrame* GetParent() const;
	bool HasChild(KeyFrame* keyframe) const;

	// Loop Edges
	void AddLoopEdge(KeyFrame* keyframe);
	std::set<KeyFrame*> GetLoopEdges() const;

	// MapPoint observation functions
	void AddMapPoint(MapPoint* mappoint, size_t idx);
	void EraseMapPointMatch(size_t idx);
	void EraseMapPointMatch(MapPoint* mappoint);
	void ReplaceMapPointMatch(size_t idx, MapPoint* mappoint);
	std::set<MapPoint*> GetMapPoints() const;
	std::vector<MapPoint*> GetMapPointMatches() const;
	int TrackedMapPoints(int minObs) const;
	MapPoint* GetMapPoint(size_t idx) const;

	// KeyPoint functions
	std::vector<size_t> GetFeaturesInArea(float x, float y, float r) const;
	Point3D UnprojectStereo(int i) const;

	// Image
	bool IsInImage(float x, float y) const;

	// Enable/Disable bad flag changes
	void SetNotErase();
	void SetErase();

	// Set/check bad flag
	void SetBadFlag();
	bool isBad() const;

	// Compute Scene Depth (q=2 median). Used in monocular.
	float ComputeSceneMedianDepth(int q) const;

	// The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

	static frameid_t nextId;
	frameid_t id;
	const frameid_t frameId;

	const double timestamp;

	// Grid (to speed up feature matching)
	FeaturesGrid grid;

	// Variables used by the tracking
	frameid_t trackReferenceForFrame;
	frameid_t fuseTargetForKF;

	// Variables used by the local mapping
	frameid_t BALocalForKF;
	frameid_t BAFixedForKF;

	// Variables used by the keyframe database
	frameid_t loopQuery;
	int loopWords;
	float loopScore;
	frameid_t relocQuery;
	int relocWords;
	float relocScore;

	// Variables used by loop closing
	CameraPose TcwGBA;
	CameraPose TcwBefGBA;
	frameid_t BAGlobalForKF;

	// Calibration parameters
	const CameraParams camera;
	const float thDepth;

	// Number of KeyPoints
	const int N;

	// KeyPoints, stereo coordinate and descriptors (all associated by an index)
	const std::vector<cv::KeyPoint> keypointsL;
	const std::vector<cv::KeyPoint> keypointsUn;
	const std::vector<float> uright; // negative value for monocular points
	const std::vector<float> depth; // negative value for monocular points
	const cv::Mat descriptorsL;

	//BoW
	DBoW2::BowVector bowVector;
	DBoW2::FeatureVector featureVector;

	// Pose relative to parent (this is computed when bad flag is activated)
	CameraPose Tcp;

	// Scale
	ScalePyramidInfo pyramid;

	// Image bounds and calibration
	ImageBounds imageBounds;

	// The following variables need to be accessed trough a mutex to be thread safe.
protected:

	// SE3 Pose and camera center
	CameraPose pose_;
	
	cv::Mat Cw_; // Stereo middel point. Only for visualization

	// MapPoints associated to keypoints
	std::vector<MapPoint*> mappoints_;

	// BoW
	KeyFrameDatabase* keyFrameDB_;
	ORBVocabulary* voc_;

	std::map<KeyFrame*, int> connectionTo_;
	std::vector<KeyFrame*> orderedConnectedKeyFrames_;
	std::vector<int> orderedWeights_;

	// Spanning Tree and Loop Edges
	bool firstConnection_;
	KeyFrame* parent_;
	std::set<KeyFrame*> children_;
	std::set<KeyFrame*> loopEdges_;

	// Bad flags
	bool notErase_;
	bool toBeErased_;
	bool bad_;

	float halfBaseline_; // Only for visualization

	Map* map_;

	mutable std::mutex mutexPose_;
	mutable std::mutex mutexConnections_;
	mutable std::mutex mutexFeatures_;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
