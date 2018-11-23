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

#include "Frame.h"
#include "ORBVocabulary.h"
#include "CameraParameters.h"

namespace ORB_SLAM2
{

class Map;
class MapPoint;
class KeyFrameDatabase;

class KeyFrame
{
public:

	using frameid_t = Frame::frameid_t;

	KeyFrame(const Frame& frame, Map* map, KeyFrameDatabase* keyframeDB);

	// Pose functions
	void SetPose(const cv::Mat &Tcw);
	cv::Mat GetPose() const;
	cv::Mat GetPoseInverse() const;
	cv::Mat GetCameraCenter() const;
	cv::Mat GetStereoCenter() const;
	cv::Mat GetRotation() const;
	cv::Mat GetTranslation() const;

	// Bag of Words Representation
	void ComputeBoW();

	// Covisibility graph functions
	void AddConnection(KeyFrame* keyframe, int weight);
	void EraseConnection(KeyFrame* pKF);
	void UpdateConnections();
	void UpdateBestCovisibles();
	std::set<KeyFrame *> GetConnectedKeyFrames() const;
	std::vector<KeyFrame* > GetVectorCovisibleKeyFrames() const;
	std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(int N) const;
	std::vector<KeyFrame*> GetCovisiblesByWeight(int w) const;
	int GetWeight(KeyFrame* pKF) const;

	// Spanning tree functions
	void AddChild(KeyFrame* pKF);
	void EraseChild(KeyFrame* pKF);
	void ChangeParent(KeyFrame* pKF);
	std::set<KeyFrame*> GetChildren() const;
	KeyFrame* GetParent() const;
	bool hasChild(KeyFrame* pKF) const;

	// Loop Edges
	void AddLoopEdge(KeyFrame* pKF);
	std::set<KeyFrame*> GetLoopEdges() const;

	// MapPoint observation functions
	void AddMapPoint(MapPoint* pMP, size_t idx);
	void EraseMapPointMatch(size_t idx);
	void EraseMapPointMatch(MapPoint* pMP);
	void ReplaceMapPointMatch(size_t idx, MapPoint* pMP);
	std::set<MapPoint*> GetMapPoints() const;
	std::vector<MapPoint*> GetMapPointMatches() const;
	int TrackedMapPoints(int minObs) const;
	MapPoint* GetMapPoint(size_t idx) const;

	// KeyPoint functions
	std::vector<size_t> GetFeaturesInArea(float x, float y, float r) const;
	cv::Mat UnprojectStereo(int i) const;

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

	static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) {
		return pKF1->id < pKF2->id;
	}


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
	cv::Mat TcwGBA;
	cv::Mat TcwBefGBA;
	frameid_t BAGlobalForKF;

	// Calibration parameters
	const CameraParams camera;
	const float thDepth;
	//const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

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
	cv::Mat Tcp;

	// Scale
	ScalePyramidInfo pyramid;

	// Image bounds and calibration
	ImageBounds imageBounds;

	// The following variables need to be accessed trough a mutex to be thread safe.
protected:

	// SE3 Pose and camera center
	cv::Mat Tcw;
	cv::Mat Twc;
	cv::Mat Ow;

	cv::Mat Cw; // Stereo middel point. Only for visualization

	// MapPoints associated to keypoints
	std::vector<MapPoint*> mappoints_;

	// BoW
	KeyFrameDatabase* keyFrameDB_;
	ORBVocabulary* voc_;

	// Grid over the image to speed up feature matching
	//std::vector<std::vector <std::vector<size_t>>> mGrid;

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
