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
	KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

	// Pose functions
	void SetPose(const cv::Mat &Tcw);
	cv::Mat GetPose();
	cv::Mat GetPoseInverse();
	cv::Mat GetCameraCenter();
	cv::Mat GetStereoCenter();
	cv::Mat GetRotation();
	cv::Mat GetTranslation();

	// Bag of Words Representation
	void ComputeBoW();

	// Covisibility graph functions
	void AddConnection(KeyFrame* keyframe, int weight);
	void EraseConnection(KeyFrame* pKF);
	void UpdateConnections();
	void UpdateBestCovisibles();
	std::set<KeyFrame *> GetConnectedKeyFrames();
	std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
	std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
	std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
	int GetWeight(KeyFrame* pKF);

	// Spanning tree functions
	void AddChild(KeyFrame* pKF);
	void EraseChild(KeyFrame* pKF);
	void ChangeParent(KeyFrame* pKF);
	std::set<KeyFrame*> GetChilds();
	KeyFrame* GetParent();
	bool hasChild(KeyFrame* pKF);

	// Loop Edges
	void AddLoopEdge(KeyFrame* pKF);
	std::set<KeyFrame*> GetLoopEdges();

	// MapPoint observation functions
	void AddMapPoint(MapPoint* pMP, const size_t &idx);
	void EraseMapPointMatch(const size_t &idx);
	void EraseMapPointMatch(MapPoint* pMP);
	void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
	std::set<MapPoint*> GetMapPoints();
	std::vector<MapPoint*> GetMapPointMatches();
	int TrackedMapPoints(const int &minObs);
	MapPoint* GetMapPoint(const size_t &idx);

	// KeyPoint functions
	std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
	cv::Mat UnprojectStereo(int i);

	// Image
	bool IsInImage(const float &x, const float &y) const;

	// Enable/Disable bad flag changes
	void SetNotErase();
	void SetErase();

	// Set/check bad flag
	void SetBadFlag();
	bool isBad();

	// Compute Scene Depth (q=2 median). Used in monocular.
	float ComputeSceneMedianDepth(const int q);

	static bool weightComp(int a, int b) {
		return a > b;
	}

	static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) {
		return pKF1->id < pKF2->id;
	}


	// The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

	static long unsigned int nextId;
	long unsigned int id;
	const long unsigned int frameId;

	const double timestamp;

	// Grid (to speed up feature matching)
	FeaturesGrid grid;

	// Variables used by the tracking
	long unsigned int trackReferenceForFrame;
	long unsigned int fuseTargetForKF;

	// Variables used by the local mapping
	long unsigned int BALocalForKF;
	long unsigned int BAFixedForKF;

	// Variables used by the keyframe database
	long unsigned int loopQuery;
	int loopWords;
	float loopScore;
	long unsigned int relocQuery;
	int relocWords;
	float relocScore;

	// Variables used by loop closing
	cv::Mat TcwGBA;
	cv::Mat TcwBefGBA;
	long unsigned int BAGlobalForKF;

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

	std::mutex mutexPose_;
	std::mutex mutexConnections_;
	std::mutex mutexFeatures_;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
