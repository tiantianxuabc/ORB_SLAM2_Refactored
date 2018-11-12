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
#include "Optimizer.h"
#include "PnPsolver.h"
#include "Usleep.h"

using namespace std;

namespace ORB_SLAM2
{

TrackPoint::TrackPoint(const Frame& frame, bool lost)
	: pReferenceKF(frame.mpReferenceKF), timestamp(frame.mTimeStamp), lost(lost)
{
	Tcr = frame.mTcw * frame.mpReferenceKF->GetPoseInverse();
}

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

struct LastFrameInfo
{
	KeyFrame* keyFrame;
	Frame frame;
	int keyFrameId;
	int relocFrameId;
	LastFrameInfo() : keyFrame(nullptr), keyFrameId(0), relocFrameId(0) {}
};

class BoWTracker
{
public:

	BoWTracker(const LastFrameInfo& Last) : mLast(Last) {}

	bool Track(Frame& mCurrentFrame, KeyFrame* mpReferenceKF) const
	{
		// Compute Bag of Words vector
		mCurrentFrame.ComputeBoW();

		// We perform first an ORB matching with the reference keyframe
		// If enough matches are found we setup a PnP solver
		ORBmatcher matcher(0.7, true);
		vector<MapPoint*> vpMapPointMatches;

		int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

		if (nmatches < 15)
			return false;

		mCurrentFrame.mvpMapPoints = vpMapPointMatches;
		mCurrentFrame.SetPose(mLast.frame.mTcw);

		Optimizer::PoseOptimization(&mCurrentFrame);

		// Discard outliers
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
					nmatches--;
				}
				else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
					nmatchesMap++;
			}
		}

		return nmatchesMap >= 10;
	}

private:
	const LastFrameInfo& mLast;
};

class NewKeyFrameCondition
{
public:

	using Parameters = Tracking::Parameters;

	NewKeyFrameCondition(Map* pMap, const LocalMap& LocalMap, const LastFrameInfo& last, const Parameters& param, int sensor)
		: mpMap(pMap), mLocalMap(LocalMap), mLast(last), param_(param), mSensor(sensor) {}

	bool Satisfy(const Frame& mCurrentFrame, LocalMapping* mpLocalMapper, int mnMatchesInliers) const
	{
		/*if (mbLocalizationMode)
			return false;*/

		// If Local Mapping is freezed by a Loop Closure do not insert keyframes
		if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
			return false;

		const int nKFs = mpMap->KeyFramesInMap();

		// Do not insert keyframes if not enough frames have passed from last relocalisation
		if (mCurrentFrame.mnId<mLast.relocFrameId + param_.maxFrames && nKFs>param_.maxFrames)
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
		const bool c1a = mCurrentFrame.mnId >= mLast.keyFrameId + param_.maxFrames;
		// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
		const bool c1b = (mCurrentFrame.mnId >= mLast.keyFrameId + param_.minFrames && bLocalMappingIdle);
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
	const LastFrameInfo& mLast;
	const Parameters& param_;
	int mSensor;
};

static void ConvertToGray(const cv::Mat& src, cv::Mat& dst, bool RGB = false)
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
		ORBmatcher matcher(0.8);
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

class TrackingImpl : public Tracking
{

public:

	TrackingImpl::TrackingImpl(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer,
		Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor) :
		mState(STATE_NO_IMAGES), mSensor(sensor), mbLocalizationMode(false), mbVO(false), mpORBVocabulary(pVoc),
		mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
		mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mLocalMap(pMap),
		mBoWTracker(mLast), mNewKeyFrameCondition(pMap, mLocalMap, mLast, param_, sensor)
	{
		// Load camera parameters from settings file

		cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
		float fx = fSettings["Camera.fx"];
		float fy = fSettings["Camera.fy"];
		float cx = fSettings["Camera.cx"];
		float cy = fSettings["Camera.cy"];

		cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
		K.at<float>(0, 0) = fx;
		K.at<float>(1, 1) = fy;
		K.at<float>(0, 2) = cx;
		K.at<float>(1, 2) = cy;
		K.copyTo(mK);

		cv::Mat DistCoef(4, 1, CV_32F);
		DistCoef.at<float>(0) = fSettings["Camera.k1"];
		DistCoef.at<float>(1) = fSettings["Camera.k2"];
		DistCoef.at<float>(2) = fSettings["Camera.p1"];
		DistCoef.at<float>(3) = fSettings["Camera.p2"];
		const float k3 = fSettings["Camera.k3"];
		if (k3 != 0)
		{
			DistCoef.resize(5);
			DistCoef.at<float>(4) = k3;
		}
		DistCoef.copyTo(mDistCoef);

		mbf = fSettings["Camera.bf"];

		float fps = fSettings["Camera.fps"];
		if (fps == 0)
			fps = 30;

		// Max/Min Frames to insert keyframes and to check relocalisation
		param_.minFrames = 0;
		param_.maxFrames = fps;

		cout << endl << "Camera Parameters: " << endl;
		cout << "- fx: " << fx << endl;
		cout << "- fy: " << fy << endl;
		cout << "- cx: " << cx << endl;
		cout << "- cy: " << cy << endl;
		cout << "- k1: " << DistCoef.at<float>(0) << endl;
		cout << "- k2: " << DistCoef.at<float>(1) << endl;
		if (DistCoef.rows == 5)
			cout << "- k3: " << DistCoef.at<float>(4) << endl;
		cout << "- p1: " << DistCoef.at<float>(2) << endl;
		cout << "- p2: " << DistCoef.at<float>(3) << endl;
		cout << "- fps: " << fps << endl;


		int nRGB = fSettings["Camera.RGB"];
		mbRGB = nRGB;

		if (mbRGB)
			cout << "- color order: RGB (ignored if grayscale)" << endl;
		else
			cout << "- color order: BGR (ignored if grayscale)" << endl;

		// Load ORB parameters

		int nFeatures = fSettings["ORBextractor.nFeatures"];
		float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
		int nLevels = fSettings["ORBextractor.nLevels"];
		int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
		int fMinThFAST = fSettings["ORBextractor.minThFAST"];

		mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

		if (sensor == System::STEREO)
			mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

		if (sensor == System::MONOCULAR)
			mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

		cout << endl << "ORB Extractor Parameters: " << endl;
		cout << "- Number of Features: " << nFeatures << endl;
		cout << "- Scale Levels: " << nLevels << endl;
		cout << "- Scale Factor: " << fScaleFactor << endl;
		cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
		cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

		if (sensor == System::STEREO || sensor == System::RGBD)
		{
			param_.thDepth = mbf*(float)fSettings["ThDepth"] / fx;
			cout << endl << "Depth Threshold (Close/Far Points): " << param_.thDepth << endl;
		}

		if (sensor == System::RGBD)
		{
			mDepthMapFactor = fSettings["DepthMapFactor"];
			if (fabs(mDepthMapFactor) < 1e-5)
				mDepthMapFactor = 1;
			else
				mDepthMapFactor = 1.0f / mDepthMapFactor;
		}

	}

	// Preprocess the input and call Track(). Extract features and performs stereo matching.
	cv::Mat GrabImageStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) override
	{
		cv::Mat imGrayRight;
		ConvertToGray(imageL, mImGray);
		ConvertToGray(imageR, imGrayRight);

		mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, param_.thDepth);

		Track();

		return mCurrentFrame.mTcw.clone();
	}

	cv::Mat GrabImageRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) override
	{
		ConvertToGray(image, mImGray);

		cv::Mat imDepth = depth;
		if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
			imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

		mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, param_.thDepth);

		Track();

		return mCurrentFrame.mTcw.clone();
	}

	cv::Mat GrabImageMonocular(const cv::Mat& image, double timestamp) override
	{
		ConvertToGray(image, mImGray);

		if (mState == STATE_NOT_INITIALIZED || mState == STATE_NO_IMAGES)
			mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, param_.thDepth);
		else
			mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, param_.thDepth);

		Track();

		return mCurrentFrame.mTcw.clone();
	}

	void SetLocalMapper(LocalMapping *pLocalMapper) override
	{
		mpLocalMapper = pLocalMapper;
	}

	void SetLoopClosing(LoopClosing *pLoopClosing) override
	{
		mpLoopClosing = pLoopClosing;
	}

	void SetViewer(Viewer *pViewer) override
	{
		mpViewer = pViewer;
	}

	// Load new settings
	// The focal lenght should be similar or scale prediction will fail when projecting points
	// TODO: Modify MapPoint::PredictScale to take into account focal lenght
	void ChangeCalibration(const string &strSettingPath) override
	{
		cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
		float fx = fSettings["Camera.fx"];
		float fy = fSettings["Camera.fy"];
		float cx = fSettings["Camera.cx"];
		float cy = fSettings["Camera.cy"];

		cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
		K.at<float>(0, 0) = fx;
		K.at<float>(1, 1) = fy;
		K.at<float>(0, 2) = cx;
		K.at<float>(1, 2) = cy;
		K.copyTo(mK);

		cv::Mat DistCoef(4, 1, CV_32F);
		DistCoef.at<float>(0) = fSettings["Camera.k1"];
		DistCoef.at<float>(1) = fSettings["Camera.k2"];
		DistCoef.at<float>(2) = fSettings["Camera.p1"];
		DistCoef.at<float>(3) = fSettings["Camera.p2"];
		const float k3 = fSettings["Camera.k3"];
		if (k3 != 0)
		{
			DistCoef.resize(5);
			DistCoef.at<float>(4) = k3;
		}
		DistCoef.copyTo(mDistCoef);

		mbf = fSettings["Camera.bf"];

		Frame::mbInitialComputations = true;
	}

	void Reset() override
	{
		cout << "System Reseting" << endl;
		if (mpViewer)
		{
			mpViewer->RequestStop();
			while (!mpViewer->isStopped())
				usleep(3000);
		}

		// Reset Local Mapping
		cout << "Reseting Local Mapper...";
		mpLocalMapper->RequestReset();
		cout << " done" << endl;

		// Reset Loop Closing
		cout << "Reseting Loop Closing...";
		mpLoopClosing->RequestReset();
		cout << " done" << endl;

		// Clear BoW Database
		cout << "Reseting Database...";
		mpKeyFrameDB->clear();
		cout << " done" << endl;

		// Clear Map (this erase MapPoints and KeyFrames)
		mpMap->clear();

		KeyFrame::nNextId = 0;
		Frame::nNextId = 0;
		mState = STATE_NO_IMAGES;

		if (mpInitializer)
		{
			delete mpInitializer;
			mpInitializer = static_cast<Initializer*>(NULL);
		}

		trajectory_.clear();

		if (mpViewer)
			mpViewer->Release();
	}

	// Use this function if you have deactivated local mapping and you only want to localize the camera.
	void InformOnlyTracking(const bool &flag) override
	{
		mbLocalizationMode = flag;
	}

	int GetState() const override
	{
		return mState;
	}

	int GetLastProcessedState() const override
	{
		return mLastProcessedState;
	}

	const Frame& GetCurrentFrame() const override
	{
		return mCurrentFrame;
	}

	const Frame& GetInitialFrame() const override
	{
		return mInitialFrame;
	}

	cv::Mat GetImGray() const override
	{
		return mImGray;
	}

	const std::vector<int>& GetIniMatches() const override
	{
		return mvIniMatches;
	}

	const std::vector<TrackPoint>& GetTrajectory() const override
	{
		return trajectory_;
	}

	bool OnlyTracking() const override
	{
		return mbLocalizationMode;
	}

public:

	State mState;
	State mLastProcessedState;

	// Input sensor
	int mSensor;

	// Current Frame
	Frame mCurrentFrame;
	cv::Mat mImGray;

	// Initialization Variables (Monocular)
	std::vector<int> mvIniLastMatches;
	std::vector<int> mvIniMatches;
	std::vector<cv::Point2f> mvbPrevMatched;
	std::vector<cv::Point3f> mvIniP3D;
	Frame mInitialFrame;

	// Lists used to recover the full camera trajectory at the end of the execution.
	// Basically we store the reference keyframe for each frame and its relative transformation
	std::vector<TrackPoint> trajectory_;

	// True if local mapping is deactivated and we are performing only localization
	bool mbLocalizationMode;

private:

	// Main tracking function. It is independent of the input sensor.
	void Track()
	{
		if (mState == STATE_NO_IMAGES)
		{
			mState = STATE_NOT_INITIALIZED;
		}

		mLastProcessedState = mState;

		// Get Map Mutex -> Map cannot be changed
		unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

		if (mState == STATE_NOT_INITIALIZED)
		{
			if (mSensor == System::STEREO || mSensor == System::RGBD)
				StereoInitialization();
			else
				MonocularInitialization();

			mpFrameDrawer->Update(this);

			if (mState == STATE_OK)
				trajectory_.push_back(TrackPoint(mCurrentFrame, false));

			return;
		}

		// System is initialized. Track Frame.
		bool success = false;

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		if (!mbLocalizationMode)
		{
			// Local Mapping is activated. This is the normal behaviour, unless
			// you explicitly activate the "only tracking" mode.

			if (mState == STATE_OK)
			{
				// Local Mapping might have changed some MapPoints tracked in last frame
				for (int i = 0; i < mLast.frame.N; i++)
				{
					MapPoint* pMP = mLast.frame.mvpMapPoints[i];
					MapPoint* pRep = pMP ? pMP->GetReplaced() : nullptr;
					if (pRep)
						mLast.frame.mvpMapPoints[i] = pRep;
				}

				const bool withMotionModel = !mVelocity.empty() && mCurrentFrame.mnId >= mLast.relocFrameId + 2;
				if (withMotionModel)
				{
					success = TrackWithMotionModel();
				}
				if (!withMotionModel || (withMotionModel && !success))
				{
					success = mBoWTracker.Track(mCurrentFrame, mLocalMap.mpReferenceKF);
				}
			}
			else
			{
				success = Relocalization();
			}
		}
		else
		{
			// Localization Mode: Local Mapping is deactivated

			if (mState == STATE_LOST)
			{
				success = Relocalization();
			}
			else
			{
				if (!mbVO)
				{
					// In last frame we tracked enough MapPoints in the map

					if (!mVelocity.empty())
					{
						success = TrackWithMotionModel();
					}
					else
					{
						success = mBoWTracker.Track(mCurrentFrame, mLocalMap.mpReferenceKF);
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
					if (!mVelocity.empty())
					{
						bOKMM = TrackWithMotionModel();
						vpMPsMM = mCurrentFrame.mvpMapPoints;
						vbOutMM = mCurrentFrame.mvbOutlier;
						TcwMM = mCurrentFrame.mTcw.clone();
					}
					bOKReloc = Relocalization();

					if (bOKMM && !bOKReloc)
					{
						mCurrentFrame.SetPose(TcwMM);
						mCurrentFrame.mvpMapPoints = vpMPsMM;
						mCurrentFrame.mvbOutlier = vbOutMM;

						if (mbVO)
						{
							for (int i = 0; i < mCurrentFrame.N; i++)
							{
								if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
								{
									mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
								}
							}
						}
					}
					else if (bOKReloc)
					{
						mbVO = false;
					}

					success = bOKReloc || bOKMM;
				}
			}
		}

		mCurrentFrame.mpReferenceKF = mLocalMap.mpReferenceKF;

		// If we have an initial estimation of the camera pose and matching. Track the local map.
		// [In Localization Mode]
		// mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
		// a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
		// the camera we will use the local map again.
		if (success && (!mbLocalizationMode || (mbLocalizationMode && !mbVO)))
		{
			// If the camera has been relocalised recently, perform a coarser search
			const bool relocalizedRecently = mCurrentFrame.mnId < mLast.relocFrameId + 2;
			const float th = relocalizedRecently ? 5 : (mSensor == System::RGBD ? 3 : 1);

			mnMatchesInliers = TrackLocalMap(mLocalMap, mCurrentFrame, th, mbLocalizationMode, mSensor == System::STEREO);

			// Decide if the tracking was succesful
			// More restrictive if there was a relocalization recently
			const int minInliers = (mCurrentFrame.mnId < mLast.relocFrameId + param_.maxFrames) ? 50 : 30;
			success = mnMatchesInliers >= minInliers;
		}

		mState = success ? STATE_OK : STATE_LOST;
		
		// Update drawer
		mpFrameDrawer->Update(this);

		// If tracking were good, check if we insert a keyframe
		if (success)
		{
			// Update motion model
			if (!mLast.frame.mTcw.empty())
			{
				cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
				mLast.frame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
				mLast.frame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
				mVelocity = mCurrentFrame.mTcw*LastTwc;
			}
			else
				mVelocity = cv::Mat();

			mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

			// Clean VO matches
			for (int i = 0; i < mCurrentFrame.N; i++)
			{
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
				if (pMP)
					if (pMP->Observations() < 1)
					{
						mCurrentFrame.mvbOutlier[i] = false;
						mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
					}
			}

			// Delete temporal MapPoints
			for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
			{
				MapPoint* pMP = *lit;
				delete pMP;
			}
			mlpTemporalPoints.clear();

			// Check if we need to insert a new keyframe
			if (!mbLocalizationMode && mNewKeyFrameCondition.Satisfy(mCurrentFrame, mpLocalMapper, mnMatchesInliers))
			{
				if (mpLocalMapper->SetNotStop(true))
				{
					KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
					mLocalMap.mpReferenceKF = pKF;
					mCurrentFrame.mpReferenceKF = pKF;

					if (mSensor != System::MONOCULAR)
						CreateMapPoints(mCurrentFrame, pKF, mpMap, param_.thDepth);

					mpLocalMapper->InsertKeyFrame(pKF);
					mpLocalMapper->SetNotStop(false);
					mLast.keyFrameId = mCurrentFrame.mnId;
					mLast.keyFrame = pKF;
				}
			}

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame.
			for (int i = 0; i < mCurrentFrame.N; i++)
			{
				if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
			}
		}

		// Reset if the camera get lost soon after initialization
		if (mState == STATE_LOST)
		{
			if (mpMap->KeyFramesInMap() <= 5)
			{
				cout << "Track lost soon after initialisation, reseting..." << endl;
				mpSystem->Reset();
				return;
			}
		}

		if (!mCurrentFrame.mpReferenceKF)
			mCurrentFrame.mpReferenceKF = mLocalMap.mpReferenceKF;

		mLast.frame = Frame(mCurrentFrame);

		// Store frame pose information to retrieve the complete camera trajectory afterwards.
		CV_Assert(mCurrentFrame.mpReferenceKF == mLocalMap.mpReferenceKF);
		const bool lost = mState == STATE_LOST;
		if (!mCurrentFrame.mTcw.empty())
		{
			trajectory_.push_back(TrackPoint(mCurrentFrame, lost));
		}
		else
		{
			// This can happen if tracking is lost
			trajectory_.push_back(trajectory_.back());
			trajectory_.back().lost = lost;
		}
	}

	// Map initialization for stereo and RGB-D
	void StereoInitialization()
	{
		if (mCurrentFrame.N > 500)
		{
			// Set Frame pose to the origin
			mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

			// Create KeyFrame
			KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

			// Insert KeyFrame in the map
			mpMap->AddKeyFrame(pKFini);

			// Create MapPoints and asscoiate to KeyFrame
			for (int i = 0; i < mCurrentFrame.N; i++)
			{
				float z = mCurrentFrame.mvDepth[i];
				if (z > 0)
				{
					cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
					MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
					pNewMP->AddObservation(pKFini, i);
					pKFini->AddMapPoint(pNewMP, i);
					pNewMP->ComputeDistinctiveDescriptors();
					pNewMP->UpdateNormalAndDepth();
					mpMap->AddMapPoint(pNewMP);

					mCurrentFrame.mvpMapPoints[i] = pNewMP;
				}
			}

			cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

			mpLocalMapper->InsertKeyFrame(pKFini);

			mLast.frame = Frame(mCurrentFrame);
			mLast.keyFrameId = mCurrentFrame.mnId;
			mLast.keyFrame = pKFini;

			mLocalMap.mvpLocalKeyFrames.push_back(pKFini);
			mLocalMap.mvpLocalMapPoints = mpMap->GetAllMapPoints();
			mLocalMap.mpReferenceKF = pKFini;
			mCurrentFrame.mpReferenceKF = pKFini;

			mpMap->SetReferenceMapPoints(mLocalMap.mvpLocalMapPoints);

			mpMap->mvpKeyFrameOrigins.push_back(pKFini);

			mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

			mState = STATE_OK;
		}
	}

	// Map initialization for monocular
	void MonocularInitialization()
	{

		if (!mpInitializer)
		{
			// Set Reference Frame
			if (mCurrentFrame.mvKeys.size() > 100)
			{
				mInitialFrame = Frame(mCurrentFrame);
				mLast.frame = Frame(mCurrentFrame);
				mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
				for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
					mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

				if (mpInitializer)
					delete mpInitializer;

				mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

				fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

				return;
			}
		}
		else
		{
			// Try to initialize
			if ((int)mCurrentFrame.mvKeys.size() <= 100)
			{
				delete mpInitializer;
				mpInitializer = static_cast<Initializer*>(NULL);
				fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
				return;
			}

			// Find correspondences
			ORBmatcher matcher(0.9, true);
			int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

			// Check if there are enough correspondences
			if (nmatches < 100)
			{
				delete mpInitializer;
				mpInitializer = static_cast<Initializer*>(NULL);
				return;
			}

			cv::Mat Rcw; // Current Camera Rotation
			cv::Mat tcw; // Current Camera Translation
			vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

			if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
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

				CreateInitialMapMonocular();
			}
		}
	}

	void CreateInitialMapMonocular()
	{
		// Create KeyFrames
		KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
		KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);


		pKFini->ComputeBoW();
		pKFcur->ComputeBoW();

		// Insert KFs in the map
		mpMap->AddKeyFrame(pKFini);
		mpMap->AddKeyFrame(pKFcur);

		// Create MapPoints and asscoiate to keyframes
		for (size_t i = 0; i < mvIniMatches.size(); i++)
		{
			if (mvIniMatches[i] < 0)
				continue;

			//Create MapPoint.
			cv::Mat worldPos(mvIniP3D[i]);

			MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

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
			mpMap->AddMapPoint(pMP);
		}

		// Update Connections
		pKFini->UpdateConnections();
		pKFcur->UpdateConnections();

		// Bundle Adjustment
		cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

		Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

		// Set median depth to 1
		float medianDepth = pKFini->ComputeSceneMedianDepth(2);
		float invMedianDepth = 1.0f / medianDepth;

		if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
		{
			cout << "Wrong initialization, reseting..." << endl;
			Reset();
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

		mpLocalMapper->InsertKeyFrame(pKFini);
		mpLocalMapper->InsertKeyFrame(pKFcur);

		mCurrentFrame.SetPose(pKFcur->GetPose());
		mLast.keyFrameId = mCurrentFrame.mnId;
		mLast.keyFrame = pKFcur;

		mLocalMap.mvpLocalKeyFrames.push_back(pKFcur);
		mLocalMap.mvpLocalKeyFrames.push_back(pKFini);
		mLocalMap.mvpLocalMapPoints = mpMap->GetAllMapPoints();
		mLocalMap.mpReferenceKF = pKFcur;
		mCurrentFrame.mpReferenceKF = pKFcur;

		mLast.frame = Frame(mCurrentFrame);

		mpMap->SetReferenceMapPoints(mLocalMap.mvpLocalMapPoints);

		mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

		mpMap->mvpKeyFrameOrigins.push_back(pKFini);

		mState = STATE_OK;
	}

	bool TrackWithMotionModel()
	{
		ORBmatcher matcher(0.9, true);

		// Update last frame pose according to its reference keyframe
		// Create "visual odometry" points if in Localization Mode

		// Update pose according to reference keyframe
		KeyFrame* pRef = mLast.frame.mpReferenceKF;
		cv::Mat Tlr = trajectory_.back().Tcr;
		mLast.frame.SetPose(Tlr*pRef->GetPose());

		if (mbLocalizationMode && mSensor != System::MONOCULAR && mLast.frame.mnId != mLast.keyFrameId)
			CreateMapPointsVO(mLast.frame, mlpTemporalPoints, mpMap, param_.thDepth);

		mCurrentFrame.SetPose(mVelocity*mLast.frame.mTcw);

		fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));

		// Project points seen in previous frame
		int th;
		if (mSensor != System::STEREO)
			th = 15;
		else
			th = 7;
		int nmatches = matcher.SearchByProjection(mCurrentFrame, mLast.frame, th, mSensor == System::MONOCULAR);

		// If few matches, uses a wider window search
		if (nmatches < 20)
		{
			fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));
			nmatches = matcher.SearchByProjection(mCurrentFrame, mLast.frame, 2 * th, mSensor == System::MONOCULAR);
		}

		if (nmatches < 20)
			return false;

		// Optimize frame pose with all matches
		Optimizer::PoseOptimization(&mCurrentFrame);

		// Discard outliers
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
					nmatches--;
				}
				else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
					nmatchesMap++;
			}
		}

		if (mbLocalizationMode)
		{
			mbVO = nmatchesMap < 10;
			return nmatches > 20;
		}

		return nmatchesMap >= 10;
	}

	bool Relocalization()
	{
		// Compute Bag of Words Vector
		mCurrentFrame.ComputeBoW();

		// Relocalization is performed when tracking is lost
		// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
		vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

		if (vpCandidateKFs.empty())
			return false;

		const int nKFs = vpCandidateKFs.size();

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
					pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
					vpPnPsolvers[i] = pSolver;
					nCandidates++;
				}
			}
		}

		// Alternatively perform some iterations of P4P RANSAC
		// Until we found a camera pose supported by enough inliers
		bool bMatch = false;
		ORBmatcher matcher2(0.9, true);

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

					const int np = vbInliers.size();

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
			mLast.relocFrameId = mCurrentFrame.mnId;
			return true;
		}
	}

	// In case of performing only localization, this flag is true when there are no matches to
	// points in the map. Still tracking will continue if there are enough matches with temporal points.
	// In that case we are doing visual odometry. The system will try to do relocalization to recover
	// "zero-drift" localization to the map.
	bool mbVO;

	//Other Thread Pointers
	LocalMapping* mpLocalMapper;
	LoopClosing* mpLoopClosing;

	//ORB
	ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
	ORBextractor* mpIniORBextractor;

	//BoW
	ORBVocabulary* mpORBVocabulary;
	KeyFrameDatabase* mpKeyFrameDB;

	// Initalization (only for monocular)
	Initializer* mpInitializer;

	//Local Map
	LocalMap mLocalMap;
	
	// System
	System* mpSystem;

	//Drawers
	Viewer* mpViewer;
	FrameDrawer* mpFrameDrawer;
	MapDrawer* mpMapDrawer;

	//Map
	Map* mpMap;

	//Calibration matrix
	cv::Mat mK;
	cv::Mat mDistCoef;
	float mbf;

	// Parameters
	Parameters param_;

	// For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
	float mDepthMapFactor;

	//Current matches in frame
	int mnMatchesInliers;

	//Last Frame, KeyFrame and Relocalisation Info
	LastFrameInfo mLast;

	//Motion Model
	cv::Mat mVelocity;

	//Color order (true RGB, false BGR, ignored if grayscale)
	bool mbRGB;

	list<MapPoint*> mlpTemporalPoints;

	BoWTracker mBoWTracker;
	NewKeyFrameCondition mNewKeyFrameCondition;
};

std::shared_ptr<Tracking> Tracking::Create(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
	KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor)
{
	return std::make_shared<TrackingImpl>(pSys, pVoc, pFrameDrawer, pMapDrawer, pMap, pKFDB, strSettingPath, sensor);
}

} //namespace ORB_SLAM
