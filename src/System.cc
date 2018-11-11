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



#include "System.h"

#include <thread>
#include <iomanip>
#include <Eigen/Geometry>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "Usleep.h"

namespace ORB_SLAM2
{

using namespace std;

namespace Converter
{

static std::vector<float> toQuaternion(const cv::Mat1f &M)
{
	Eigen::Matrix3d eigMat;
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) eigMat(i, j) = M(i, j);
	Eigen::Quaterniond q(eigMat);

	std::vector<float> v(4);
	v[0] = q.x();
	v[1] = q.y();
	v[2] = q.z();
	v[3] = q.w();

	return v;
}

} // namespace Converter

#define LOCK_MUTEX_RESET() unique_lock<mutex> lock1(mMutexReset);
#define LOCK_MUTEX_MODE()  unique_lock<mutex> lock2(mMutexMode);
#define LOCK_MUTEX_STATE() unique_lock<mutex> lock3(mMutexState);

class SystemImpl : public System
{
public:

	using Path = System::Path;

	// Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
	SystemImpl(const Path& vocabularyFile, const Path& settingsFile, Sensor sensor, bool useViewer)
		: mSensor(sensor), mpViewer(nullptr), mbReset(false), mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false)
	{
		// Output welcome message
		cout << endl <<
			"ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
			"This program comes with ABSOLUTELY NO WARRANTY;" << endl <<
			"This is free software, and you are welcome to redistribute it" << endl <<
			"under certain conditions. See LICENSE.txt." << endl << endl;

		cout << "Input sensor was set to: ";

		const char* sensors[3] = { "Monocular", "Stereo", "RGB-D" };
		cout << sensors[mSensor] << endl;

		//Check settings file
		cv::FileStorage settings(settingsFile.c_str(), cv::FileStorage::READ);
		if (!settings.isOpened())
		{
			cerr << "Failed to open settings file at: " << settingsFile << endl;
			exit(-1);
		}

		//Load ORB Vocabulary
		cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

		mpVocabulary = new ORBVocabulary();
		bool bVocLoad = mpVocabulary->loadFromTextFile(vocabularyFile);
		if (!bVocLoad)
		{
			cerr << "Wrong path to vocabulary. " << endl;
			cerr << "Falied to open at: " << vocabularyFile << endl;
			exit(-1);
		}
		cout << "Vocabulary loaded!" << endl << endl;

		//Create KeyFrame Database
		mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

		//Create the Map
		mpMap = new Map();

		//Create Drawers. These are used by the Viewer
		mpFrameDrawer = new FrameDrawer(mpMap);
		mpMapDrawer = new MapDrawer(mpMap, settingsFile);

		//Initialize the Tracking thread
		//(it will live in the main thread of execution, the one that called this constructor)
		mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
			mpMap, mpKeyFrameDatabase, settingsFile, mSensor);

		//Initialize the Local Mapping thread and launch
		mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
		mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

		//Initialize the Loop Closing thread and launch
		mpLoopCloser = LoopClosing::Create(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
		mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

		//Initialize the Viewer thread and launch
		if (useViewer)
		{
			mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, settingsFile);
			mptViewer = new thread(&Viewer::Run, mpViewer);
			mpTracker->SetViewer(mpViewer);
		}

		//Set pointers between threads
		mpTracker->SetLocalMapper(mpLocalMapper);
		mpTracker->SetLoopClosing(mpLoopCloser.get());

		mpLocalMapper->SetTracker(mpTracker);
		mpLocalMapper->SetLoopCloser(mpLoopCloser.get());

		mpLoopCloser->SetTracker(mpTracker);
		mpLoopCloser->SetLocalMapper(mpLocalMapper);
	}

	// Proccess the given stereo frame. Images must be synchronized and rectified.
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) override
	{
		if (mSensor != STEREO)
		{
			cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
			exit(-1);
		}

		// Check mode change
		{
			LOCK_MUTEX_MODE();
			if (mbActivateLocalizationMode)
			{
				mpLocalMapper->RequestStop();

				// Wait until Local Mapping has effectively stopped
				while (!mpLocalMapper->isStopped())
				{
					usleep(1000);
				}

				mpTracker->InformOnlyTracking(true);
				mbActivateLocalizationMode = false;
			}
			if (mbDeactivateLocalizationMode)
			{
				mpTracker->InformOnlyTracking(false);
				mpLocalMapper->Release();
				mbDeactivateLocalizationMode = false;
			}
		}

		// Check reset
		{
			LOCK_MUTEX_RESET();
			if (mbReset)
			{
				mpTracker->Reset();
				mbReset = false;
			}
		}

		cv::Mat Tcw = mpTracker->GrabImageStereo(imageL, imageR, timestamp);

		LOCK_MUTEX_STATE();
		mTrackingState = mpTracker->mState;
		mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
		mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
		return Tcw;
	}

	// Process the given rgbd frame. Depthmap must be registered to the RGB frame.
	// Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Input depthmap: Float (CV_32F).
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) override
	{
		if (mSensor != RGBD)
		{
			cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
			exit(-1);
		}

		// Check mode change
		{
			LOCK_MUTEX_MODE();
			if (mbActivateLocalizationMode)
			{
				mpLocalMapper->RequestStop();

				// Wait until Local Mapping has effectively stopped
				while (!mpLocalMapper->isStopped())
				{
					usleep(1000);
				}

				mpTracker->InformOnlyTracking(true);
				mbActivateLocalizationMode = false;
			}
			if (mbDeactivateLocalizationMode)
			{
				mpTracker->InformOnlyTracking(false);
				mpLocalMapper->Release();
				mbDeactivateLocalizationMode = false;
			}
		}

		// Check reset
		{
			LOCK_MUTEX_RESET();
			if (mbReset)
			{
				mpTracker->Reset();
				mbReset = false;
			}
		}

		cv::Mat Tcw = mpTracker->GrabImageRGBD(image, depth, timestamp);

		LOCK_MUTEX_STATE();
		mTrackingState = mpTracker->mState;
		mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
		mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
		return Tcw;
	}

	// Proccess the given monocular frame
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackMonocular(const cv::Mat& image, double timestamp) override
	{
		if (mSensor != MONOCULAR)
		{
			cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
			exit(-1);
		}

		// Check mode change
		{
			LOCK_MUTEX_MODE();
			if (mbActivateLocalizationMode)
			{
				mpLocalMapper->RequestStop();

				// Wait until Local Mapping has effectively stopped
				while (!mpLocalMapper->isStopped())
				{
					usleep(1000);
				}

				mpTracker->InformOnlyTracking(true);
				mbActivateLocalizationMode = false;
			}
			if (mbDeactivateLocalizationMode)
			{
				mpTracker->InformOnlyTracking(false);
				mpLocalMapper->Release();
				mbDeactivateLocalizationMode = false;
			}
		}

		// Check reset
		{
			LOCK_MUTEX_RESET();
			if (mbReset)
			{
				mpTracker->Reset();
				mbReset = false;
			}
		}

		cv::Mat Tcw = mpTracker->GrabImageMonocular(image, timestamp);

		LOCK_MUTEX_STATE();
		mTrackingState = mpTracker->mState;
		mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
		mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

		return Tcw;
	}

	// This stops local mapping thread (map building) and performs only camera tracking.
	void ActivateLocalizationMode() override
	{
		LOCK_MUTEX_MODE();
		mbActivateLocalizationMode = true;
	}

	// This resumes local mapping thread and performs SLAM again.
	void DeactivateLocalizationMode() override
	{
		LOCK_MUTEX_MODE();
		mbDeactivateLocalizationMode = true;
	}

	// Returns true if there have been a big map change (loop closure, global BA)
	// since last call to this function
	bool MapChanged() const override
	{
		static int n = 0;
		int curn = mpMap->GetLastBigChangeIdx();
		if (n < curn)
		{
			n = curn;
			return true;
		}
		else
			return false;
	}

	// Reset the system (clear map)
	void Reset() override
	{
		LOCK_MUTEX_RESET();
		mbReset = true;
	}

	// All threads will be requested to finish.
	// It waits until all threads have finished.
	// This function must be called before saving the trajectory.
	void Shutdown() override
	{
		mpLocalMapper->RequestFinish();
		mpLoopCloser->RequestFinish();
		if (mpViewer)
		{
			mpViewer->RequestFinish();
			while (!mpViewer->isFinished())
				usleep(5000);
		}

		// Wait until all thread have effectively stopped
		while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
		{
			usleep(5000);
		}
	}

	// Save camera trajectory in the TUM RGB-D dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	void SaveTrajectoryTUM(const Path& filename) const override
	{
		cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
		if (mSensor == MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
			return;
		}

		vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		cv::Mat Two = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		list<bool>::iterator lbL = mpTracker->mlbLost.begin();
		for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
			lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
		{
			if (*lbL)
				continue;

			KeyFrame* pKF = *lRit;

			cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

			// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
			while (pKF->isBad())
			{
				Trw = Trw*pKF->mTcp;
				pKF = pKF->GetParent();
			}

			Trw = Trw*pKF->GetPose()*Two;

			cv::Mat Tcw = (*lit)*Trw;
			cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
			cv::Mat twc = -Rwc*Tcw.rowRange(0, 3).col(3);

			vector<float> q = Converter::toQuaternion(Rwc);

			f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
		}
		f.close();
		cout << endl << "trajectory saved!" << endl;
	}

	// Save keyframe poses in the TUM RGB-D dataset format.
	// This method works for all sensor input.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	void SaveKeyFrameTrajectoryTUM(const Path& filename) const override
	{
		cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

		vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		//cv::Mat Two = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];

			// pKF->SetPose(pKF->GetPose()*Two);

			if (pKF->isBad())
				continue;

			cv::Mat R = pKF->GetRotation().t();
			vector<float> q = Converter::toQuaternion(R);
			cv::Mat t = pKF->GetCameraCenter();
			f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
				<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

		}

		f.close();
		cout << endl << "trajectory saved!" << endl;
	}

	// Save camera trajectory in the KITTI dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
	void SaveTrajectoryKITTI(const Path& filename) const override
	{
		cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
		if (mSensor == MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
			return;
		}

		vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		cv::Mat Two = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++)
		{
			ORB_SLAM2::KeyFrame* pKF = *lRit;

			cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

			while (pKF->isBad())
			{
				//  cout << "bad parent" << endl;
				Trw = Trw*pKF->mTcp;
				pKF = pKF->GetParent();
			}

			Trw = Trw*pKF->GetPose()*Two;

			cv::Mat Tcw = (*lit)*Trw;
			cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
			cv::Mat twc = -Rwc*Tcw.rowRange(0, 3).col(3);

			f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " " <<
				Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " " <<
				Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
		}
		f.close();
		cout << endl << "trajectory saved!" << endl;
	}

	// TODO: Save/Load functions
	// SaveMap(const Path& filename);
	// LoadMap(const Path& filename);

	// Information from most recent processed frame
	// You can call this right after TrackMonocular (or stereo or RGBD)
	int GetTrackingState() const override
	{
		LOCK_MUTEX_STATE();
		return mTrackingState;
	}

	vector<MapPoint*> GetTrackedMapPoints() const override
	{
		LOCK_MUTEX_STATE();
		return mTrackedMapPoints;
	}

	vector<cv::KeyPoint> GetTrackedKeyPointsUn() const override
	{
		LOCK_MUTEX_STATE();
		return mTrackedKeyPointsUn;
	}

private:

	// Input sensor
	Sensor mSensor;

	// ORB vocabulary used for place recognition and feature matching.
	ORBVocabulary* mpVocabulary;

	// KeyFrame database for place recognition (relocalization and loop detection).
	KeyFrameDatabase* mpKeyFrameDatabase;

	// Map structure that stores the pointers to all KeyFrames and MapPoints.
	Map* mpMap;

	// Tracker. It receives a frame and computes the associated camera pose.
	// It also decides when to insert a new keyframe, create some new MapPoints and
	// performs relocalization if tracking fails.
	Tracking* mpTracker;

	// Local Mapper. It manages the local map and performs local bundle adjustment.
	LocalMapping* mpLocalMapper;

	// Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
	// a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
	std::shared_ptr<LoopClosing> mpLoopCloser;

	// The viewer draws the map and the current camera pose. It uses Pangolin.
	Viewer* mpViewer;

	FrameDrawer* mpFrameDrawer;
	MapDrawer* mpMapDrawer;

	// System threads: Local Mapping, Loop Closing, Viewer.
	// The Tracking thread "lives" in the main execution thread that creates the System object.
	std::thread* mptLocalMapping;
	std::thread* mptLoopClosing;
	std::thread* mptViewer;

	// Reset flag
	mutable std::mutex mMutexReset;
	bool mbReset;

	// Change mode flags
	mutable std::mutex mMutexMode;
	bool mbActivateLocalizationMode;
	bool mbDeactivateLocalizationMode;

	// Tracking state
	int mTrackingState;
	std::vector<MapPoint*> mTrackedMapPoints;
	std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
	mutable std::mutex mMutexState;
};

System::Pointer System::Create(const Path& vocabularyFile, const Path& settingsFile, Sensor sensor, bool useViewer)
{
	return std::make_unique<SystemImpl>(vocabularyFile, settingsFile, sensor, useViewer);
}

} //namespace ORB_SLAM
