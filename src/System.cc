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

std::vector<float> toQuaternion(const cv::Mat &M);

} // namespace Converter

#define LOCK_MUTEX_RESET() unique_lock<mutex> lock1(mutexReset_);
#define LOCK_MUTEX_MODE()  unique_lock<mutex> lock2(mutexMode_);
#define LOCK_MUTEX_STATE() unique_lock<mutex> lock3(mMutexState);

class ModeManager
{
public:

	ModeManager(const std::shared_ptr<Tracking>& tracker, const std::shared_ptr<LocalMapping>& localMapper)
		: tracker_(tracker), localMapper_(localMapper), activateLocalizationMode_(false), deactivateLocalizationMode_(false) {}

	void Update()
	{
		LOCK_MUTEX_MODE();
		if (activateLocalizationMode_)
		{
			localMapper_->RequestStop();

			// Wait until Local Mapping has effectively stopped
			while (!localMapper_->isStopped())
			{
				usleep(1000);
			}

			tracker_->InformOnlyTracking(true);
			activateLocalizationMode_ = false;
		}
		if (deactivateLocalizationMode_)
		{
			tracker_->InformOnlyTracking(false);
			localMapper_->Release();
			deactivateLocalizationMode_ = false;
		}
	}

	void ActivateLocalizationMode()
	{
		LOCK_MUTEX_MODE();
		activateLocalizationMode_ = true;
	}

	void DeactivateLocalizationMode()
	{
		LOCK_MUTEX_MODE();
		deactivateLocalizationMode_ = true;
	}

private:
	std::shared_ptr<Tracking> tracker_;
	std::shared_ptr<LocalMapping> localMapper_;
	// Change mode flags
	mutable std::mutex mutexMode_;
	bool activateLocalizationMode_;
	bool deactivateLocalizationMode_;
};

static void GetTracingResults(const Tracking& tracker, int& state, std::vector<MapPoint*>& mappoints, std::vector<cv::KeyPoint>& keypoints)
{
	state = tracker.GetState();
	mappoints = tracker.GetCurrentFrame().mappoints;
	keypoints = tracker.GetCurrentFrame().keypointsUn;
}

class ResetManager
{
public:

	ResetManager(const std::shared_ptr<Tracking>& tracker) : tracker_(tracker), reset_(false) {}

	void Update()
	{
		LOCK_MUTEX_RESET();
		if (reset_)
		{
			tracker_->Reset();
			reset_ = false;
		}
	}

	void Reset()
	{
		LOCK_MUTEX_RESET();
		reset_ = true;
	}

private:
	std::shared_ptr<Tracking> tracker_;
	// Reset flag
	mutable std::mutex mutexReset_;
	bool reset_;
};

class SystemImpl : public System
{
public:

	using Path = System::Path;

	// Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
	SystemImpl(const Path& vocabularyFile, const Path& settingsFile, Sensor sensor, bool useViewer)
		: sensor_(sensor), mpViewer(nullptr)
	{
		// Output welcome message
		cout << endl <<
			"ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
			"This program comes with ABSOLUTELY NO WARRANTY;" << endl <<
			"This is free software, and you are welcome to redistribute it" << endl <<
			"under certain conditions. See LICENSE.txt." << endl << endl;

		cout << "Input sensor was set to: ";

		const char* sensors[3] = { "Monocular", "Stereo", "RGB-D" };
		cout << sensors[sensor_] << endl;

		//Check settings file
		cv::FileStorage settings(settingsFile.c_str(), cv::FileStorage::READ);
		if (!settings.isOpened())
		{
			cerr << "Failed to open settings file at: " << settingsFile << endl;
			exit(-1);
		}

		//Load ORB Vocabulary
		cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

		mpVocabulary = std::make_shared<ORBVocabulary>();
		bool bVocLoad = mpVocabulary->loadFromTextFile(vocabularyFile);
		if (!bVocLoad)
		{
			cerr << "Wrong path to vocabulary. " << endl;
			cerr << "Falied to open at: " << vocabularyFile << endl;
			exit(-1);
		}
		cout << "Vocabulary loaded!" << endl << endl;

		//Create KeyFrame Database
		mpKeyFrameDatabase = std::make_shared<KeyFrameDatabase>(*mpVocabulary);

		//Create the Map
		mpMap = std::make_shared<Map>();

		//Create Drawers. These are used by the Viewer
		mpFrameDrawer = std::make_shared<FrameDrawer>(mpMap.get());
		mpMapDrawer = std::make_shared<MapDrawer>(mpMap.get(), settingsFile);

		//Initialize the Tracking thread
		//(it will live in the main thread of execution, the one that called this constructor)
		tracker_ = Tracking::Create(this, mpVocabulary.get(), mpFrameDrawer.get(), mpMapDrawer.get(),
			mpMap.get(), mpKeyFrameDatabase.get(), settingsFile, sensor_);

		//Initialize the Local Mapping thread and launch
		mpLocalMapper = std::make_shared<LocalMapping>(mpMap.get(), sensor_ == MONOCULAR);
		threads_[THREAD_LOCAL_MAPPING] = thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

		//Initialize the Loop Closing thread and launch
		mpLoopCloser = LoopClosing::Create(mpMap.get(), mpKeyFrameDatabase.get(), mpVocabulary.get(), sensor_ != MONOCULAR);
		threads_[THREAD_LOOP_CLOSING] = thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

		//Initialize the Viewer thread and launch
		if (useViewer)
		{
			mpViewer = std::make_shared<Viewer>(this, mpFrameDrawer.get(), mpMapDrawer.get(), tracker_.get(), settingsFile);
			threads_[THREAD_VIEWER] = thread(&Viewer::Run, mpViewer.get());
			tracker_->SetViewer(mpViewer.get());
		}

		//Set pointers between threads
		tracker_->SetLocalMapper(mpLocalMapper.get());
		tracker_->SetLoopClosing(mpLoopCloser.get());

		mpLocalMapper->SetTracker(tracker_.get());
		mpLocalMapper->SetLoopCloser(mpLoopCloser.get());

		mpLoopCloser->SetTracker(tracker_.get());
		mpLoopCloser->SetLocalMapper(mpLocalMapper.get());

		resetManager_ = std::make_shared<ResetManager>(tracker_);
		modeManager_ = std::make_shared<ModeManager>(tracker_, mpLocalMapper);
	}

	// Proccess the given stereo frame. Images must be synchronized and rectified.
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) override
	{
		if (sensor_ != STEREO)
		{
			cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
			exit(-1);
		}

		// Check mode change
		modeManager_->Update();

		// Check reset
		resetManager_->Update();

		const cv::Mat Tcw = tracker_->GrabImageStereo(imageL, imageR, timestamp);

		LOCK_MUTEX_STATE();
		GetTracingResults(*tracker_, trackingState_, trackedMapPoints_, trackedKeyPointsUn_);

		return Tcw;
	}

	// Process the given rgbd frame. Depthmap must be registered to the RGB frame.
	// Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Input depthmap: Float (CV_32F).
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) override
	{
		if (sensor_ != RGBD)
		{
			cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
			exit(-1);
		}

		// Check mode change
		modeManager_->Update();

		// Check reset
		resetManager_->Update();

		const cv::Mat Tcw = tracker_->GrabImageRGBD(image, depth, timestamp);

		LOCK_MUTEX_STATE();
		GetTracingResults(*tracker_, trackingState_, trackedMapPoints_, trackedKeyPointsUn_);

		return Tcw;
	}

	// Proccess the given monocular frame
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackMonocular(const cv::Mat& image, double timestamp) override
	{
		if (sensor_ != MONOCULAR)
		{
			cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
			exit(-1);
		}

		// Check mode change
		modeManager_->Update();

		// Check reset
		resetManager_->Update();

		const cv::Mat Tcw = tracker_->GrabImageMonocular(image, timestamp);

		LOCK_MUTEX_STATE();
		GetTracingResults(*tracker_, trackingState_, trackedMapPoints_, trackedKeyPointsUn_);

		return Tcw;
	}

	// This stops local mapping thread (map building) and performs only camera tracking.
	void ActivateLocalizationMode() override
	{
		modeManager_->ActivateLocalizationMode();
	}

	// This resumes local mapping thread and performs SLAM again.
	void DeactivateLocalizationMode() override
	{
		modeManager_->DeactivateLocalizationMode();
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
		resetManager_->Reset();
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

		for (auto& t : threads_)
			if (t.joinable()) t.join();
	}

	// Save camera trajectory in the TUM RGB-D dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	void SaveTrajectoryTUM(const Path& filename) const override
	{
		cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
		if (sensor_ == MONOCULAR)
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

		// For each frame we have a reference keyframe, the timestamp and a flag
		// which is true when tracking failed.
		for (const auto& track : tracker_->GetTrajectory())
		{
			if (track.lost)
				continue;

			KeyFrame* pKF = (KeyFrame*)track.referenceKF;

			cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

			// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
			while (pKF->isBad())
			{
				Trw = Trw*pKF->mTcp;
				pKF = pKF->GetParent();
			}

			Trw = Trw*pKF->GetPose()*Two;

			cv::Mat Tcw = track.Tcr * Trw;
			cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
			cv::Mat twc = -Rwc*Tcw.rowRange(0, 3).col(3);

			vector<float> q = Converter::toQuaternion(Rwc);

			f << setprecision(6) << track.timestamp << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
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
		if (sensor_ == MONOCULAR)
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

		// For each frame we have a reference keyframe, the timestamp and a flag
		// which is true when tracking failed.
		for (const auto& track : tracker_->GetTrajectory())
		{
			ORB_SLAM2::KeyFrame* pKF = (KeyFrame*)track.referenceKF;

			cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

			while (pKF->isBad())
			{
				//  cout << "bad parent" << endl;
				Trw = Trw*pKF->mTcp;
				pKF = pKF->GetParent();
			}

			Trw = Trw*pKF->GetPose()*Two;

			cv::Mat Tcw = track.Tcr * Trw;
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
		return trackingState_;
	}

	vector<MapPoint*> GetTrackedMapPoints() const override
	{
		LOCK_MUTEX_STATE();
		return trackedMapPoints_;
	}

	vector<cv::KeyPoint> GetTrackedKeyPointsUn() const override
	{
		LOCK_MUTEX_STATE();
		return trackedKeyPointsUn_;
	}

private:

	// Input sensor
	Sensor sensor_;

	// ORB vocabulary used for place recognition and feature matching.
	std::shared_ptr<ORBVocabulary> mpVocabulary;

	// KeyFrame database for place recognition (relocalization and loop detection).
	std::shared_ptr<KeyFrameDatabase> mpKeyFrameDatabase;

	// Map structure that stores the pointers to all KeyFrames and MapPoints.
	std::shared_ptr<Map> mpMap;

	// Tracker. It receives a frame and computes the associated camera pose.
	// It also decides when to insert a new keyframe, create some new MapPoints and
	// performs relocalization if tracking fails.
	std::shared_ptr<Tracking> tracker_;

	// Local Mapper. It manages the local map and performs local bundle adjustment.
	std::shared_ptr<LocalMapping> mpLocalMapper;

	// Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
	// a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
	std::shared_ptr<LoopClosing> mpLoopCloser;

	// The viewer draws the map and the current camera pose. It uses Pangolin.
	std::shared_ptr<Viewer> mpViewer;

	std::shared_ptr<FrameDrawer> mpFrameDrawer;
	std::shared_ptr<MapDrawer> mpMapDrawer;

	// System threads: Local Mapping, Loop Closing, Viewer.
	// The Tracking thread "lives" in the main execution thread that creates the System object.
	enum { THREAD_LOCAL_MAPPING, THREAD_LOOP_CLOSING, THREAD_VIEWER, NUM_THREADS };
	std::thread threads_[NUM_THREADS];

	// Reset flag
	std::shared_ptr<ResetManager> resetManager_;

	// Change mode flags
	std::shared_ptr<ModeManager> modeManager_;

	// Tracking state
	int trackingState_;
	std::vector<MapPoint*> trackedMapPoints_;
	std::vector<cv::KeyPoint> trackedKeyPointsUn_;
	mutable std::mutex mMutexState;
};

System::Pointer System::Create(const Path& vocabularyFile, const Path& settingsFile, Sensor sensor, bool useViewer)
{
	return std::make_unique<SystemImpl>(vocabularyFile, settingsFile, sensor, useViewer);
}

} //namespace ORB_SLAM
