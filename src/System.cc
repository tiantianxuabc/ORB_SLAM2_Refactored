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

#include "Frame.h"
#include "KeyFrame.h"
#include "Tracking.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "Usleep.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2
{

#define LOCK_MUTEX_RESET() unique_lock<mutex> lock1(mutexReset_);
#define LOCK_MUTEX_MODE()  unique_lock<mutex> lock2(mutexMode_);
#define LOCK_MUTEX_STATE() unique_lock<mutex> lock3(mutexState_);

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
	std::cout << std::endl << "Camera Parameters: " << std::endl;
	std::cout << "- fx: " << camera.fx << std::endl;
	std::cout << "- fy: " << camera.fy << std::endl;
	std::cout << "- cx: " << camera.cx << std::endl;
	std::cout << "- cy: " << camera.cy << std::endl;
	std::cout << "- k1: " << distCoeffs(0) << std::endl;
	std::cout << "- k2: " << distCoeffs(1) << std::endl;
	if (distCoeffs.rows == 5)
		std::cout << "- k3: " << distCoeffs(4) << std::endl;
	std::cout << "- p1: " << distCoeffs(2) << std::endl;
	std::cout << "- p2: " << distCoeffs(3) << std::endl;
	std::cout << "- fps: " << fps << std::endl;

	std::cout << "- color order: " << (rgb ? "RGB" : "BGR") << " (ignored if grayscale)" << std::endl;

	std::cout << std::endl << "ORB Extractor Parameters: " << std::endl;
	std::cout << "- Number of Features: " << param.nfeatures << std::endl;
	std::cout << "- Scale Levels: " << param.nlevels << std::endl;
	std::cout << "- Scale Factor: " << param.scaleFactor << std::endl;
	std::cout << "- Initial Fast Threshold: " << param.iniThFAST << std::endl;
	std::cout << "- Minimum Fast Threshold: " << param.minThFAST << std::endl;

	if (sensor == System::STEREO || sensor == System::RGBD)
		std::cout << std::endl << "Depth Threshold (Close/Far Points): " << thDepth << std::endl;
}

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

static void GetScalePyramidInfo(const ORBextractor* extractor, ScalePyramidInfo& pyramid)
{
	pyramid.nlevels = extractor->GetLevels();
	pyramid.scaleFactor = extractor->GetScaleFactor();
	pyramid.logScaleFactor = log(pyramid.scaleFactor);
	pyramid.scaleFactors = extractor->GetScaleFactors();
	pyramid.invScaleFactors = extractor->GetInverseScaleFactors();
	pyramid.sigmaSq = extractor->GetScaleSigmaSquares();
	pyramid.invSigmaSq = extractor->GetInverseScaleSigmaSquares();
}

// Undistort keypoints given OpenCV distortion parameters.
// Only for the RGB-D case. Stereo must be already rectified!
// (called in the constructor).
static void UndistortKeyPoints(const KeyPoints& src, KeyPoints& dst, const cv::Mat& K, const cv::Mat1f& distCoeffs)
{
	if (distCoeffs(0) == 0.f)
	{
		dst = src;
		return;
	}

	std::vector<cv::Point2f> points(src.size());
	for (size_t i = 0; i < src.size(); i++)
		points[i] = src[i].pt;

	cv::undistortPoints(points, points, K, distCoeffs, cv::Mat(), K);

	dst.resize(src.size());
	for (size_t i = 0; i < src.size(); i++)
	{
		cv::KeyPoint keypoint = src[i];
		keypoint.pt = points[i];
		dst[i] = keypoint;
	}
}

// Computes image bounds for the undistorted image (called in the constructor).
static ImageBounds ComputeImageBounds(const cv::Mat& image, const cv::Mat& K, const cv::Mat1f& distCoeffs)
{
	const float h = static_cast<float>(image.rows);
	const float w = static_cast<float>(image.cols);

	if (distCoeffs(0) == 0.f)
		return ImageBounds(0.f, w, 0.f, h);

	std::vector<cv::Point2f> corners = { { 0, 0 },{ w, 0 },{ 0, h },{ w, h } };
	cv::undistortPoints(corners, corners, K, distCoeffs, cv::Mat(), K);

	ImageBounds imageBounds;
	imageBounds.minx = std::min(corners[0].x, corners[2].x);
	imageBounds.maxx = std::max(corners[1].x, corners[3].x);
	imageBounds.miny = std::min(corners[0].y, corners[1].y);
	imageBounds.maxy = std::max(corners[2].y, corners[3].y);
	return imageBounds;
}

// Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
static void ComputeStereoFromRGBD(const KeyPoints& keypoints, const KeyPoints& keypointsUn, const cv::Mat& depthImage,
	const CameraParams& camera, std::vector<float>& uright, std::vector<float>& depth)
{
	const int nkeypoints = static_cast<int>(keypoints.size());

	uright.assign(nkeypoints, -1.f);
	depth.assign(nkeypoints, -1.f);

	for (int i = 0; i < nkeypoints; i++)
	{
		const cv::KeyPoint& keypoint = keypoints[i];
		const cv::KeyPoint& keypointUn = keypointsUn[i];

		const int v = static_cast<int>(keypoint.pt.y);
		const int u = static_cast<int>(keypoint.pt.x);
		const float d = depthImage.at<float>(v, u);
		if (d > 0)
		{
			depth[i] = d;
			uright[i] = keypointUn.pt.x - camera.bf / d;
		}
	}
}

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

static void GetTracingResults(const Tracking& tracker, const Frame& currFrame,
	int& state, std::vector<MapPoint*>& mappoints, std::vector<cv::KeyPoint>& keypoints)
{
	state = tracker.GetState();
	mappoints = currFrame.mappoints;
	keypoints = currFrame.keypointsUn;
}

class ResetManager
{
public:

	ResetManager(System* system) : system_(system), reset_(false) {}

	void Update()
	{
		LOCK_MUTEX_RESET();
		if (reset_)
		{
			system_->Reset();
			reset_ = false;
		}
	}

	void Reset()
	{
		LOCK_MUTEX_RESET();
		reset_ = true;
	}

private:
	System* system_;
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
		: sensor_(sensor), viewer_(nullptr)
	{
		// Output welcome message
		std::cout << std::endl <<
			"ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << std::endl <<
			"This program comes with ABSOLUTELY NO WARRANTY;" << std::endl <<
			"This is free software, and you are welcome to redistribute it" << std::endl <<
			"under certain conditions. See LICENSE.txt." << std::endl << std::endl;

		std::cout << "Input sensor was set to: ";

		const char* sensors[3] = { "Monocular", "Stereo", "RGB-D" };
		std::cout << sensors[sensor_] << std::endl;

		//Check settings file
		cv::FileStorage settings(settingsFile.c_str(), cv::FileStorage::READ);
		if (!settings.isOpened())
		{
			cerr << "Failed to open settings file at: " << settingsFile << std::endl;
			std::exit(-1);
		}

		//Load ORB Vocabulary
		std::cout << std::endl << "Loading ORB Vocabulary. This could take a while..." << std::endl;

		if (!voc_.loadFromTextFile(vocabularyFile))
		{
			cerr << "Wrong path to vocabulary. " << std::endl;
			cerr << "Falied to open at: " << vocabularyFile << std::endl;
			std::exit(-1);
		}
		std::cout << "Vocabulary loaded!" << std::endl << std::endl;

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
		const float thDepth = camera_.baseline * static_cast<float>(settings["ThDepth"]);
		thDepth_ = thDepth;

		// Load depth factor
		depthFactor_ = sensor == System::RGBD ? ReadDepthFactor(settings) : 1.f;

		// Print settings
		PrintSettings(camera_, distCoeffs_, fps, RGB_, extractorParams, thDepth, sensor);

		// Initialize ORB extractors
		extractorL_ = std::make_unique<ORBextractor>(extractorParams);
		extractorR_ = std::make_unique<ORBextractor>(extractorParams);

		if (sensor == System::MONOCULAR)
		{
			extractorParams.nfeatures *= 2;
			extractorIni_ = std::make_unique<ORBextractor>(extractorParams);
		}

		// Scale Level Info
		GetScalePyramidInfo(extractorL_.get(), pyramid_);
		
		//Create KeyFrame Database
		keyFrameDB_ = std::make_shared<KeyFrameDatabase>(voc_);

		//Initialize the Tracking thread
		//(it will live in the main thread of execution, the one that called this constructor)
		const Tracking::Parameters trackParams(minFrames, maxFrames, thDepth);
		tracker_ = Tracking::Create(this, &voc_, &map_, keyFrameDB_.get(), sensor_, trackParams);

		//Initialize the Local Mapping thread and launch
		localMapper_ = LocalMapping::Create(&map_, sensor_ == MONOCULAR, thDepth);
		threads_[THREAD_LOCAL_MAPPING] = std::thread(&ORB_SLAM2::LocalMapping::Run, localMapper_);

		//Initialize the Loop Closing thread and launch
		loopCloser_ = LoopClosing::Create(&map_, keyFrameDB_.get(), &voc_, sensor_ != MONOCULAR);
		threads_[THREAD_LOOP_CLOSING] = std::thread(&ORB_SLAM2::LoopClosing::Run, loopCloser_);

		//Initialize the Viewer thread and launch
		if (useViewer)
		{
			viewer_ = std::make_shared<Viewer>(this, &map_, settingsFile);
			threads_[THREAD_VIEWER] = std::thread(&Viewer::Run, viewer_);
		}

		//Set pointers between threads
		tracker_->SetLocalMapper(localMapper_);
		tracker_->SetLoopClosing(loopCloser_);

		localMapper_->SetTracker(tracker_);
		localMapper_->SetLoopCloser(loopCloser_);

		loopCloser_->SetTracker(tracker_);
		loopCloser_->SetLocalMapper(localMapper_);

		resetManager_ = std::make_shared<ResetManager>(this);
		modeManager_ = std::make_shared<ModeManager>(tracker_, localMapper_);
	}

	// Proccess the given stereo frame. Images must be synchronized and rectified.
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackStereo(const cv::Mat& imageL, const cv::Mat& imageR, double timestamp) override
	{
		if (sensor_ != STEREO)
		{
			cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << std::endl;
			std::exit(-1);
		}

		// Check mode change
		modeManager_->Update();

		// Check reset
		resetManager_->Update();

		// Color conversion
		ConvertToGray(imageL, imageL_, RGB_);
		ConvertToGray(imageR, imageR_, RGB_);

		// ORB extraction
		std::thread threadL([&]() { extractorL_->Extract(imageL_, keypointsL_, descriptorsL_); });
		std::thread threadR([&]() { extractorR_->Extract(imageR_, keypointsR_, descriptorsR_); });
		threadL.join();
		threadR.join();

		// Undistortion
		UndistortKeyPoints(keypointsL_, keypointsUn_, camera_.Mat(), distCoeffs_);

		// Stereo matching
		ComputeStereoMatches(
			keypointsL_, descriptorsL_, extractorL_->GetImagePyramid(),
			keypointsR_, descriptorsR_, extractorR_->GetImagePyramid(),
			pyramid_.scaleFactors, pyramid_.invScaleFactors, camera_, uright_, depth_);

		// Computes image bounds for the undistorted image
		if (imageBounds_.Empty())
			imageBounds_ = ComputeImageBounds(imageL_, camera_.Mat(), distCoeffs_);

		// Create frame
		currFrame_ = Frame(&voc_, timestamp, camera_, thDepth_, keypointsL_, keypointsUn_,
			uright_, depth_, descriptorsL_, pyramid_, imageBounds_);

		// Update tracker
		const cv::Mat Tcw = tracker_->Update(currFrame_);

		if (viewer_)
		{
			viewer_->UpdateFrame(tracker_.get(), currFrame_, imageL_);
			if (tracker_->GetState() == Tracking::STATE_OK)
				viewer_->SetCurrentCameraPose(Tcw);
		}

		LOCK_MUTEX_STATE();
		GetTracingResults(*tracker_, currFrame_, trackingState_, trackedMapPoints_, trackedKeyPointsUn_);

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
			cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << std::endl;
			std::exit(-1);
		}

		// Check mode change
		modeManager_->Update();

		// Check reset
		resetManager_->Update();

		// Color conversion
		ConvertToGray(image, imageL_, RGB_);

		// ORB extraction
		extractorL_->Extract(imageL_, keypointsL_, descriptorsL_);

		// Undistortion
		UndistortKeyPoints(keypointsL_, keypointsUn_, camera_.Mat(), distCoeffs_);

		// Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
		depth.convertTo(depthMap_, CV_32F, depthFactor_);
		ComputeStereoFromRGBD(keypointsL_, keypointsUn_, depthMap_, camera_, uright_, depth_);

		// Computes image bounds for the undistorted image
		if (imageBounds_.Empty())
			imageBounds_ = ComputeImageBounds(imageL_, camera_.Mat(), distCoeffs_);

		// Create frame
		currFrame_ = Frame(&voc_, timestamp, camera_, thDepth_, keypointsL_, keypointsUn_,
			uright_, depth_, descriptorsL_, pyramid_, imageBounds_);

		// Update tracker
		const cv::Mat Tcw = tracker_->Update(currFrame_);;

		if (viewer_)
		{
			viewer_->UpdateFrame(tracker_.get(), currFrame_, imageL_);
			if (tracker_->GetState() == Tracking::STATE_OK)
				viewer_->SetCurrentCameraPose(Tcw);
		}

		LOCK_MUTEX_STATE();
		GetTracingResults(*tracker_, currFrame_, trackingState_, trackedMapPoints_, trackedKeyPointsUn_);

		return Tcw;
	}

	// Proccess the given monocular frame
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	cv::Mat TrackMonocular(const cv::Mat& image, double timestamp) override
	{
		if (sensor_ != MONOCULAR)
		{
			cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << std::endl;
			std::exit(-1);
		}

		// Check mode change
		modeManager_->Update();

		// Check reset
		resetManager_->Update();

		// Color conversion
		ConvertToGray(image, imageL_, RGB_);

		const int state = tracker_->GetState();
		const bool init = state == Tracking::STATE_NOT_INITIALIZED || state == Tracking::STATE_NO_IMAGES;
		auto& extractor = init ? extractorIni_ : extractorL_;

		// ORB extraction
		extractor->Extract(imageL_, keypointsL_, descriptorsL_);

		// Undistortion
		UndistortKeyPoints(keypointsL_, keypointsUn_, camera_.Mat(), distCoeffs_);

		// Create frame
		currFrame_ = Frame(&voc_, timestamp, camera_, thDepth_, keypointsL_, keypointsUn_,
			descriptorsL_, pyramid_, imageBounds_);

		// Update tracker
		const cv::Mat Tcw = tracker_->Update(currFrame_);;

		if (viewer_)
		{
			viewer_->UpdateFrame(tracker_.get(), currFrame_, imageL_);
			if (tracker_->GetState() == Tracking::STATE_OK)
				viewer_->SetCurrentCameraPose(Tcw);
		}

		LOCK_MUTEX_STATE();
		GetTracingResults(*tracker_, currFrame_, trackingState_, trackedMapPoints_, trackedKeyPointsUn_);

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
		const int curn = map_.GetLastBigChangeIdx();
		if (n < curn)
		{
			n = curn;
			return true;
		}
		else
			return false;
	}

	// Reset the system (clear map)
	void RequestReset() override
	{
		resetManager_->Reset();
	}

	void Reset() override
	{
		std::cout << "System Reseting" << std::endl;
		if (viewer_)
		{
			viewer_->RequestStop();
			while (!viewer_->isStopped())
				usleep(3000);
		}

		// Reset Tracking
		tracker_->Reset();

		// Reset Local Mapping
		std::cout << "Reseting Local Mapper...";
		localMapper_->RequestReset();
		std::cout << " done" << std::endl;

		// Reset Loop Closing
		std::cout << "Reseting Loop Closing...";
		loopCloser_->RequestReset();
		std::cout << " done" << std::endl;

		// Clear BoW Database
		std::cout << "Reseting Database...";
		keyFrameDB_->clear();
		std::cout << " done" << std::endl;

		// Clear Map (this erase MapPoints and KeyFrames)
		map_.Clear();

		KeyFrame::nextId = 0;
		Frame::nextId = 0;

		if (viewer_)
			viewer_->Release();
	}

	// All threads will be requested to finish.
	// It waits until all threads have finished.
	// This function must be called before saving the trajectory.
	void Shutdown() override
	{
		localMapper_->RequestFinish();
		loopCloser_->RequestFinish();
		if (viewer_)
		{
			viewer_->RequestFinish();
			while (!viewer_->isFinished())
				usleep(5000);
		}

		// Wait until all thread have effectively stopped
		while (!localMapper_->isFinished() || !loopCloser_->isFinished() || loopCloser_->isRunningGBA())
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
		std::cout << std::endl << "Saving camera trajectory to " << filename << " ..." << std::endl;
		if (sensor_ == MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << std::endl;
			return;
		}

		std::vector<KeyFrame*> keyframes = map_.GetAllKeyFrames();
		std::sort(std::begin(keyframes), std::end(keyframes),
			[](const KeyFrame* lhs, const KeyFrame* rhs) { return lhs->id < rhs->id; });

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		const CameraPose Two = keyframes.front()->GetPose().Inverse();

		std::ofstream ofs(filename);
		ofs << std::fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe, the timestamp and a flag
		// which is true when tracking failed.
		for (const auto& track : tracker_->GetTrajectory())
		{
			if (track.lost)
				continue;

			KeyFrame* keyframe = (KeyFrame*)track.referenceKF;

			CameraPose Trw = CameraPose::Origin();

			// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
			while (keyframe->isBad())
			{
				Trw = Trw * keyframe->Tcp;
				keyframe = keyframe->GetParent();
			}

			Trw = Trw * keyframe->GetPose() * Two;

			const CameraPose Tcw = track.Tcr * Trw;
			const CameraPose Twc = Tcw.Inverse();
			const auto Rwc = Twc.R();
			const auto twc = Twc.t();

			const auto q = Converter::toQuaternion(Rwc);

			ofs << std::setprecision(6) << track.timestamp << " ";
			ofs << std::setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " ";
			ofs << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
		}

		std::cout << std::endl << "trajectory saved!" << std::endl;
	}

	// Save keyframe poses in the TUM RGB-D dataset format.
	// This method works for all sensor input.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	void SaveKeyFrameTrajectoryTUM(const Path& filename) const override
	{
		std::cout << std::endl << "Saving keyframe trajectory to " << filename << " ..." << std::endl;

		std::vector<KeyFrame*> keyframes = map_.GetAllKeyFrames();
		std::sort(std::begin(keyframes), std::end(keyframes),
			[](const KeyFrame* lhs, const KeyFrame* rhs) { return lhs->id < rhs->id; });

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		//cv::Mat Two = vpKFs[0]->GetPoseInverse();

		std::ofstream ofs(filename);
		ofs << std::fixed;

		for (size_t i = 0; i < keyframes.size(); i++)
		{
			KeyFrame* keyframe = keyframes[i];

			if (keyframe->isBad())
				continue;

			const auto R = keyframe->GetPose().InvR();
			const auto t = keyframe->GetCameraCenter();
			const auto q = Converter::toQuaternion(R);
			ofs << std::setprecision(6) << keyframe->timestamp << " ";
			ofs << std::setprecision(7) << t(0) << " " << t(1) << " " << t(2) << " ";
			ofs << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;

		}

		std::cout << std::endl << "trajectory saved!" << std::endl;
	}

	// Save camera trajectory in the KITTI dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
	void SaveTrajectoryKITTI(const Path& filename) const override
	{
		std::cout << std::endl << "Saving camera trajectory to " << filename << " ..." << std::endl;
		if (sensor_ == MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << std::endl;
			return;
		}

		std::vector<KeyFrame*> keyframes = map_.GetAllKeyFrames();
		std::sort(std::begin(keyframes), std::end(keyframes),
			[](const KeyFrame* lhs, const KeyFrame* rhs) { return lhs->id < rhs->id; });

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		const CameraPose Two = keyframes.front()->GetPose().Inverse();

		std::ofstream ofs(filename);
		ofs << std::fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe, the timestamp and a flag
		// which is true when tracking failed.
		for (const auto& track : tracker_->GetTrajectory())
		{
			KeyFrame* keyframe = (KeyFrame*)track.referenceKF;

			CameraPose Trw = CameraPose::Origin();

			while (keyframe->isBad())
			{
				//  std::cout << "bad parent" << std::endl;
				Trw = Trw * keyframe->Tcp;
				keyframe = keyframe->GetParent();
			}

			Trw = Trw * keyframe->GetPose() * Two;

			const CameraPose Tcw = track.Tcr * Trw;
			const CameraPose Twc = Tcw.Inverse();
			const auto Rwc = Twc.R();
			const auto twc = Twc.t();

			ofs << std::setprecision(9) <<
				Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " " <<
				Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " " <<
				Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << std::endl;
		}

		std::cout << std::endl << "trajectory saved!" << std::endl;
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

	void ChangeCalibration(const string& settingsFile) override
	{
		cv::FileStorage settings(settingsFile, cv::FileStorage::READ);
		camera_ = ReadCameraParams(settings);
		distCoeffs_ = ReadDistCoeffs(settings);
		imageBounds_ = ImageBounds();
	}

private:

	// Input sensor
	Sensor sensor_;

	// ORB vocabulary used for place recognition and feature matching.
	ORBVocabulary voc_;

	// KeyFrame database for place recognition (relocalization and loop detection).
	std::shared_ptr<KeyFrameDatabase> keyFrameDB_;

	// Map structure that stores the pointers to all KeyFrames and MapPoints.
	Map map_;

	// Tracker. It receives a frame and computes the associated camera pose.
	// It also decides when to insert a new keyframe, create some new MapPoints and
	// performs relocalization if tracking fails.
	std::shared_ptr<Tracking> tracker_;

	// Local Mapper. It manages the local map and performs local bundle adjustment.
	std::shared_ptr<LocalMapping> localMapper_;

	// Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
	// a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
	std::shared_ptr<LoopClosing> loopCloser_;

	// The viewer draws the map and the current camera pose. It uses Pangolin.
	std::shared_ptr<Viewer> viewer_;

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
	mutable std::mutex mutexState_;

	// Current Frame
	Frame currFrame_;
	cv::Mat imageL_;
	cv::Mat imageR_;
	cv::Mat depthMap_;

	KeyPoints keypointsL_, keypointsR_, keypointsUn_;
	std::vector<float> uright_, depth_;
	cv::Mat descriptorsL_, descriptorsR_;
	ImageBounds imageBounds_;

	// ORB
	std::unique_ptr<ORBextractor> extractorL_;
	std::unique_ptr<ORBextractor> extractorR_;
	std::unique_ptr<ORBextractor> extractorIni_;

	// Scale Level Info
	ScalePyramidInfo pyramid_;

	// Calibration matrix
	CameraParams camera_;
	cv::Mat1f distCoeffs_;

	// For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
	float depthFactor_;

	// Color order (true RGB, false BGR, ignored if grayscale)
	bool RGB_;

	float thDepth_;
};

System::Pointer System::Create(const Path& vocabularyFile, const Path& settingsFile, Sensor sensor, bool useViewer)
{
	return std::make_unique<SystemImpl>(vocabularyFile, settingsFile, sensor, useViewer);
}

} //namespace ORB_SLAM
