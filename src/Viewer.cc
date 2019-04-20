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

#include "Viewer.h"

#include <mutex>

#include <pangolin/pangolin.h>

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"
#include "Usleep.h"

#define LOCK_MUTEX_STOP()   std::unique_lock<std::mutex> lock1(mutexStop_);
#define LOCK_MUTEX_FINISH() std::unique_lock<std::mutex> lock2(mutexFinish_);

namespace ORB_SLAM2
{

Viewer::Viewer(System* system, Map* map, const std::string& settingsFile)
	: system_(system), finishRequested_(false), finished_(true), stopped_(true), stopRequested_(false)
{
	cv::FileStorage settings(settingsFile, cv::FileStorage::READ);

	float fps = settings["Camera.fps"];
	if (fps < 1) fps = 30;

	waittime_ = static_cast<int>(1e3 / fps);

	viewpointX_ = settings["Viewer.ViewpointX"];
	viewpointY_ = settings["Viewer.ViewpointY"];
	viewpointZ_ = settings["Viewer.ViewpointZ"];
	viewpointF_ = settings["Viewer.ViewpointF"];

	frameDrawer_ = std::make_unique<FrameDrawer>(map);
	mapDrawer_ = std::make_unique<MapDrawer>(map, settingsFile);
}

Viewer::~Viewer()
{
}

void Viewer::Run()
{
	finished_ = false;
	stopped_ = false;

	pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);

	// 3D Mouse handler requires depth testing to be enabled
	glEnable(GL_DEPTH_TEST);

	// Issue specific OpenGl we might need
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
	pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
	pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
	pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
	pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
	pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false, true);
	pangolin::Var<bool> menuReset("menu.Reset", false, false);

	// Define Camera Render Object (for view / scene browsing)
	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(1024, 768, viewpointF_, viewpointF_, 512, 389, 0.1, 1000),
		pangolin::ModelViewLookAt(viewpointX_, viewpointY_, viewpointZ_, 0, 0, 0, 0.0, -1.0, 0.0)
	);

	// Add named OpenGL viewport to window and provide 3D Handler
	pangolin::View& d_cam = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
		.SetHandler(new pangolin::Handler3D(s_cam));

	pangolin::OpenGlMatrix Twc;
	Twc.SetIdentity();

	cv::namedWindow("ORB-SLAM2: Current Frame");

	bool followCamera = true;
	bool localizationMode = false;

	while (true)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		mapDrawer_->GetCurrentOpenGLCameraMatrix(Twc);

		if (menuFollowCamera && followCamera)
		{
			s_cam.Follow(Twc);
		}
		else if (menuFollowCamera && !followCamera)
		{
			s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(viewpointX_, viewpointY_, viewpointZ_, 0, 0, 0, 0.0, -1.0, 0.0));
			s_cam.Follow(Twc);
			followCamera = true;
		}
		else if (!menuFollowCamera && followCamera)
		{
			followCamera = false;
		}

		if (menuLocalizationMode && !localizationMode)
		{
			system_->ActivateLocalizationMode();
			localizationMode = true;
		}
		else if (!menuLocalizationMode && localizationMode)
		{
			system_->DeactivateLocalizationMode();
			localizationMode = false;
		}

		d_cam.Activate(s_cam);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		mapDrawer_->DrawCurrentCamera(Twc);
		if (menuShowKeyFrames || menuShowGraph)
			mapDrawer_->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
		if (menuShowPoints)
			mapDrawer_->DrawMapPoints();

		pangolin::FinishFrame();

		const cv::Mat image = frameDrawer_->DrawFrame();
		cv::imshow("ORB-SLAM2: Current Frame", image);
		cv::waitKey(waittime_);

		if (menuReset)
		{
			menuShowGraph = true;
			menuShowKeyFrames = true;
			menuShowPoints = true;
			menuLocalizationMode = false;
			if (localizationMode)
				system_->DeactivateLocalizationMode();
			localizationMode = false;
			followCamera = true;
			menuFollowCamera = true;
			system_->RequestReset();
			menuReset = false;
		}

		if (Stop())
		{
			while (isStopped())
			{
				usleep(3000);
			}
		}

		if (CheckFinish())
			break;
	}

	pangolin::DestroyWindow("ORB-SLAM2: Map Viewer");

	SetFinish();
}

void Viewer::RequestFinish()
{
	LOCK_MUTEX_FINISH();
	finishRequested_ = true;
}

bool Viewer::CheckFinish() const
{
	LOCK_MUTEX_FINISH();
	return finishRequested_;
}

void Viewer::SetFinish()
{
	LOCK_MUTEX_FINISH();
	finished_ = true;
}

bool Viewer::isFinished() const
{
	LOCK_MUTEX_FINISH();
	return finished_;
}

void Viewer::RequestStop()
{
	LOCK_MUTEX_STOP();
	if (!stopped_)
		stopRequested_ = true;
}

bool Viewer::isStopped() const
{
	LOCK_MUTEX_STOP();
	return stopped_;
}

bool Viewer::Stop()
{
	LOCK_MUTEX_STOP();
	LOCK_MUTEX_FINISH();

	if (finishRequested_)
		return false;
	else if (stopRequested_)
	{
		stopped_ = true;
		stopRequested_ = false;
		return true;
	}

	return false;

}

void Viewer::Release()
{
	LOCK_MUTEX_STOP();
	stopped_ = false;
}

void Viewer::SetCurrentCameraPose(const cv::Mat& Tcw)
{
	mapDrawer_->SetCurrentCameraPose(Tcw);
}

void Viewer::UpdateFrame(const Tracking* tracker, const Frame& currFrame, const cv::Mat& image)
{
	frameDrawer_->Update(tracker, currFrame, image);
}

}
