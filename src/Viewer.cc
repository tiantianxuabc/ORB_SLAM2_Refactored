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

#include "Viewer.h"

#include <mutex>

#include <pangolin/pangolin.h>

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"
#include "Usleep.h"

namespace ORB_SLAM2
{

static void GetCurrentOpenGLCameraMatrix(const cv::Mat& Tcw, pangolin::OpenGlMatrix &M)
{
	if (!Tcw.empty())
	{
		cv::Mat1f Rwc(3, 3);
		cv::Mat1f twc(3, 1);
		{
			// unique_lock<mutex> lock(mMutexCamera);
			Rwc = CameraPose::GetR(Tcw).t();
			twc = -Rwc * CameraPose::Gett(Tcw);
		}

		M.m[0] = Rwc(0, 0);
		M.m[1] = Rwc(1, 0);
		M.m[2] = Rwc(2, 0);
		M.m[3] = 0.0;

		M.m[4] = Rwc(0, 1);
		M.m[5] = Rwc(1, 1);
		M.m[6] = Rwc(2, 1);
		M.m[7] = 0.0;

		M.m[8] = Rwc(0, 2);
		M.m[9] = Rwc(1, 2);
		M.m[10] = Rwc(2, 2);
		M.m[11] = 0.0;

		M.m[12] = twc(0);
		M.m[13] = twc(1);
		M.m[14] = twc(2);
		M.m[15] = 1.0;
	}
	else
	{
		M.SetIdentity();
	}
}

static void DrawCurrentCamera(const pangolin::OpenGlMatrix &Twc, float cameraSize = 0.7f, float cameraLineWidth = 3.f)
{
	const float w = cameraSize;
	const float h = 0.75f * w;
	const float z = 0.6f * w;

	glPushMatrix();

#ifdef HAVE_GLES
	glMultMatrixf(Twc.m);
#else
	glMultMatrixd(Twc.m);
#endif

	glLineWidth(cameraLineWidth);
	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(w, h, z);
	glVertex3f(0, 0, 0);
	glVertex3f(w, -h, z);
	glVertex3f(0, 0, 0);
	glVertex3f(-w, -h, z);
	glVertex3f(0, 0, 0);
	glVertex3f(-w, h, z);

	glVertex3f(w, h, z);
	glVertex3f(w, -h, z);

	glVertex3f(-w, h, z);
	glVertex3f(-w, -h, z);

	glVertex3f(-w, h, z);
	glVertex3f(w, h, z);

	glVertex3f(-w, -h, z);
	glVertex3f(w, -h, z);
	glEnd();

	glPopMatrix();
}

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

	cameraSize_ = settings["Viewer.CameraSize"];
	cameraLineWidth_ = settings["Viewer.CameraLineWidth"];

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

		GetCurrentOpenGLCameraMatrix(mapDrawer_->GetCurrentCameraPose(), Twc);

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
		DrawCurrentCamera(Twc, cameraSize_, cameraLineWidth_);
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
			system_->Reset();
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
	unique_lock<mutex> lock(mMutexFinish);
	finishRequested_ = true;
}

bool Viewer::CheckFinish()
{
	unique_lock<mutex> lock(mMutexFinish);
	return finishRequested_;
}

void Viewer::SetFinish()
{
	unique_lock<mutex> lock(mMutexFinish);
	finished_ = true;
}

bool Viewer::isFinished()
{
	unique_lock<mutex> lock(mMutexFinish);
	return finished_;
}

void Viewer::RequestStop()
{
	unique_lock<mutex> lock(mMutexStop);
	if (!stopped_)
		stopRequested_ = true;
}

bool Viewer::isStopped()
{
	unique_lock<mutex> lock(mMutexStop);
	return stopped_;
}

bool Viewer::Stop()
{
	unique_lock<mutex> lock(mMutexStop);
	unique_lock<mutex> lock2(mMutexFinish);

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
	unique_lock<mutex> lock(mMutexStop);
	stopped_ = false;
}

void Viewer::SetCurrentCameraPose(const cv::Mat& Tcw)
{
	mapDrawer_->SetCurrentCameraPose(Tcw);
}

void Viewer::UpdateFrame(Tracking* tracker)
{
	frameDrawer_->Update(tracker);
}

}
