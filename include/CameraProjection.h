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

#ifndef CAMERAPROJECTION_H
#define CAMERAPROJECTION_H

#include "Point.h"
#include "CameraPose.h"
#include "CameraParameters.h"

namespace ORB_SLAM2
{

struct CameraProjection
{
	CameraProjection(const CameraPose& pose, const CameraParams& camera)
	{
		Rcw = pose.R();
		tcw = pose.t();
		fu = camera.fx;
		fv = camera.fy;
		u0 = camera.cx;
		v0 = camera.cy;
		bf = camera.bf;
	}

	inline Point3D WorldToCamera(const Point3D& Xw) const
	{
		return Rcw * Xw + tcw;
	}

	inline Point2D CameraToImage(const Point3D& Xc) const
	{
		const float invZ = 1.f / Xc(2);
		const float u = invZ * fu * Xc(0) + u0;
		const float v = invZ * fv * Xc(1) + v0;
		return Point2D(u, v);
	}

	inline Point2D WorldToImage(const Point3D& Xw) const
	{
		return CameraToImage(WorldToCamera(Xw));
	}

	inline float DepthToDisparity(float Z) const
	{
		return bf / Z;
	}

	cv::Matx33f Rcw;
	cv::Matx31f tcw;
	float fu, fv, u0, v0, bf;
};

struct CameraUnProjection
{
	CameraUnProjection(const CameraPose& pose, const CameraParams& camera)
	{
		Rwc = pose.InvR();
		twc = pose.Invt();
		invfu = 1.f / camera.fx;
		invfv = 1.f / camera.fy;
		u0 = camera.cx;
		v0 = camera.cy;
		bf = camera.bf;
	}

	inline Point3D uvZToCamera(float u, float v, float Z) const
	{
		const float X = invfu * (u - u0) * Z;
		const float Y = invfv * (v - v0) * Z;
		return Point3D(X, Y, Z);
	}

	inline Point3D CameraToWorld(const Point3D& Xc) const
	{
		return Rwc * Xc + twc;
	}

	inline Point3D uvZToWorld(float u, float v, float Z) const
	{
		return CameraToWorld(uvZToCamera(u, v, Z));
	}

	cv::Matx33f Rwc;
	cv::Matx31f twc;
	float invfu, invfv, u0, v0, bf;
};

} // namespace ORB_SLAM2

#endif // !CAMERAPROJECTION_H
