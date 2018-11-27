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

#ifndef POINT_H
#define POINT_H

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

using Point2D = cv::Point2f;
using Point3D = cv::Matx31f;
using Vec3D = cv::Matx31f;

static Vec3D Normalized(const Vec3D& v)
{
	const double invn = 1. / cv::norm(v);
	return invn * v;
}

} // namespace ORB_SLAM2

#endif // !POINT_H
