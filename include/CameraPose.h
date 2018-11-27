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

#ifndef CAMERAPOSE_H
#define CAMERAPOSE_H

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

static inline cv::Mat GetR(const cv::Mat& T) { return T(cv::Range(0, 3), cv::Range(0, 3)); }
static inline cv::Mat Gett(const cv::Mat& T) { return T(cv::Range(0, 3), cv::Range(3, 4)); }

class CameraPose
{
public:

	using Mat33 = cv::Matx33f;
	using Mat31 = cv::Matx31f;

	static CameraPose Origin() { return CameraPose(Mat33::eye(), Mat31::zeros()); }
	CameraPose() : R_(Mat33::zeros()), t_(Mat31::zeros()), empty_(true) {}
	CameraPose(const Mat33& R, const Mat31& t) : R_(R), t_(t), empty_(false) {}

	const Mat33& R() const { return R_; }
	const Mat31& t() const { return t_; }
	Mat33& R() { return R_; }
	Mat31& t() { return t_; }
	Mat33 InvR() const { return R_.t(); }
	Mat31 Invt() const { return -R_.t() * t_; }
	bool Empty() const { return empty_; }

	CameraPose Inverse() const { return CameraPose(InvR(), Invt()); }

	CameraPose& operator*=(const CameraPose& T2)
	{
		CameraPose& T1(*this);
		T1.t_ = T1.R_ * T2.t_ + T1.t_;
		T1.R_ = T1.R_ * T2.R_;
		return *this;
	}

	CameraPose operator*(const CameraPose& T2) const
	{
		CameraPose T1(*this);
		T1 *= T2;
		return T1;
	}

	// cv::Mat1f => CameraPose
	CameraPose(const cv::Mat1f& m) : empty_(false)
	{
		CV_Assert(m.rows == 4 && m.cols == 4);
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R_(i, j) = m(i, j);
		for (int i = 0; i < 3; i++) t_(i) = m(i, 3);
	}

	// CameraPose => cv::Mat1f
	operator cv::Mat1f() const
	{
		cv::Mat1f m = cv::Mat1f::eye(4, 4);
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) m(i, j) = R_(i, j);
		for (int i = 0; i < 3; i++) m(i, 3) = t_(i);
		return m;
	}

	cv::Mat1f Mat() const { return cv::Mat1f(*this); }

protected:

	cv::Matx33f R_;
	cv::Matx31f t_;
	bool empty_;
};

} // namespace ORB_SLAM2

#endif // !CAMERAPOSE_H
