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

#ifndef SIM3_H
#define SIM3_H

#include "CameraPose.h"
#include "Point.h"

namespace ORB_SLAM2
{

class Sim3 : public CameraPose
{
public:

	using CameraPose::CameraPose;

	Sim3() : CameraPose(), s_(0) {}
	Sim3(const Mat33& R, const Mat31& t, float s = 1) : CameraPose(R, t), s_(s) {}
	Sim3(const CameraPose& T) : CameraPose(T), s_(1) {}
	float Scale() const { return s_; }
	Point3D Map(const Point3D& x) const { return s_ * R_ * x + t_; }

	Mat33 InvR() const { return R_.t(); }
	Mat31 Invt() const { return -(1.f / s_) * R_.t() * t_; }
	float Invs() const { return 1.f / s_; }

	Sim3 Inverse() const { return Sim3(InvR(), Invt(), Invs()); }

	Sim3& operator*=(const Sim3& S2)
	{
		Sim3& S1(*this);

		S1.t_ = S1.s_ * S1.R_ * S2.t_ + S1.t_;
		S1.R_ = S1.R_ * S2.R_;
		S1.s_ = S1.s_ * S2.s_;
		return *this;
	}

	Sim3 operator*(const Sim3& T2) const
	{
		Sim3 T1(*this);
		T1 *= T2;
		return T1;
	}

private:
	float s_;
};

} // namespace ORB_SLAM2

#endif // !SIM3_H
