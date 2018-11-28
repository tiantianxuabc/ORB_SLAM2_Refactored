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


#include "Converter.h"

#include <Eigen/Dense>

namespace ORB_SLAM2
{
namespace Converter
{

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
	std::vector<cv::Mat> vDesc;
	vDesc.reserve(Descriptors.rows);
	for (int j = 0; j<Descriptors.rows; j++)
		vDesc.push_back(Descriptors.row(j));

	return vDesc;
}

Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3)
{
	Eigen::Matrix<double, 3, 3> M;

	M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
		cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
		cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

	return M;
}

std::vector<float> toQuaternion(const cv::Mat &M)
{
	Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
	Eigen::Quaterniond q(eigMat);

	std::vector<float> v(4);
	v[0] = static_cast<float>(q.x());
	v[1] = static_cast<float>(q.y());
	v[2] = static_cast<float>(q.z());
	v[3] = static_cast<float>(q.w());

	return v;
}
}

} //namespace ORB_SLAM
