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


#include "Sim3Solver.h"

#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <Thirdparty/DBoW2/DUtils/Random.h>

#include "KeyFrame.h"
#include "MapPoint.h"

namespace ORB_SLAM2
{

using Sim3 = Sim3Solver::Sim3;

static void CopySim3(const Sim3& src, Sim3& dst)
{
	dst.T = src.T.clone();
	dst.R = src.R.clone();
	dst.t = src.t.clone();
	dst.scale = src.scale;
}

static void ComputeCentroid(const cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
	cv::reduce(P, C, 1, CV_REDUCE_SUM);
	C = C / P.cols;

	for (int i = 0; i < P.cols; i++)
	{
		Pr.col(i) = P.col(i) - C;
	}
}

static void ComputeRotation(const cv::Mat1f& M, cv::Mat& R)
{
	// Step 3: Compute N matrix

	float N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

	N11 = M(0, 0) + M(1, 1) + M(2, 2);
	N12 = M(1, 2) - M(2, 1);
	N13 = M(2, 0) - M(0, 2);
	N14 = M(0, 1) - M(1, 0);
	N22 = M(0, 0) - M(1, 1) - M(2, 2);
	N23 = M(0, 1) + M(1, 0);
	N24 = M(2, 0) + M(0, 2);
	N33 = -M(0, 0) + M(1, 1) - M(2, 2);
	N34 = M(1, 2) + M(2, 1);
	N44 = -M(0, 0) - M(1, 1) + M(2, 2);

	cv::Mat1f N = (cv::Mat1f(4, 4) <<
		N11, N12, N13, N14,
		N12, N22, N23, N24,
		N13, N23, N33, N34,
		N14, N24, N34, N44);

	// Step 4: Eigenvector of the highest eigenvalue

	cv::Mat eval, evec;

	cv::eigen(N, eval, evec); //evec[0] is the quaternion of the desired rotation

	cv::Mat vec(1, 3, evec.type());
	(evec.row(0).colRange(1, 4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

	// Rotation angle. sin is the norm of the imaginary part, cos is the real part
	double ang = atan2(norm(vec), evec.at<float>(0, 0));

	vec = 2 * ang*vec / norm(vec); //Angle-axis representation. quaternion angle is the half

	R.create(3, 3, CV_32F);

	cv::Rodrigues(vec, R); // computes the rotation matrix from angle-axis
}

static void ComputeSim3(const cv::Mat& P1, const cv::Mat& P2, Sim3& S12, Sim3& S21, bool fixScale)
{
	// Custom implementation of:
	// Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

	// Step 1: Centroid and relative coordinates

	cv::Mat Pr1(P1.size(), P1.type()); // Relative coordinates to centroid (set 1)
	cv::Mat Pr2(P2.size(), P2.type()); // Relative coordinates to centroid (set 2)
	cv::Mat O1(3, 1, Pr1.type()); // Centroid of P1
	cv::Mat O2(3, 1, Pr2.type()); // Centroid of P2

	ComputeCentroid(P1, Pr1, O1);
	ComputeCentroid(P2, Pr2, O2);

	// Step 2: Compute M matrix

	cv::Mat M = Pr2*Pr1.t();

	// Step 3 ~ Step 4
	ComputeRotation(M, S12.R);

	// Step 5: Rotate set 2

	cv::Mat P3 = S12.R * Pr2;

	// Step 6: Scale
	S12.scale = 1.f;
	if (!fixScale)
	{
		auto Squared = [](float x) { return x * x; };
		const double nom = Pr1.dot(P3);
		double den = 0;
		for (int i = 0; i < P3.rows; i++)
			for (int j = 0; j < P3.cols; j++)
				den += Squared(P3.at<float>(i, j) * P3.at<float>(i, j));
		S12.scale = static_cast<float>(nom / den);
	}

	// Step 7: Translation

	S12.t.create(1, 3, P1.type());
	S12.t = O1 - S12.scale * S12.R * O2;

	// Step 8: Transformation
	cv::Mat sR12, sR21;

	S12.T = cv::Mat::eye(4, 4, P1.type());
	S21.T = cv::Mat::eye(4, 4, P1.type());

	// Step 8.1 T12
	sR12 = S12.scale * S12.R;
	sR12.copyTo(CameraPose::GetR(S12.T));
	S12.t.copyTo(CameraPose::Gett(S12.T));

	// Step 8.2 T21
	S21.R = S12.R.t();
	S21.scale = 1.f / S12.scale;
	sR21 = S21.scale * S21.R;
	S21.t = -sR21 * S12.t;
	sR21.copyTo(CameraPose::GetR(S21.T));
	S21.t.copyTo(CameraPose::Gett(S21.T));
}

static void Project(const std::vector<cv::Mat>& points3D, std::vector<cv::Mat>& points2D, const cv::Mat& Tcw,
	const CameraParams& camera)
{
	cv::Mat Rcw = CameraPose::GetR(Tcw);
	cv::Mat tcw = CameraPose::Gett(Tcw);

	const float fx = camera.fx;
	const float fy = camera.fy;
	const float cx = camera.cx;
	const float cy = camera.cy;

	points2D.clear();
	points2D.reserve(points3D.size());

	for (size_t i = 0; i < points3D.size(); i++)
	{
		const cv::Mat P3Dc = Rcw * points3D[i] + tcw;
		const float invZ = 1.f / (P3Dc.at<float>(2));
		const float u = P3Dc.at<float>(0) * invZ;
		const float v = P3Dc.at<float>(1) * invZ;
		points2D.push_back((cv::Mat_<float>(2, 1) << fx * u + cx, fy * v + cy));
	}
}

static void FromCameraToImage(const std::vector<cv::Mat>& points3D, std::vector<cv::Mat>& points2D,
	const CameraParams& camera)
{
	const float fx = camera.fx;
	const float fy = camera.fy;
	const float cx = camera.cx;
	const float cy = camera.cy;

	points2D.clear();
	points2D.reserve(points3D.size());

	for (size_t i = 0; i < points3D.size(); i++)
	{
		const float invZ = 1 / (points3D[i].at<float>(2));
		const float u = points3D[i].at<float>(0) * invZ;
		const float v = points3D[i].at<float>(1) * invZ;
		points2D.push_back((cv::Mat_<float>(2, 1) << fx * u + cx, fy * v + cy));
	}
}

Sim3Solver::Sim3Solver(const KeyFrame* keyframe1, const KeyFrame* keyframe2, const std::vector<MapPoint*>& matches,
	bool fixScale) : iterations_(0), maxInliers_(0), fixScale_(fixScale)
{
	const std::vector<MapPoint*> mappoints1 = keyframe1->GetMapPointMatches();

	nkeypoints1_ = static_cast<int>(matches.size());

	indices1_.reserve(nkeypoints1_);
	Xc1_.reserve(nkeypoints1_);
	Xc2_.reserve(nkeypoints1_);

	const cv::Mat Rcw1 = keyframe1->GetRotation();
	const cv::Mat tcw1 = keyframe1->GetTranslation();
	const cv::Mat Rcw2 = keyframe2->GetRotation();
	const cv::Mat tcw2 = keyframe2->GetTranslation();

	allIndices_.reserve(nkeypoints1_);

	size_t idx = 0;
	for (int i1 = 0; i1 < nkeypoints1_; i1++)
	{
		const MapPoint* mappoint1 = mappoints1[i1];
		const MapPoint* mappoint2 = matches[i1];

		if (!mappoint1 || !mappoint2 || mappoint1->isBad() || mappoint2->isBad())
			continue;

		const int indexKF1 = mappoint1->GetIndexInKeyFrame(keyframe1);
		const int indexKF2 = mappoint2->GetIndexInKeyFrame(keyframe2);

		if (indexKF1 < 0 || indexKF2 < 0)
			continue;

		const cv::KeyPoint& keypoint1 = keyframe1->keypointsUn[indexKF1];
		const cv::KeyPoint& keypoint2 = keyframe2->keypointsUn[indexKF2];

		const float sigmaSq1 = keyframe1->pyramid.sigmaSq[keypoint1.octave];
		const float sigmaSq2 = keyframe2->pyramid.sigmaSq[keypoint2.octave];

		maxErrorSq1_.push_back(9.21 * sigmaSq1);
		maxErrorSq2_.push_back(9.21 * sigmaSq2);

		const cv::Mat X3D1w = mappoint1->GetWorldPos();
		const cv::Mat X3D2w = mappoint2->GetWorldPos();
		Xc1_.push_back(Rcw1 * X3D1w + tcw1);
		Xc2_.push_back(Rcw2 * X3D2w + tcw2);

		indices1_.push_back(i1);
		allIndices_.push_back(idx++);
	}

	camera1_ = keyframe1->camera;
	camera2_ = keyframe2->camera;

	FromCameraToImage(Xc1_, points1_, camera1_);
	FromCameraToImage(Xc2_, points2_, camera2_);

	SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
	probability_ = probability;
	minInliers_ = minInliers;
	maxIterations_ = maxIterations;

	nmatches_ = static_cast<int>(allIndices_.size()); // number of correspondences

	inliers_.resize(nmatches_);

	// Adjust Parameters according to number of correspondences
	const float epsilon = 1.f * minInliers_ / nmatches_;

	// Set RANSAC iterations according to probability, epsilon, and max iterations
	int niterations = 1;
	if (minInliers_ != nmatches_)
		niterations = static_cast<int>(ceil(log(1 - probability_) / log(1 - pow(epsilon, 3))));

	maxIterations_ = std::max(1, std::min(niterations, maxIterations_));

	iterations_ = 0;
}

bool Sim3Solver::iterate(int maxk, Sim3& sim3, std::vector<bool>& isInlier)
{
	terminate_ = false;
	isInlier.assign(nkeypoints1_, false);

	if (nmatches_ < minInliers_)
	{
		terminate_ = true;
		return false;
	}

	std::vector<size_t> availableIndices;

	cv::Mat P1(3, 3, CV_32F);
	cv::Mat P2(3, 3, CV_32F);

	for (int k = 0; iterations_ < maxIterations_ && k < maxk; k++, iterations_++)
	{
		availableIndices = allIndices_;

		// Get min set of points
		for (int i = 0; i < 3; ++i)
		{
			const int randi = DUtils::Random::RandomInt(0, static_cast<int>(availableIndices.size() - 1));
			const size_t idx = availableIndices[randi];

			Xc1_[idx].copyTo(P1.col(i));
			Xc2_[idx].copyTo(P2.col(i));

			availableIndices[randi] = availableIndices.back();
			availableIndices.pop_back();
		}

		Sim3 S12, S21;
		ComputeSim3(P1, P2, S12, S21, fixScale_);

		//CheckInliers();
		std::vector<cv::Mat> proj1, proj2;
		Project(Xc2_, proj1, S12.T, camera1_);
		Project(Xc1_, proj2, S21.T, camera2_);

		int ninliers = 0;
		for (size_t i = 0; i < points1_.size(); i++)
		{
			cv::Mat diff1 = points1_[i] - proj1[i];
			cv::Mat diff2 = points2_[i] - proj2[i];

			const double errorSq1 = diff1.dot(diff1);
			const double errorSq2 = diff2.dot(diff2);
			const bool inlier = errorSq1 < maxErrorSq1_[i] && errorSq2 < maxErrorSq2_[i];
			inliers_[i] = inlier;
			if (inlier)
				ninliers++;
		}

		if (ninliers >= maxInliers_)
		{
			maxInliers_ = ninliers;
			CopySim3(S12, bestS12_);
			if (ninliers > minInliers_)
			{
				CopySim3(bestS12_, sim3);
				for (int i = 0; i < nmatches_; i++)
					if (inliers_[i])
						isInlier[indices1_[i]] = true;
				return true;
			}
		}
	}

	if (iterations_ >= maxIterations_)
		terminate_ = true;

	return false;
}

bool Sim3Solver::terminate() const
{
	return terminate_;
}

} //namespace ORB_SLAM
