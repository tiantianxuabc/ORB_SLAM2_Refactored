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

static void ComputeCentroid(const cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
	cv::reduce(P, C, 1, CV_REDUCE_SUM);
	C = C / P.cols;

	for (int i = 0; i < P.cols; i++)
	{
		Pr.col(i) = P.col(i) - C;
	}
}

static void ComputeSim3(const cv::Mat &P1, const cv::Mat &P2, Sim3& S12, Sim3& S21, bool mbFixScale)
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

	// Step 3: Compute N matrix

	double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

	cv::Mat N(4, 4, P1.type());

	N11 = M.at<float>(0, 0) + M.at<float>(1, 1) + M.at<float>(2, 2);
	N12 = M.at<float>(1, 2) - M.at<float>(2, 1);
	N13 = M.at<float>(2, 0) - M.at<float>(0, 2);
	N14 = M.at<float>(0, 1) - M.at<float>(1, 0);
	N22 = M.at<float>(0, 0) - M.at<float>(1, 1) - M.at<float>(2, 2);
	N23 = M.at<float>(0, 1) + M.at<float>(1, 0);
	N24 = M.at<float>(2, 0) + M.at<float>(0, 2);
	N33 = -M.at<float>(0, 0) + M.at<float>(1, 1) - M.at<float>(2, 2);
	N34 = M.at<float>(1, 2) + M.at<float>(2, 1);
	N44 = -M.at<float>(0, 0) - M.at<float>(1, 1) + M.at<float>(2, 2);

	N = (cv::Mat_<float>(4, 4) << N11, N12, N13, N14,
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

	S12.R.create(3, 3, P1.type());

	cv::Rodrigues(vec, S12.R); // computes the rotation matrix from angle-axis

							   // Step 5: Rotate set 2

	cv::Mat P3 = S12.R*Pr2;

	// Step 6: Scale

	if (!mbFixScale)
	{
		double nom = Pr1.dot(P3);
		cv::Mat aux_P3(P3.size(), P3.type());
		aux_P3 = P3;
		cv::pow(P3, 2, aux_P3);
		double den = 0;

		for (int i = 0; i < aux_P3.rows; i++)
		{
			for (int j = 0; j < aux_P3.cols; j++)
			{
				den += aux_P3.at<float>(i, j);
			}
		}

		S12.scale = nom / den;
	}
	else
		S12.scale = 1.0f;

	// Step 7: Translation

	S12.t.create(1, 3, P1.type());
	S12.t = O1 - S12.scale*S12.R*O2;

	// Step 8: Transformation

	// Step 8.1 T12
	S12.T = cv::Mat::eye(4, 4, P1.type());

	cv::Mat sR = S12.scale*S12.R;

	sR.copyTo(S12.T.rowRange(0, 3).colRange(0, 3));
	S12.t.copyTo(S12.T.rowRange(0, 3).col(3));

	// Step 8.2 T21

	S21.T = cv::Mat::eye(4, 4, P1.type());

	cv::Mat sRinv = (1.0 / S12.scale)*S12.R.t();

	sRinv.copyTo(S21.T.rowRange(0, 3).colRange(0, 3));
	cv::Mat tinv = -sRinv*S12.t;
	tinv.copyTo(S21.T.rowRange(0, 3).col(3));
}

static void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
	cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
	cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
	const float &fx = K.at<float>(0, 0);
	const float &fy = K.at<float>(1, 1);
	const float &cx = K.at<float>(0, 2);
	const float &cy = K.at<float>(1, 2);

	vP2D.clear();
	vP2D.reserve(vP3Dw.size());

	for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++)
	{
		cv::Mat P3Dc = Rcw*vP3Dw[i] + tcw;
		const float invz = 1 / (P3Dc.at<float>(2));
		const float x = P3Dc.at<float>(0)*invz;
		const float y = P3Dc.at<float>(1)*invz;

		vP2D.push_back((cv::Mat_<float>(2, 1) << fx*x + cx, fy*y + cy));
	}
}

static void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K)
{
	const float &fx = K.at<float>(0, 0);
	const float &fy = K.at<float>(1, 1);
	const float &cx = K.at<float>(0, 2);
	const float &cy = K.at<float>(1, 2);

	vP2D.clear();
	vP2D.reserve(vP3Dc.size());

	for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++)
	{
		const float invz = 1 / (vP3Dc[i].at<float>(2));
		const float x = vP3Dc[i].at<float>(0)*invz;
		const float y = vP3Dc[i].at<float>(1)*invz;

		vP2D.push_back((cv::Mat_<float>(2, 1) << fx*x + cx, fy*y + cy));
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

	allIndices.reserve(nkeypoints1_);

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
		allIndices.push_back(idx++);
	}

	K1_ = keyframe1->camera.Mat();
	K2_ = keyframe2->camera.Mat();

	FromCameraToImage(Xc1_, points1_, K1_);
	FromCameraToImage(Xc2_, points2_, K2_);

	SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
	probability_ = probability;
	minInliers_ = minInliers;
	maxIterations_ = maxIterations;

	N = static_cast<int>(allIndices.size()); // number of correspondences

	inliers_.resize(N);

	// Adjust Parameters according to number of correspondences
	const float epsilon = 1.f * minInliers_ / N;

	// Set RANSAC iterations according to probability, epsilon, and max iterations
	int niterations = 1;
	if (minInliers_ != N)
		niterations = static_cast<int>(ceil(log(1 - probability_) / log(1 - pow(epsilon, 3))));
	
	maxIterations_ = std::max(1, std::min(niterations, maxIterations_));

	iterations_ = 0;
}

cv::Mat Sim3Solver::iterate(int maxk, bool& terminate, std::vector<bool>& isInlier, int& ninliers)
{
	terminate = false;
	isInlier.assign(nkeypoints1_, false);
	ninliers = 0;

	if (N < minInliers_)
	{
		terminate = true;
		return cv::Mat();
	}

	std::vector<size_t> availableIndices;

	cv::Mat P1(3, 3, CV_32F);
	cv::Mat P2(3, 3, CV_32F);

	for (int k = 0; iterations_ < maxIterations_ && k < maxk; k++, iterations_++)
	{
		availableIndices = allIndices;

		// Get min set of points
		for (int i = 0; i < 3; ++i)
		{
			const int randi = DUtils::Random::RandomInt(0, availableIndices.size() - 1);
			const int idx = availableIndices[randi];

			Xc1_[idx].copyTo(P1.col(i));
			Xc2_[idx].copyTo(P2.col(i));

			availableIndices[randi] = availableIndices.back();
			availableIndices.pop_back();
		}

		Sim3 S12, S21;
		ComputeSim3(P1, P2, S12, S21, fixScale_);

		//CheckInliers();
		std::vector<cv::Mat> proj1, proj2;
		Project(Xc2_, proj1, S12.T, K1_);
		Project(Xc1_, proj2, S21.T, K2_);

		int _ninliers = 0;
		for (size_t i = 0; i < points1_.size(); i++)
		{
			cv::Mat diff1 = points1_[i] - proj1[i];
			cv::Mat diff2 = points2_[i] - proj2[i];

			const double errorSq1 = diff1.dot(diff1);
			const double errorSq2 = diff2.dot(diff2);
			const bool inlier = errorSq1 < maxErrorSq1_[i] && errorSq2 < maxErrorSq2_[i];
			inliers_[i] = inlier;
			if (inlier)
				_ninliers++;
		}

		if (_ninliers >= maxInliers_)
		{
			//bestInliers_ = inliers_;
			maxInliers_ = _ninliers;
			bestT12_ = S12.T.clone();
			bestRotation_ = S12.R.clone();
			bestTranslation_ = S12.t.clone();
			bestScale_ = S12.scale;

			if (_ninliers > minInliers_)
			{
				ninliers = _ninliers;
				for (int i = 0; i < N; i++)
					if (inliers_[i])
						isInlier[indices1_[i]] = true;
				return bestT12_;
			}
		}
	}

	if (iterations_ >= maxIterations_)
		terminate = true;

	return cv::Mat();
}

cv::Mat Sim3Solver::find(std::vector<bool> &vbInliers12, int &nInliers)
{
	bool bFlag;
	return iterate(maxIterations_, bFlag, vbInliers12, nInliers);
}

cv::Mat Sim3Solver::GetEstimatedRotation()
{
	return bestRotation_.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
	return bestTranslation_.clone();
}

float Sim3Solver::GetEstimatedScale()
{
	return bestScale_;
}

} //namespace ORB_SLAM
