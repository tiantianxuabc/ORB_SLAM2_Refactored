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

Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const std::vector<MapPoint *> &vpMatched12, const bool bFixScale) :
	mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
	mpKF1 = pKF1;
	mpKF2 = pKF2;

	std::vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();

	mN1 = vpMatched12.size();

	mvpMapPoints1.reserve(mN1);
	mvpMapPoints2.reserve(mN1);
	mvpMatches12 = vpMatched12;
	mvnIndices1.reserve(mN1);
	mvX3Dc1.reserve(mN1);
	mvX3Dc2.reserve(mN1);

	cv::Mat Rcw1 = pKF1->GetRotation();
	cv::Mat tcw1 = pKF1->GetTranslation();
	cv::Mat Rcw2 = pKF2->GetRotation();
	cv::Mat tcw2 = pKF2->GetTranslation();

	mvAllIndices.reserve(mN1);

	size_t idx = 0;
	for (int i1 = 0; i1 < mN1; i1++)
	{
		if (vpMatched12[i1])
		{
			MapPoint* pMP1 = vpKeyFrameMP1[i1];
			MapPoint* pMP2 = vpMatched12[i1];

			if (!pMP1)
				continue;

			if (pMP1->isBad() || pMP2->isBad())
				continue;

			int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
			int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

			if (indexKF1 < 0 || indexKF2 < 0)
				continue;

			const cv::KeyPoint &kp1 = pKF1->keypointsUn[indexKF1];
			const cv::KeyPoint &kp2 = pKF2->keypointsUn[indexKF2];

			const float sigmaSquare1 = pKF1->pyramid.sigmaSq[kp1.octave];
			const float sigmaSquare2 = pKF2->pyramid.sigmaSq[kp2.octave];

			mvnMaxError1.push_back(9.210*sigmaSquare1);
			mvnMaxError2.push_back(9.210*sigmaSquare2);

			mvpMapPoints1.push_back(pMP1);
			mvpMapPoints2.push_back(pMP2);
			mvnIndices1.push_back(i1);

			cv::Mat X3D1w = pMP1->GetWorldPos();
			mvX3Dc1.push_back(Rcw1*X3D1w + tcw1);

			cv::Mat X3D2w = pMP2->GetWorldPos();
			mvX3Dc2.push_back(Rcw2*X3D2w + tcw2);

			mvAllIndices.push_back(idx);
			idx++;
		}
	}

	mK1 = pKF1->camera.Mat();
	mK2 = pKF2->camera.Mat();

	FromCameraToImage(mvX3Dc1, mvP1im1, mK1);
	FromCameraToImage(mvX3Dc2, mvP2im2, mK2);

	SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
	mRansacProb = probability;
	mRansacMinInliers = minInliers;
	mRansacMaxIts = maxIterations;

	N = mvpMapPoints1.size(); // number of correspondences

	mvbInliersi.resize(N);

	// Adjust Parameters according to number of correspondences
	float epsilon = (float)mRansacMinInliers / N;

	// Set RANSAC iterations according to probability, epsilon, and max iterations
	int nIterations;

	if (mRansacMinInliers == N)
		nIterations = 1;
	else
		nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

	mRansacMaxIts = std::max(1, std::min(nIterations, mRansacMaxIts));

	mnIterations = 0;
}

cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers)
{
	bNoMore = false;
	vbInliers = std::vector<bool>(mN1, false);
	nInliers = 0;

	if (N < mRansacMinInliers)
	{
		bNoMore = true;
		return cv::Mat();
	}

	std::vector<size_t> vAvailableIndices;

	cv::Mat P3Dc1i(3, 3, CV_32F);
	cv::Mat P3Dc2i(3, 3, CV_32F);

	int nCurrentIterations = 0;
	while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations)
	{
		nCurrentIterations++;
		mnIterations++;

		vAvailableIndices = mvAllIndices;

		// Get min set of points
		for (short i = 0; i < 3; ++i)
		{
			int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

			int idx = vAvailableIndices[randi];

			mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
			mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

			vAvailableIndices[randi] = vAvailableIndices.back();
			vAvailableIndices.pop_back();
		}

		Sim3 S12, S21;
		ComputeSim3(P3Dc1i, P3Dc2i, S12, S21, mbFixScale);

		//CheckInliers();
		std::vector<cv::Mat> vP1im2, vP2im1;
		Project(mvX3Dc2, vP2im1, S12.T, mK1);
		Project(mvX3Dc1, vP1im2, S21.T, mK2);

		mnInliersi = 0;

		for (size_t i = 0; i < mvP1im1.size(); i++)
		{
			cv::Mat dist1 = mvP1im1[i] - vP2im1[i];
			cv::Mat dist2 = vP1im2[i] - mvP2im2[i];

			const float err1 = dist1.dot(dist1);
			const float err2 = dist2.dot(dist2);

			if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i])
			{
				mvbInliersi[i] = true;
				mnInliersi++;
			}
			else
				mvbInliersi[i] = false;
		}

		if (mnInliersi >= mnBestInliers)
		{
			mvbBestInliers = mvbInliersi;
			mnBestInliers = mnInliersi;
			mBestT12 = S12.T.clone();
			mBestRotation = S12.R.clone();
			mBestTranslation = S12.t.clone();
			mBestScale = S12.scale;

			if (mnInliersi > mRansacMinInliers)
			{
				nInliers = mnInliersi;
				for (int i = 0; i < N; i++)
					if (mvbInliersi[i])
						vbInliers[mvnIndices1[i]] = true;
				return mBestT12;
			}
		}
	}

	if (mnIterations >= mRansacMaxIts)
		bNoMore = true;

	return cv::Mat();
}

cv::Mat Sim3Solver::find(std::vector<bool> &vbInliers12, int &nInliers)
{
	bool bFlag;
	return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers);
}

cv::Mat Sim3Solver::GetEstimatedRotation()
{
	return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
	return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
	return mBestScale;
}

} //namespace ORB_SLAM
