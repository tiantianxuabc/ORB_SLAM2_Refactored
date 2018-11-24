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


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

class KeyFrame;
class MapPoint;

class Sim3Solver
{
public:

	struct Sim3
	{
		cv::Mat R;
		cv::Mat t;
		float scale;
		cv::Mat T;
	};

	Sim3Solver(const KeyFrame* pKF1, const KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, bool bFixScale = true);

	void SetRansacParameters(double probability = 0.99, int minInliers = 6, int maxIterations = 300);

	cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);

	cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

	cv::Mat GetEstimatedRotation();
	cv::Mat GetEstimatedTranslation();
	float GetEstimatedScale();

private:

	std::vector<cv::Mat> Xc1_;
	std::vector<cv::Mat> Xc2_;
	//std::vector<MapPoint*> mvpMapPoints1;
	//std::vector<MapPoint*> mvpMapPoints2;
	//std::vector<MapPoint*> mvpMatches12;
	std::vector<size_t> indices1_;
	std::vector<double> maxErrorSq1_;
	std::vector<double> maxErrorSq2_;

	int N;
	int nkeypoints1_;

	std::vector<bool> mvbInliersi;
	
	// Current Ransac State
	int iterations_;
	std::vector<bool> mvbBestInliers;
	int maxInliers_;
	cv::Mat mBestT12;
	cv::Mat mBestRotation;
	cv::Mat mBestTranslation;
	float mBestScale;

	// Scale is fixed to 1 in the stereo/RGBD case
	bool fixScale_;

	// Indices for random selection
	std::vector<size_t> allIndices;

	// Projections
	std::vector<cv::Mat> points1_;
	std::vector<cv::Mat> points2_;

	// RANSAC probability
	double mRansacProb;

	// RANSAC min inliers
	int mRansacMinInliers;

	// RANSAC max iterations
	int mRansacMaxIts;

	// Calibration
	cv::Mat K1_;
	cv::Mat K2_;
};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
