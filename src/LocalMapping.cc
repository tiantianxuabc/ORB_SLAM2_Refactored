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

#include "LocalMapping.h"

#include "Tracking.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Usleep.h"
#include "KeyFrame.h"
#include "Map.h"

#include<mutex>

#define LOCK_MUTEX_NEW_KF()    std::unique_lock<std::mutex> lock1(mutexNewKFs_);
#define LOCK_MUTEX_RESET()     std::unique_lock<std::mutex> lock2(mutexReset_);
#define LOCK_MUTEX_FINISH()    std::unique_lock<std::mutex> lock3(mutexFinish_);
#define LOCK_MUTEX_STOP()      std::unique_lock<std::mutex> lock4(mutexStop_);
#define LOCK_MUTEX_ACCEPT_KF() std::unique_lock<std::mutex> lock5(mutexAccept_);

namespace ORB_SLAM2
{

class LocalMappingImpl : public LocalMapping
{
public:

	LocalMappingImpl::LocalMappingImpl(Map* map, bool monocular) :
		monocular_(monocular), resetRequested_(false), finishRequested_(false), finished_(true), map_(map),
		abortBA_(false), stopped_(false), stopRequested_(false), notStop_(false), acceptKeyFrames_(true)
	{
	}

	void SetLoopCloser(LoopClosing* loopCloser) override
	{
		loopCloser_ = loopCloser;
	}

	void SetTracker(Tracking* tracker) override
	{
		tracker_ = tracker;
	}

	// Main function
	void Run() override
	{
		finished_ = false;

		while (1)
		{
			// Tracking will see that Local Mapping is busy
			SetAcceptKeyFrames(false);

			// Check if there are keyframes in the queue
			if (CheckNewKeyFrames())
			{
				// BoW conversion and insertion in Map
				ProcessNewKeyFrame();

				// Check recent MapPoints
				MapPointCulling();

				// Triangulate new MapPoints
				CreateNewMapPoints();

				if (!CheckNewKeyFrames())
				{
					// Find more matches in neighbor keyframes and fuse point duplications
					SearchInNeighbors();
				}

				abortBA_ = false;

				if (!CheckNewKeyFrames() && !stopRequested())
				{
					// Local BA
					if (map_->KeyFramesInMap() > 2)
						Optimizer::LocalBundleAdjustment(currKeyFrame_, &abortBA_, map_);

					// Check redundant local Keyframes
					KeyFrameCulling();
				}

				loopCloser_->InsertKeyFrame(currKeyFrame_);
			}
			else if (Stop())
			{
				// Safe area to stop
				while (isStopped() && !CheckFinish())
				{
					usleep(3000);
				}
				if (CheckFinish())
					break;
			}

			ResetIfRequested();

			// Tracking will see that Local Mapping is busy
			SetAcceptKeyFrames(true);

			if (CheckFinish())
				break;

			usleep(3000);
		}

		SetFinish();
	}

	void InsertKeyFrame(KeyFrame* keyframe) override
	{
		LOCK_MUTEX_NEW_KF();
		newKeyFrames_.push_back(keyframe);
		abortBA_ = true;
	}

	// Thread Synch
	void RequestStop() override
	{
		LOCK_MUTEX_STOP();
		stopRequested_ = true;
		LOCK_MUTEX_NEW_KF();
		abortBA_ = true;
	}

	void RequestReset() override
	{
		{
			LOCK_MUTEX_RESET();
			resetRequested_ = true;
		}

		while (1)
		{
			{
				LOCK_MUTEX_RESET();
				if (!resetRequested_)
					break;
			}
			usleep(3000);
		}
	}

	bool Stop() override
	{
		LOCK_MUTEX_STOP();
		if (stopRequested_ && !notStop_)
		{
			stopped_ = true;
			cout << "Local Mapping STOP" << endl;
			return true;
		}

		return false;
	}

	void Release() override
	{
		LOCK_MUTEX_STOP();
		LOCK_MUTEX_FINISH();

		if (finished_)
			return;

		stopped_ = false;
		stopRequested_ = false;
		for (list<KeyFrame*>::iterator lit = newKeyFrames_.begin(), lend = newKeyFrames_.end(); lit != lend; lit++)
			delete *lit;
		newKeyFrames_.clear();

		cout << "Local Mapping RELEASE" << endl;
	}

	bool isStopped() const override
	{
		LOCK_MUTEX_STOP();
		return stopped_;
	}

	bool stopRequested() const override
	{
		LOCK_MUTEX_STOP();
		return stopRequested_;
	}

	bool AcceptKeyFrames() const override
	{
		LOCK_MUTEX_ACCEPT_KF();
		return acceptKeyFrames_;
	}

	void SetAcceptKeyFrames(bool flag) override
	{
		LOCK_MUTEX_ACCEPT_KF();
		acceptKeyFrames_ = flag;
	}

	bool SetNotStop(bool flag) override
	{
		LOCK_MUTEX_STOP();

		if (flag && stopped_)
			return false;

		notStop_ = flag;

		return true;
	}

	void InterruptBA() override
	{
		abortBA_ = true;
	}

	void RequestFinish() override
	{
		LOCK_MUTEX_FINISH();
		finishRequested_ = true;
	}

	bool isFinished() const override
	{
		LOCK_MUTEX_FINISH();
		return finished_;
	}

	int KeyframesInQueue() const override
	{
		LOCK_MUTEX_NEW_KF();
		return newKeyFrames_.size();
	}

private:

	bool CheckNewKeyFrames()
	{
		LOCK_MUTEX_NEW_KF();
		return(!newKeyFrames_.empty());
	}

	void ProcessNewKeyFrame()
	{
		{
			LOCK_MUTEX_NEW_KF();
			currKeyFrame_ = newKeyFrames_.front();
			newKeyFrames_.pop_front();
		}

		// Compute Bags of Words structures
		currKeyFrame_->ComputeBoW();

		// Associate MapPoints to the new keyframe and update normal and descriptor
		const vector<MapPoint*> vpMapPointMatches = currKeyFrame_->GetMapPointMatches();

		for (size_t i = 0; i < vpMapPointMatches.size(); i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					if (!pMP->IsInKeyFrame(currKeyFrame_))
					{
						pMP->AddObservation(currKeyFrame_, i);
						pMP->UpdateNormalAndDepth();
						pMP->ComputeDistinctiveDescriptors();
					}
					else // this can only happen for new stereo points inserted by the Tracking
					{
						mlpRecentAddedMapPoints.push_back(pMP);
					}
				}
			}
		}

		// Update links in the Covisibility Graph
		currKeyFrame_->UpdateConnections();

		// Insert Keyframe in Map
		map_->AddKeyFrame(currKeyFrame_);
	}

	void MapPointCulling()
	{
		// Check Recent Added MapPoints
		list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
		const unsigned long int nCurrentKFid = currKeyFrame_->mnId;

		int nThObs;
		if (monocular_)
			nThObs = 2;
		else
			nThObs = 3;
		const int cnThObs = nThObs;

		while (lit != mlpRecentAddedMapPoints.end())
		{
			MapPoint* pMP = *lit;
			if (pMP->isBad())
			{
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if (pMP->GetFoundRatio() < 0.25f)
			{
				pMP->SetBadFlag();
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
			{
				pMP->SetBadFlag();
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
				lit = mlpRecentAddedMapPoints.erase(lit);
			else
				lit++;
		}
	}

	void CreateNewMapPoints()
	{
		// Retrieve neighbor keyframes in covisibility graph
		int nn = 10;
		if (monocular_)
			nn = 20;
		const vector<KeyFrame*> vpNeighKFs = currKeyFrame_->GetBestCovisibilityKeyFrames(nn);

		ORBmatcher matcher(0.6, false);

		cv::Mat Rcw1 = currKeyFrame_->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = currKeyFrame_->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));
		cv::Mat Ow1 = currKeyFrame_->GetCameraCenter();

		const float &fx1 = currKeyFrame_->camera.fx;
		const float &fy1 = currKeyFrame_->camera.fy;
		const float &cx1 = currKeyFrame_->camera.cx;
		const float &cy1 = currKeyFrame_->camera.cy;
		const float &invfx1 = 1.f / fx1;
		const float &invfy1 = 1.f / fy1;

		const float ratioFactor = 1.5f*currKeyFrame_->pyramid.scaleFactor;

		int nnew = 0;

		// Search matches with epipolar restriction and triangulate
		for (size_t i = 0; i < vpNeighKFs.size(); i++)
		{
			if (i > 0 && CheckNewKeyFrames())
				return;

			KeyFrame* pKF2 = vpNeighKFs[i];

			// Check first that baseline is not too short
			cv::Mat Ow2 = pKF2->GetCameraCenter();
			cv::Mat vBaseline = Ow2 - Ow1;
			const float baseline = cv::norm(vBaseline);

			if (!monocular_)
			{
				if (baseline < pKF2->camera.baseline)
					continue;
			}
			else
			{
				const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
				const float ratioBaselineDepth = baseline / medianDepthKF2;

				if (ratioBaselineDepth < 0.01)
					continue;
			}

			// Compute Fundamental Matrix
			cv::Mat F12 = ComputeF12(currKeyFrame_, pKF2);

			// Search matches that fullfil epipolar constraint
			vector<pair<size_t, size_t> > vMatchedIndices;
			matcher.SearchForTriangulation(currKeyFrame_, pKF2, F12, vMatchedIndices, false);

			cv::Mat Rcw2 = pKF2->GetRotation();
			cv::Mat Rwc2 = Rcw2.t();
			cv::Mat tcw2 = pKF2->GetTranslation();
			cv::Mat Tcw2(3, 4, CV_32F);
			Rcw2.copyTo(Tcw2.colRange(0, 3));
			tcw2.copyTo(Tcw2.col(3));

			const float &fx2 = pKF2->camera.fx;
			const float &fy2 = pKF2->camera.fy;
			const float &cx2 = pKF2->camera.cx;
			const float &cy2 = pKF2->camera.cy;
			const float &invfx2 = 1.f / fx2;
			const float &invfy2 = 1.f / fy2;

			// Triangulate each match
			const int nmatches = vMatchedIndices.size();
			for (int ikp = 0; ikp < nmatches; ikp++)
			{
				const int &idx1 = vMatchedIndices[ikp].first;
				const int &idx2 = vMatchedIndices[ikp].second;

				const cv::KeyPoint &kp1 = currKeyFrame_->mvKeysUn[idx1];
				const float kp1_ur = currKeyFrame_->mvuRight[idx1];
				bool bStereo1 = kp1_ur >= 0;

				const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
				const float kp2_ur = pKF2->mvuRight[idx2];
				bool bStereo2 = kp2_ur >= 0;

				// Check parallax between rays
				cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1)*invfx1, (kp1.pt.y - cy1)*invfy1, 1.0);
				cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2)*invfx2, (kp2.pt.y - cy2)*invfy2, 1.0);

				cv::Mat ray1 = Rwc1*xn1;
				cv::Mat ray2 = Rwc2*xn2;
				const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

				float cosParallaxStereo = cosParallaxRays + 1;
				float cosParallaxStereo1 = cosParallaxStereo;
				float cosParallaxStereo2 = cosParallaxStereo;

				if (bStereo1)
					cosParallaxStereo1 = cos(2 * atan2(currKeyFrame_->camera.baseline / 2, currKeyFrame_->mvDepth[idx1]));
				else if (bStereo2)
					cosParallaxStereo2 = cos(2 * atan2(pKF2->camera.baseline / 2, pKF2->mvDepth[idx2]));

				cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

				cv::Mat x3D;
				if (cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
				{
					// Linear Triangulation Method
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
					A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
					A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

					cv::Mat w, u, vt;
					cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

					x3D = vt.row(3).t();

					if (x3D.at<float>(3) == 0)
						continue;

					// Euclidean coordinates
					x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

				}
				else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
				{
					x3D = currKeyFrame_->UnprojectStereo(idx1);
				}
				else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
				{
					x3D = pKF2->UnprojectStereo(idx2);
				}
				else
					continue; //No stereo and very low parallax

				cv::Mat x3Dt = x3D.t();

				//Check triangulation in front of cameras
				float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
				if (z1 <= 0)
					continue;

				float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
				if (z2 <= 0)
					continue;

				//Check reprojection error in first keyframe
				const float &sigmaSquare1 = currKeyFrame_->pyramid.sigmaSq[kp1.octave];
				const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
				const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
				const float invz1 = 1.0 / z1;

				if (!bStereo1)
				{
					float u1 = fx1*x1*invz1 + cx1;
					float v1 = fy1*y1*invz1 + cy1;
					float errX1 = u1 - kp1.pt.x;
					float errY1 = v1 - kp1.pt.y;
					if ((errX1*errX1 + errY1*errY1) > 5.991*sigmaSquare1)
						continue;
				}
				else
				{
					float u1 = fx1*x1*invz1 + cx1;
					float u1_r = u1 - currKeyFrame_->camera.bf*invz1;
					float v1 = fy1*y1*invz1 + cy1;
					float errX1 = u1 - kp1.pt.x;
					float errY1 = v1 - kp1.pt.y;
					float errX1_r = u1_r - kp1_ur;
					if ((errX1*errX1 + errY1*errY1 + errX1_r*errX1_r) > 7.8*sigmaSquare1)
						continue;
				}

				//Check reprojection error in second keyframe
				const float sigmaSquare2 = pKF2->pyramid.sigmaSq[kp2.octave];
				const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
				const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
				const float invz2 = 1.0 / z2;
				if (!bStereo2)
				{
					float u2 = fx2*x2*invz2 + cx2;
					float v2 = fy2*y2*invz2 + cy2;
					float errX2 = u2 - kp2.pt.x;
					float errY2 = v2 - kp2.pt.y;
					if ((errX2*errX2 + errY2*errY2) > 5.991*sigmaSquare2)
						continue;
				}
				else
				{
					float u2 = fx2*x2*invz2 + cx2;
					float u2_r = u2 - currKeyFrame_->camera.bf*invz2;
					float v2 = fy2*y2*invz2 + cy2;
					float errX2 = u2 - kp2.pt.x;
					float errY2 = v2 - kp2.pt.y;
					float errX2_r = u2_r - kp2_ur;
					if ((errX2*errX2 + errY2*errY2 + errX2_r*errX2_r) > 7.8*sigmaSquare2)
						continue;
				}

				//Check scale consistency
				cv::Mat normal1 = x3D - Ow1;
				float dist1 = cv::norm(normal1);

				cv::Mat normal2 = x3D - Ow2;
				float dist2 = cv::norm(normal2);

				if (dist1 == 0 || dist2 == 0)
					continue;

				const float ratioDist = dist2 / dist1;
				const float ratioOctave = currKeyFrame_->pyramid.scaleFactors[kp1.octave] / pKF2->pyramid.scaleFactors[kp2.octave];

				/*if(fabs(ratioDist-ratioOctave)>ratioFactor)
				continue;*/
				if (ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
					continue;

				// Triangulation is succesfull
				MapPoint* pMP = new MapPoint(x3D, currKeyFrame_, map_);

				pMP->AddObservation(currKeyFrame_, idx1);
				pMP->AddObservation(pKF2, idx2);

				currKeyFrame_->AddMapPoint(pMP, idx1);
				pKF2->AddMapPoint(pMP, idx2);

				pMP->ComputeDistinctiveDescriptors();

				pMP->UpdateNormalAndDepth();

				map_->AddMapPoint(pMP);
				mlpRecentAddedMapPoints.push_back(pMP);

				nnew++;
			}
		}
	}

	void SearchInNeighbors()
	{
		// Retrieve neighbor keyframes
		int nn = 10;
		if (monocular_)
			nn = 20;
		const vector<KeyFrame*> vpNeighKFs = currKeyFrame_->GetBestCovisibilityKeyFrames(nn);
		vector<KeyFrame*> vpTargetKFs;
		for (vector<KeyFrame*>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			if (pKFi->isBad() || pKFi->mnFuseTargetForKF == currKeyFrame_->mnId)
				continue;
			vpTargetKFs.push_back(pKFi);
			pKFi->mnFuseTargetForKF = currKeyFrame_->mnId;

			// Extend to some second neighbors
			const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
			for (vector<KeyFrame*>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
			{
				KeyFrame* pKFi2 = *vit2;
				if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == currKeyFrame_->mnId || pKFi2->mnId == currKeyFrame_->mnId)
					continue;
				vpTargetKFs.push_back(pKFi2);
			}
		}


		// Search matches by projection from current KF in target KFs
		ORBmatcher matcher;
		vector<MapPoint*> vpMapPointMatches = currKeyFrame_->GetMapPointMatches();
		for (vector<KeyFrame*>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;

			matcher.Fuse(pKFi, vpMapPointMatches);
		}

		// Search matches by projection from target KFs in current KF
		vector<MapPoint*> vpFuseCandidates;
		vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

		for (vector<KeyFrame*>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
		{
			KeyFrame* pKFi = *vitKF;

			vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

			for (vector<MapPoint*>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
			{
				MapPoint* pMP = *vitMP;
				if (!pMP)
					continue;
				if (pMP->isBad() || pMP->mnFuseCandidateForKF == currKeyFrame_->mnId)
					continue;
				pMP->mnFuseCandidateForKF = currKeyFrame_->mnId;
				vpFuseCandidates.push_back(pMP);
			}
		}

		matcher.Fuse(currKeyFrame_, vpFuseCandidates);


		// Update points
		vpMapPointMatches = currKeyFrame_->GetMapPointMatches();
		for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					pMP->ComputeDistinctiveDescriptors();
					pMP->UpdateNormalAndDepth();
				}
			}
		}

		// Update connections in covisibility graph
		currKeyFrame_->UpdateConnections();
	}

	cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
	{
		cv::Mat R1w = pKF1->GetRotation();
		cv::Mat t1w = pKF1->GetTranslation();
		cv::Mat R2w = pKF2->GetRotation();
		cv::Mat t2w = pKF2->GetTranslation();

		cv::Mat R12 = R1w*R2w.t();
		cv::Mat t12 = -R1w*R2w.t()*t2w + t1w;

		cv::Mat t12x = SkewSymmetricMatrix(t12);

		const cv::Mat &K1 = pKF1->camera.Mat();
		const cv::Mat &K2 = pKF2->camera.Mat();


		return K1.t().inv()*t12x*R12*K2.inv();
	}

	void KeyFrameCulling()
	{
		// Check redundant keyframes (only local keyframes)
		// A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
		// in at least other 3 keyframes (in the same or finer scale)
		// We only consider close stereo points
		vector<KeyFrame*> vpLocalKeyFrames = currKeyFrame_->GetVectorCovisibleKeyFrames();

		for (vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
		{
			KeyFrame* pKF = *vit;
			if (pKF->mnId == 0)
				continue;
			const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

			int nObs = 3;
			const int thObs = nObs;
			int nRedundantObservations = 0;
			int nMPs = 0;
			for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isBad())
					{
						if (!monocular_)
						{
							if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
								continue;
						}

						nMPs++;
						if (pMP->Observations() > thObs)
						{
							const int &scaleLevel = pKF->mvKeysUn[i].octave;
							const map<KeyFrame*, size_t> observations = pMP->GetObservations();
							int nObs = 0;
							for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
							{
								KeyFrame* pKFi = mit->first;
								if (pKFi == pKF)
									continue;
								const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

								if (scaleLeveli <= scaleLevel + 1)
								{
									nObs++;
									if (nObs >= thObs)
										break;
								}
							}
							if (nObs >= thObs)
							{
								nRedundantObservations++;
							}
						}
					}
				}
			}

			if (nRedundantObservations > 0.9*nMPs)
				pKF->SetBadFlag();
		}
	}

	cv::Mat SkewSymmetricMatrix(const cv::Mat &v)
	{
		return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
			v.at<float>(2), 0, -v.at<float>(0),
			-v.at<float>(1), v.at<float>(0), 0);
	}

	void ResetIfRequested()
	{
		LOCK_MUTEX_RESET();
		if (resetRequested_)
		{
			newKeyFrames_.clear();
			mlpRecentAddedMapPoints.clear();
			resetRequested_ = false;
		}
	}

	bool CheckFinish()
	{
		LOCK_MUTEX_FINISH();
		return finishRequested_;
	}

	void SetFinish()
	{
		LOCK_MUTEX_FINISH();
		finished_ = true;
		LOCK_MUTEX_STOP();
		stopped_ = true;
	}

	bool monocular_;
	bool resetRequested_;
	bool finishRequested_;
	bool finished_;

	Map* map_;

	LoopClosing* loopCloser_;
	Tracking* tracker_;

	std::list<KeyFrame*> newKeyFrames_;

	KeyFrame* currKeyFrame_;

	std::list<MapPoint*> mlpRecentAddedMapPoints;


	bool abortBA_;
	bool stopped_;
	bool stopRequested_;
	bool notStop_;
	bool acceptKeyFrames_;

	mutable std::mutex mutexNewKFs_;
	mutable std::mutex mutexReset_;
	mutable std::mutex mutexFinish_;
	mutable std::mutex mutexStop_;
	mutable std::mutex mutexAccept_;
};

std::shared_ptr<LocalMapping> LocalMapping::Create(Map* map, bool monocular)
{
	return std::make_shared<LocalMappingImpl>(map, monocular);
}

} //namespace ORB_SLAM
