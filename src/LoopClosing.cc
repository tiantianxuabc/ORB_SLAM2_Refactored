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

#include "LoopClosing.h"

#include <mutex>
#include <thread>

#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "Map.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "Usleep.h"

#define LOCK_MUTEX_LOOP_QUEUE() std::unique_lock<std::mutex> lock1(mutexLoopQueue_);
#define LOCK_MUTEX_FINISH()     std::unique_lock<std::mutex> lock2(mutexFinish_);
#define LOCK_MUTEX_RESET()      std::unique_lock<std::mutex> lock3(mutexReset_);
#define LOCK_MUTEX_GLOBAL_BA()  std::unique_lock<std::mutex> lock4(mMutexGBA);

namespace ORB_SLAM2
{

class LoopDetector
{

public:

	struct Loop
	{
		KeyFrame* mpMatchedKF;
		cv::Mat mScw;
		g2o::Sim3 mg2oScw;
		std::vector<MapPoint*> mvpCurrentMatchedPoints;
		std::vector<MapPoint*> mvpLoopMapPoints;
	};

	LoopDetector(KeyFrameDatabase* keyframeDB, ORBVocabulary* voc, bool fixScale)
		: keyFrameDB_(keyframeDB), voc_(voc), fixScale_(fixScale), minConsistency_(3) {}

	bool Detect(KeyFrame* currentKF, Loop& loop, int lastLoopKFId)
	{
		std::vector<KeyFrame*> loopKFCandidates;
		if (!DetectLoop(currentKF, loopKFCandidates, lastLoopKFId))
			return false;

		if (!ComputeSim3(currentKF, loopKFCandidates, loop))
			return false;

		return true;
	}

	bool DetectLoop(KeyFrame* currentKF, std::vector<KeyFrame*>& loopKFCandidates, int lastLoopKFId)
	{
		//If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
		if (currentKF->mnId < lastLoopKFId + 10)
		{
			keyFrameDB_->add(currentKF);
			currentKF->SetErase();
			return false;
		}

		// Compute reference BoW similarity score
		// This is the lowest score to a connected keyframe in the covisibility graph
		// We will impose loop candidates to have a higher similarity than this
		float minScore = 1.f;
		for (KeyFrame* neighborKF : currentKF->GetVectorCovisibleKeyFrames())
		{
			if (neighborKF->isBad())
				continue;

			const float score = voc_->score(currentKF->mBowVec, neighborKF->mBowVec);
			minScore = std::min(minScore, score);
		}

		// Query the database imposing the minimum score
		vector<KeyFrame*> vpCandidateKFs = keyFrameDB_->DetectLoopCandidates(currentKF, minScore);

		// If there are no loop candidates, just add new keyframe and return false
		if (vpCandidateKFs.empty())
		{
			keyFrameDB_->add(currentKF);
			mvConsistentGroups.clear();
			currentKF->SetErase();
			return false;
		}

		// For each loop candidate check consistency with previous loop candidates
		// Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
		// A group is consistent with a previous group if they share at least a keyframe
		// We must detect a consistent loop in several consecutive keyframes to accept it
		loopKFCandidates.clear();

		vector<ConsistentGroup> vCurrentConsistentGroups;
		vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
		for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++)
		{
			KeyFrame* pCandidateKF = vpCandidateKFs[i];

			set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
			spCandidateGroup.insert(pCandidateKF);

			bool bEnoughConsistent = false;
			bool bConsistentForSomeGroup = false;
			for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
			{
				set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

				bool bConsistent = false;
				for (set<KeyFrame*>::iterator sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
				{
					if (sPreviousGroup.count(*sit))
					{
						bConsistent = true;
						bConsistentForSomeGroup = true;
						break;
					}
				}

				if (bConsistent)
				{
					int nPreviousConsistency = mvConsistentGroups[iG].second;
					int nCurrentConsistency = nPreviousConsistency + 1;
					if (!vbConsistentGroup[iG])
					{
						ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
						vCurrentConsistentGroups.push_back(cg);
						vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
					}
					if (nCurrentConsistency >= minConsistency_ && !bEnoughConsistent)
					{
						loopKFCandidates.push_back(pCandidateKF);
						bEnoughConsistent = true; //this avoid to insert the same candidate more than once
					}
				}
			}

			// If the group is not consistent with any previous group insert with consistency counter set to zero
			if (!bConsistentForSomeGroup)
			{
				ConsistentGroup cg = make_pair(spCandidateGroup, 0);
				vCurrentConsistentGroups.push_back(cg);
			}
		}

		// Update Covisibility Consistent Groups
		mvConsistentGroups = vCurrentConsistentGroups;


		// Add Current Keyframe to database
		keyFrameDB_->add(currentKF);

		if (loopKFCandidates.empty())
		{
			currentKF->SetErase();
			return false;
		}
		else
		{
			return true;
		}

		currentKF->SetErase();
		return false;
	}

	bool ComputeSim3(KeyFrame* mpCurrentKF, std::vector<KeyFrame*>& mvpEnoughConsistentCandidates, Loop& loop)
	{
		// For each consistent loop candidate we try to compute a Sim3

		const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

		// We compute first ORB matches for each candidate
		// If enough matches are found, we setup a Sim3Solver
		ORBmatcher matcher(0.75, true);

		vector<Sim3Solver*> vpSim3Solvers;
		vpSim3Solvers.resize(nInitialCandidates);

		vector<vector<MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nInitialCandidates);

		vector<bool> vbDiscarded;
		vbDiscarded.resize(nInitialCandidates);

		int nCandidates = 0; //candidates with enough matches

		for (int i = 0; i < nInitialCandidates; i++)
		{
			KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

			// avoid that local mapping erase it while it is being processed in this thread
			pKF->SetNotErase();

			if (pKF->isBad())
			{
				vbDiscarded[i] = true;
				continue;
			}

			int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

			if (nmatches < 20)
			{
				vbDiscarded[i] = true;
				continue;
			}
			else
			{
				Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], fixScale_);
				pSolver->SetRansacParameters(0.99, 20, 300);
				vpSim3Solvers[i] = pSolver;
			}

			nCandidates++;
		}

		bool bMatch = false;

		// Perform alternatively RANSAC iterations for each candidate
		// until one is succesful or all fail
		while (nCandidates > 0 && !bMatch)
		{
			for (int i = 0; i < nInitialCandidates; i++)
			{
				if (vbDiscarded[i])
					continue;

				KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

				// Perform 5 Ransac Iterations
				vector<bool> vbInliers;
				int nInliers;
				bool bNoMore;

				Sim3Solver* pSolver = vpSim3Solvers[i];
				cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

				// If Ransac reachs max. iterations discard keyframe
				if (bNoMore)
				{
					vbDiscarded[i] = true;
					nCandidates--;
				}

				// If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
				if (!Scm.empty())
				{
					vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
					for (size_t j = 0, jend = vbInliers.size(); j < jend; j++)
					{
						if (vbInliers[j])
							vpMapPointMatches[j] = vvpMapPointMatches[i][j];
					}

					cv::Mat R = pSolver->GetEstimatedRotation();
					cv::Mat t = pSolver->GetEstimatedTranslation();
					const float s = pSolver->GetEstimatedScale();
					matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

					g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
					const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, fixScale_);

					// If optimization is succesful stop ransacs and continue
					if (nInliers >= 20)
					{
						bMatch = true;
						loop.mpMatchedKF = pKF;
						g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
						loop.mg2oScw = gScm*gSmw;
						loop.mScw = Converter::toCvMat(loop.mg2oScw);

						loop.mvpCurrentMatchedPoints = vpMapPointMatches;
						break;
					}
				}
			}
		}

		if (!bMatch)
		{
			for (int i = 0; i < nInitialCandidates; i++)
				mvpEnoughConsistentCandidates[i]->SetErase();
			mpCurrentKF->SetErase();
			return false;
		}

		// Retrieve MapPoints seen in Loop Keyframe and neighbors
		vector<KeyFrame*> vpLoopConnectedKFs = loop.mpMatchedKF->GetVectorCovisibleKeyFrames();
		vpLoopConnectedKFs.push_back(loop.mpMatchedKF);
		loop.mvpLoopMapPoints.clear();
		for (vector<KeyFrame*>::iterator vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
		{
			KeyFrame* pKF = *vit;
			vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
			for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
					{
						loop.mvpLoopMapPoints.push_back(pMP);
						pMP->mnLoopPointForKF = mpCurrentKF->mnId;
					}
				}
			}
		}

		// Find more matches projecting with the computed Sim3
		matcher.SearchByProjection(mpCurrentKF, loop.mScw, loop.mvpLoopMapPoints, loop.mvpCurrentMatchedPoints, 10);

		// If enough matches accept Loop
		int nTotalMatches = 0;
		for (size_t i = 0; i < loop.mvpCurrentMatchedPoints.size(); i++)
		{
			if (loop.mvpCurrentMatchedPoints[i])
				nTotalMatches++;
		}

		if (nTotalMatches >= 40)
		{
			for (int i = 0; i < nInitialCandidates; i++)
				if (mvpEnoughConsistentCandidates[i] != loop.mpMatchedKF)
					mvpEnoughConsistentCandidates[i]->SetErase();
			return true;
		}
		else
		{
			for (int i = 0; i < nInitialCandidates; i++)
				mvpEnoughConsistentCandidates[i]->SetErase();
			mpCurrentKF->SetErase();
			return false;
		}

	}

private:

	using ConsistentGroup = std::pair<std::set<KeyFrame*>, int>;

	KeyFrameDatabase* keyFrameDB_;
	ORBVocabulary* voc_;
	std::vector<ConsistentGroup> mvConsistentGroups;
	bool fixScale_;
	int minConsistency_;
};

class ReusableThread
{
public:

	ReusableThread() : thread_(nullptr) {}
	~ReusableThread() { Join(); }

	template <class... Args>
	void Reset(Args&&... args)
	{
		Detach();
		thread_ = new std::thread(std::forward<Args>(args)...);
	}

	void Join()
	{
		if (!thread_ || !thread_->joinable())
			return;
		thread_->join();
		Clear();
	}

	void Detach()
	{
		if (!thread_ || !thread_->joinable())
			return;
		thread_->detach();
		Clear();
	}

	void Clear()
	{
		delete thread_;
		thread_ = nullptr;
	}

private:
	std::thread* thread_;
};

class GlobalBA
{
public:

	GlobalBA(Map* pMap) : mpMap(pMap), mpLocalMapper(nullptr), mbRunningGBA(false),
		mbFinishedGBA(true), mbStopGBA(false), mnFullBAIdx(0) {}

	void SetLocalMapper(LocalMapping *pLocalMapper)
	{
		mpLocalMapper = pLocalMapper;
	}

	// This function will run in a separate thread
	void _Run(int nLoopKF)
	{
		cout << "Starting Global Bundle Adjustment" << endl;

		int idx = mnFullBAIdx;
		Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);

		// Update all MapPoints and KeyFrames
		// Local Mapping was active during BA, that means that there might be new keyframes
		// not included in the Global BA and they are not consistent with the updated map.
		// We need to propagate the correction through the spanning tree
		{
			LOCK_MUTEX_GLOBAL_BA();
			if (idx != mnFullBAIdx)
				return;

			if (!mbStopGBA)
			{
				cout << "Global Bundle Adjustment finished" << endl;
				cout << "Updating map ..." << endl;
				mpLocalMapper->RequestStop();
				// Wait until Local Mapping has effectively stopped

				while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
				{
					usleep(1000);
				}

				// Get Map Mutex
				unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

				// Correct keyframes starting at map first keyframe
				list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

				while (!lpKFtoCheck.empty())
				{
					KeyFrame* pKF = lpKFtoCheck.front();
					const set<KeyFrame*> sChilds = pKF->GetChilds();
					cv::Mat Twc = pKF->GetPoseInverse();
					for (set<KeyFrame*>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++)
					{
						KeyFrame* pChild = *sit;
						if (pChild->mnBAGlobalForKF != nLoopKF)
						{
							cv::Mat Tchildc = pChild->GetPose()*Twc;
							pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
							pChild->mnBAGlobalForKF = nLoopKF;

						}
						lpKFtoCheck.push_back(pChild);
					}

					pKF->mTcwBefGBA = pKF->GetPose();
					pKF->SetPose(pKF->mTcwGBA);
					lpKFtoCheck.pop_front();
				}

				// Correct MapPoints
				const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

				for (size_t i = 0; i < vpMPs.size(); i++)
				{
					MapPoint* pMP = vpMPs[i];

					if (pMP->isBad())
						continue;

					if (pMP->mnBAGlobalForKF == nLoopKF)
					{
						// If optimized by Global BA, just update
						pMP->SetWorldPos(pMP->mPosGBA);
					}
					else
					{
						// Update according to the correction of its reference keyframe
						KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

						if (pRefKF->mnBAGlobalForKF != nLoopKF)
							continue;

						// Map to non-corrected camera
						cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
						cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
						cv::Mat Xc = Rcw*pMP->GetWorldPos() + tcw;

						// Backproject using corrected camera
						cv::Mat Twc = pRefKF->GetPoseInverse();
						cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
						cv::Mat twc = Twc.rowRange(0, 3).col(3);

						pMP->SetWorldPos(Rwc*Xc + twc);
					}
				}

				mpMap->InformNewBigChange();

				mpLocalMapper->Release();

				cout << "Map updated!" << endl;
			}

			mbFinishedGBA = true;
			mbRunningGBA = false;
		}
	}

	void Run(int nLoopKF)
	{
		mbRunningGBA = true;
		mbFinishedGBA = false;
		mbStopGBA = false;
		mpThreadGBA.Reset(&GlobalBA::_Run, this, nLoopKF);
	}

	void Stop()
	{
		LOCK_MUTEX_GLOBAL_BA();
		mbStopGBA = true;

		mnFullBAIdx++;
		mpThreadGBA.Detach();
	}

	bool isRunningGBA() const
	{
		unique_lock<std::mutex> lock(mMutexGBA);
		return mbRunningGBA;
	}

	bool isFinishedGBA() const
	{
		unique_lock<std::mutex> lock(mMutexGBA);
		return mbFinishedGBA;
	}

private:

	Map* mpMap;
	LocalMapping* mpLocalMapper;
	bool mbRunningGBA;
	bool mbFinishedGBA;
	bool mbStopGBA;
	int mnFullBAIdx;
	mutable std::mutex mMutexGBA;
	ReusableThread mpThreadGBA;
};

class LoopCorrector
{

private:

	Map* mpMap;
	LocalMapping* mpLocalMapper;
	GlobalBA* mGBA;
	// Fix scale in the stereo/RGB-D case
	bool mbFixScale;

public:

	LoopCorrector(Map* pMap, GlobalBA* pGBA, bool bFixScale)
		: mpMap(pMap), mGBA(pGBA), mbFixScale(bFixScale) {}

	void SetLocalMapper(LocalMapping *pLocalMapper)
	{
		mpLocalMapper = pLocalMapper;
	}

	void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, std::vector<MapPoint*>& mvpLoopMapPoints)
	{
		ORBmatcher matcher(0.8);

		for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++)
		{
			KeyFrame* pKF = mit->first;

			g2o::Sim3 g2oScw = mit->second;
			cv::Mat cvScw = Converter::toCvMat(g2oScw);

			vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(), static_cast<MapPoint*>(NULL));
			matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

			// Get Map Mutex
			unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
			const int nLP = mvpLoopMapPoints.size();
			for (int i = 0; i < nLP; i++)
			{
				MapPoint* pRep = vpReplacePoints[i];
				if (pRep)
				{
					pRep->Replace(mvpLoopMapPoints[i]);
				}
			}
		}
	}

	void Correct(KeyFrame* mpCurrentKF, LoopDetector::Loop& loop)
	{
		cout << "Loop detected!" << endl;

		KeyFrame* mpMatchedKF = loop.mpMatchedKF;
		g2o::Sim3& mg2oScw = loop.mg2oScw;
		std::vector<MapPoint*>& mvpCurrentMatchedPoints = loop.mvpCurrentMatchedPoints;
		std::vector<MapPoint*>& mvpLoopMapPoints = loop.mvpLoopMapPoints;

		// Send a stop signal to Local Mapping
		// Avoid new keyframes are inserted while correcting the loop
		mpLocalMapper->RequestStop();

		// If a Global Bundle Adjustment is running, abort it
		if (mGBA->isRunningGBA())
		{
			mGBA->Stop();
		}

		// Wait until Local Mapping has effectively stopped
		while (!mpLocalMapper->isStopped())
		{
			usleep(1000);
		}

		// Ensure current keyframe is updated
		mpCurrentKF->UpdateConnections();

		// Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
		std::vector<KeyFrame*> mvpCurrentConnectedKFs;
		mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
		mvpCurrentConnectedKFs.push_back(mpCurrentKF);

		KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
		CorrectedSim3[mpCurrentKF] = mg2oScw;
		cv::Mat Twc = mpCurrentKF->GetPoseInverse();


		{
			// Get Map Mutex
			unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

			for (vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
			{
				KeyFrame* pKFi = *vit;

				cv::Mat Tiw = pKFi->GetPose();

				if (pKFi != mpCurrentKF)
				{
					cv::Mat Tic = Tiw*Twc;
					cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
					cv::Mat tic = Tic.rowRange(0, 3).col(3);
					g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
					g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
					//Pose corrected with the Sim3 of the loop closure
					CorrectedSim3[pKFi] = g2oCorrectedSiw;
				}

				cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
				cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
				g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
				//Pose without correction
				NonCorrectedSim3[pKFi] = g2oSiw;
			}

			// Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
			for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;
				g2o::Sim3 g2oCorrectedSiw = mit->second;
				g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

				g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

				vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
				for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
				{
					MapPoint* pMPi = vpMPsi[iMP];
					if (!pMPi)
						continue;
					if (pMPi->isBad())
						continue;
					if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
						continue;

					// Project with non-corrected pose and project back with corrected pose
					cv::Mat P3Dw = pMPi->GetWorldPos();
					Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
					Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

					cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
					pMPi->SetWorldPos(cvCorrectedP3Dw);
					pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
					pMPi->mnCorrectedReference = pKFi->mnId;
					pMPi->UpdateNormalAndDepth();
				}

				// Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
				Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
				Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
				double s = g2oCorrectedSiw.scale();

				eigt *= (1. / s); //[R t/s;0 1]

				cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

				pKFi->SetPose(correctedTiw);

				// Make sure connections are updated
				pKFi->UpdateConnections();
			}

			// Start Loop Fusion
			// Update matched map points and replace if duplicated
			for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
			{
				if (mvpCurrentMatchedPoints[i])
				{
					MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
					MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
					if (pCurMP)
						pCurMP->Replace(pLoopMP);
					else
					{
						mpCurrentKF->AddMapPoint(pLoopMP, i);
						pLoopMP->AddObservation(mpCurrentKF, i);
						pLoopMP->ComputeDistinctiveDescriptors();
					}
				}
			}

		}

		// Project MapPoints observed in the neighborhood of the loop keyframe
		// into the current keyframe and neighbors using corrected poses.
		// Fuse duplications.
		SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);


		// After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
		map<KeyFrame*, set<KeyFrame*> > LoopConnections;

		for (vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

			// Update connections. Detect new links.
			pKFi->UpdateConnections();
			LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
			for (vector<KeyFrame*>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
			{
				LoopConnections[pKFi].erase(*vit_prev);
			}
			for (vector<KeyFrame*>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
			{
				LoopConnections[pKFi].erase(*vit2);
			}
		}

		// Optimize graph
		Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

		mpMap->InformNewBigChange();

		// Add loop edge
		mpMatchedKF->AddLoopEdge(mpCurrentKF);
		mpCurrentKF->AddLoopEdge(mpMatchedKF);

		// Launch a new thread to perform Global Bundle Adjustment
		mGBA->Run(mpCurrentKF->mnId);

		// Loop closed. Release Local Mapping.
		mpLocalMapper->Release();

		//mLastLoopKFid = mpCurrentKF->mnId;
	}
};

class LoopClosingImpl : public LoopClosing
{

public:

	LoopClosingImpl(Map *map, KeyFrameDatabase* keyframeDB, ORBVocabulary *voc, bool fixScale)
		: resetRequested_(false), finishRequested_(false), finished_(true), lastLoopKFId_(0),
		detector_(keyframeDB, voc, fixScale), GBA_(map), corrector_(map, &GBA_, fixScale)
	{
	}

	void SetTracker(Tracking* tracker) override
	{
		tracker_ = tracker;
	}

	void SetLocalMapper(LocalMapping* localMapper) override
	{
		localMapper_ = localMapper;
		GBA_.SetLocalMapper(localMapper);
		corrector_.SetLocalMapper(localMapper);
	}

	// Main function
	void Run() override
	{
		finished_ = false;

		while (true)
		{
			// Check if there are keyframes in the queue
			if (CheckNewKeyFrames())
			{
				KeyFrame* currentKF = nullptr;
				{
					LOCK_MUTEX_LOOP_QUEUE();
					currentKF = loopKeyFrameQueue_.front();
					loopKeyFrameQueue_.pop_front();
					currentKF->SetNotErase();
				}

				// Detect loop candidates and check covisibility consistency
				// Compute similarity transformation [sR|t]
				// In the stereo/RGBD case s=1
				LoopDetector::Loop loop;
				const bool found = detector_.Detect(currentKF, loop, lastLoopKFId_);
				if (found)
				{
					// Perform loop fusion and pose graph optimization
					corrector_.Correct(currentKF, loop);
					lastLoopKFId_ = currentKF->mnId;
				}
			}

			ResetIfRequested();

			if (CheckFinish())
				break;

			usleep(5000);
		}

		SetFinish();
	}

	void InsertKeyFrame(KeyFrame* keyframe) override
	{
		LOCK_MUTEX_LOOP_QUEUE();
		if (keyframe->mnId != 0)
			loopKeyFrameQueue_.push_back(keyframe);
	}

	void RequestReset() override
	{
		{
			LOCK_MUTEX_RESET();
			resetRequested_ = true;
		}

		while (true)
		{
			{
				LOCK_MUTEX_RESET();
				if (!resetRequested_)
					break;
			}
			usleep(5000);
		}
	}

	bool isRunningGBA() const override
	{
		return GBA_.isRunningGBA();
	}

	bool isFinishedGBA() const override
	{
		return GBA_.isFinishedGBA();
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

	bool CheckNewKeyFrames() const
	{
		LOCK_MUTEX_LOOP_QUEUE();
		return(!loopKeyFrameQueue_.empty());
	}

	void ResetIfRequested()
	{
		LOCK_MUTEX_RESET();
		if (resetRequested_)
		{
			loopKeyFrameQueue_.clear();
			lastLoopKFId_ = 0;
			resetRequested_ = false;
		}
	}

	bool CheckFinish() const
	{
		LOCK_MUTEX_FINISH();
		return finishRequested_;
	}

	void SetFinish()
	{
		LOCK_MUTEX_FINISH();
		finished_ = true;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

	bool resetRequested_;
	bool finishRequested_;
	bool finished_;

	Tracking* tracker_;
	LocalMapping *localMapper_;

	std::list<KeyFrame*> loopKeyFrameQueue_;

	// Loop detector variables
	LoopDetector detector_;

	long unsigned int lastLoopKFId_;

	// Variables related to Global Bundle Adjustment
	LoopCorrector corrector_;
	GlobalBA GBA_;

	mutable std::mutex mutexReset_;
	mutable std::mutex mutexFinish_;
	mutable std::mutex mutexLoopQueue_;
};

std::shared_ptr<LoopClosing> LoopClosing::Create(Map* map, KeyFrameDatabase* keyframeDB, ORBVocabulary* voc, bool fixScale)
{
	return std::make_shared<LoopClosingImpl>(map, keyframeDB, voc, fixScale);
}

} //namespace ORB_SLAM
