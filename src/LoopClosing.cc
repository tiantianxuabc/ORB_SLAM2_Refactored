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
		KeyFrame* matchedKF;
		cv::Mat Scw;
		g2o::Sim3 ScwG2O;
		std::vector<MapPoint*> matchedPoints;
		std::vector<MapPoint*> loopMapPoints;
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

	bool DetectLoop(KeyFrame* currentKF, std::vector<KeyFrame*>& candidateKFs, int lastLoopKFId)
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
		const std::vector<KeyFrame*> tmpCandidateKFs = keyFrameDB_->DetectLoopCandidates(currentKF, minScore);

		// If there are no loop candidates, just add new keyframe and return false
		if (tmpCandidateKFs.empty())
		{
			keyFrameDB_->add(currentKF);
			prevConsistentGroups_.clear();
			currentKF->SetErase();
			return false;
		}

		// For each loop candidate check consistency with previous loop candidates
		// Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
		// A group is consistent with a previous group if they share at least a keyframe
		// We must detect a consistent loop in several consecutive keyframes to accept it
		candidateKFs.clear();

		auto IsConsistent = [](const std::set<KeyFrame*>& prevGroup, const std::set<KeyFrame*>& currGroup)
		{
			for (KeyFrame* grounpKF : currGroup)
				if (prevGroup.count(grounpKF))
					return true;
			return false;
		};

		std::vector<ConsistentGroup> currConsistentGroups;
		std::vector<bool> consistentFound(prevConsistentGroups_.size(), false);
		for (KeyFrame* candidateKF : tmpCandidateKFs)
		{
			std::set<KeyFrame*> currGroup = candidateKF->GetConnectedKeyFrames();
			currGroup.insert(candidateKF);

			bool candidateFound = false;
			std::vector<size_t> consistentGroupsIds;
			for (size_t iG = 0; iG < prevConsistentGroups_.size(); iG++)
			{
				const std::set<KeyFrame*>& prevGroup = prevConsistentGroups_[iG].first;
				if (IsConsistent(prevGroup, currGroup))
					consistentGroupsIds.push_back(iG);
			}

			for (size_t iG : consistentGroupsIds)
			{
				const int currConsistency = prevConsistentGroups_[iG].second + 1;
				if (!consistentFound[iG])
				{
					currConsistentGroups.push_back(std::make_pair(currGroup, currConsistency));
					consistentFound[iG] = true; //this avoid to include the same group more than once
				}
				if (currConsistency >= minConsistency_ && !candidateFound)
				{
					candidateKFs.push_back(candidateKF);
					candidateFound = true; //this avoid to insert the same candidate more than once
				}
			}

			// If the group is not consistent with any previous group insert with consistency counter set to zero
			if (consistentGroupsIds.empty())
				currConsistentGroups.push_back(std::make_pair(currGroup, 0));
		}

		// Update Covisibility Consistent Groups
		prevConsistentGroups_ = currConsistentGroups;

		// Add Current Keyframe to database
		keyFrameDB_->add(currentKF);

		if (candidateKFs.empty())
		{
			currentKF->SetErase();
			return false;
		}

		return true;
	}

	static bool FindLoopInCandidateKFs(KeyFrame* currentKF, std::vector<KeyFrame*>& candidateKFs, Loop& loop, bool fixScale)
	{
		// For each consistent loop candidate we try to compute a Sim3

		const int ninitialCandidates = static_cast<int>(candidateKFs.size());

		// We compute first ORB matches for each candidate
		// If enough matches are found, we setup a Sim3Solver
		ORBmatcher matcher(0.75f, true);

		std::vector<Sim3Solver*> solvers;
		solvers.resize(ninitialCandidates);

		std::vector<vector<MapPoint*>> vmatches;
		vmatches.resize(ninitialCandidates);

		std::vector<bool> discarded;
		discarded.resize(ninitialCandidates);

		int ncandidates = 0; //candidates with enough matches

		for (int i = 0; i < ninitialCandidates; i++)
		{
			KeyFrame* candidateKF = candidateKFs[i];

			// avoid that local mapping erase it while it is being processed in this thread
			candidateKF->SetNotErase();

			if (candidateKF->isBad())
			{
				discarded[i] = true;
				continue;
			}

			const int nmatches = matcher.SearchByBoW(currentKF, candidateKF, vmatches[i]);
			if (nmatches < 20)
			{
				discarded[i] = true;
				continue;
			}
			else
			{
				Sim3Solver* solver = new Sim3Solver(currentKF, candidateKF, vmatches[i], fixScale);
				solver->SetRansacParameters(0.99, 20, 300);
				solvers[i] = solver;
			}

			ncandidates++;
		}

		// Perform alternatively RANSAC iterations for each candidate
		// until one is succesful or all fail
		while (ncandidates > 0)
		{
			for (int i = 0; i < ninitialCandidates; i++)
			{
				if (discarded[i])
					continue;

				KeyFrame* candidateKF = candidateKFs[i];

				// Perform 5 Ransac Iterations
				std::vector<bool> isInlier;
				int nInliers;
				bool bNoMore;

				Sim3Solver* solver = solvers[i];
				cv::Mat Scm = solver->iterate(5, bNoMore, isInlier, nInliers);

				// If Ransac reachs max. iterations discard keyframe
				if (bNoMore)
				{
					discarded[i] = true;
					ncandidates--;
				}

				// If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
				if (!Scm.empty())
				{
					std::vector<MapPoint*> matches(vmatches[i].size());
					for (size_t j = 0; j < isInlier.size(); j++)
						matches[j] = isInlier[j] ? vmatches[i][j] : nullptr;

					const cv::Mat R = solver->GetEstimatedRotation();
					const cv::Mat t = solver->GetEstimatedTranslation();
					const float s = solver->GetEstimatedScale();
					matcher.SearchBySim3(currentKF, candidateKF, matches, s, R, t, 7.5f);

					g2o::Sim3 Scm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
					const int nInliers = Optimizer::OptimizeSim3(currentKF, candidateKF, matches, Scm, 10, fixScale);

					// If optimization is succesful stop ransacs and continue
					if (nInliers >= 20)
					{
						g2o::Sim3 Smw(Converter::toMatrix3d(candidateKF->GetRotation()), Converter::toVector3d(candidateKF->GetTranslation()), 1.0);
						loop.matchedKF = candidateKF;
						loop.ScwG2O = Scm * Smw;
						loop.Scw = Converter::toCvMat(loop.ScwG2O);
						loop.matchedPoints = matches;
						return true;
					}
				}
			}
		}
		return false;
	}

	bool ComputeSim3(KeyFrame* currentKF, std::vector<KeyFrame*>& candidateKFs, Loop& loop)
	{
		const bool bMatch = FindLoopInCandidateKFs(currentKF, candidateKFs, loop, fixScale_);
		if (!bMatch)
		{
			for (KeyFrame* candidateKF : candidateKFs)
				candidateKF->SetErase();
			currentKF->SetErase();
			return false;
		}

		// Retrieve MapPoints seen in Loop Keyframe and neighbors
		std::vector<KeyFrame*> vpLoopConnectedKFs = loop.matchedKF->GetVectorCovisibleKeyFrames();
		vpLoopConnectedKFs.push_back(loop.matchedKF);
		loop.loopMapPoints.clear();
		for (vector<KeyFrame*>::iterator vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
		{
			KeyFrame* pKF = *vit;
			vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
			for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isBad() && pMP->mnLoopPointForKF != currentKF->mnId)
					{
						loop.loopMapPoints.push_back(pMP);
						pMP->mnLoopPointForKF = currentKF->mnId;
					}
				}
			}
		}

		// Find more matches projecting with the computed Sim3
		ORBmatcher matcher(0.75f, true);
		matcher.SearchByProjection(currentKF, loop.Scw, loop.loopMapPoints, loop.matchedPoints, 10);

		// If enough matches accept Loop
		int nTotalMatches = 0;
		for (size_t i = 0; i < loop.matchedPoints.size(); i++)
		{
			if (loop.matchedPoints[i])
				nTotalMatches++;
		}

		if (nTotalMatches >= 40)
		{
			for (KeyFrame* candidateKF : candidateKFs)
				if (candidateKF != loop.matchedKF)
					candidateKF->SetErase();
			return true;
		}
		else
		{
			for (KeyFrame* candidateKF : candidateKFs)
				candidateKF->SetErase();
			currentKF->SetErase();
			return false;
		}

	}

private:

	using ConsistentGroup = std::pair<std::set<KeyFrame*>, int>;

	KeyFrameDatabase* keyFrameDB_;
	ORBVocabulary* voc_;
	std::vector<ConsistentGroup> prevConsistentGroups_;
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

		KeyFrame* mpMatchedKF = loop.matchedKF;
		g2o::Sim3& mg2oScw = loop.ScwG2O;
		std::vector<MapPoint*>& mvpCurrentMatchedPoints = loop.matchedPoints;
		std::vector<MapPoint*>& mvpLoopMapPoints = loop.loopMapPoints;

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
