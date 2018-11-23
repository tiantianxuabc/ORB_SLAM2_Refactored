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

#include "KeyFrameDatabase.h"

#include <mutex>

#include <Thirdparty/DBoW2/DBoW2/BowVector.h>

#include "KeyFrame.h"

using namespace std;

#define LOCK_MUTEX_DATABASE() std::unique_lock<std::mutex> lock(mutex_);

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary &voc) : voc_(&voc)
{
	wordIdToKFs_.resize(voc.size());
}


void KeyFrameDatabase::add(KeyFrame* keyframe)
{
	LOCK_MUTEX_DATABASE();

	for (const auto& word : keyframe->mBowVec)
		wordIdToKFs_[word.first].push_back(keyframe);
}

void KeyFrameDatabase::erase(KeyFrame* keyframe)
{
	LOCK_MUTEX_DATABASE();

	// Erase elements in the Inverse File for the entry
	for (const auto& word : keyframe->mBowVec)
	{
		// List of keyframes that share the word
		std::list<KeyFrame*>& keyframes = wordIdToKFs_[word.first];
		auto it = std::find(std::begin(keyframes), std::end(keyframes), keyframe);
		if (it != std::end(keyframes))
			keyframes.erase(it);
	}
}

void KeyFrameDatabase::clear()
{
	wordIdToKFs_.clear();
	wordIdToKFs_.resize(voc_->size());
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* keyframe, float minScore)
{
	std::set<KeyFrame*> connectedKFs = keyframe->GetConnectedKeyFrames();
	std::list<KeyFrame*> wordSharingKFs;

	// Search all keyframes that share a word with current keyframes
	// Discard keyframes connected to the query keyframe
	{
		LOCK_MUTEX_DATABASE();

		for (const auto& word : keyframe->mBowVec)
		{
			for (KeyFrame* sharingKF : wordIdToKFs_[word.first])
			{
				if (sharingKF->mnLoopQuery != keyframe->mnId)
				{
					sharingKF->mnLoopWords = 0;
					if (!connectedKFs.count(sharingKF))
					{
						sharingKF->mnLoopQuery = keyframe->mnId;
						wordSharingKFs.push_back(sharingKF);
					}
				}
				sharingKF->mnLoopWords++;
			}
		}
	}

	if (wordSharingKFs.empty())
		return std::vector<KeyFrame*>();

	std::list<std::pair<float, KeyFrame*>> scoreAndMatches;

	// Only compare against those keyframes that share enough words
	int maxCommonWords = 0;
	//for (list<KeyFrame*>::iterator lit = sharingKFs.begin(), lend = sharingKFs.end(); lit != lend; lit++)
	for (KeyFrame* sharingKF : wordSharingKFs)
		maxCommonWords = std::max(maxCommonWords, sharingKF->mnLoopWords);

	const int minCommonWords = static_cast<float>(0.8f * maxCommonWords);

	int nscores = 0;

	// Compute similarity score. Retain the matches whose score is higher than minScore
	for (KeyFrame* sharingKF : wordSharingKFs)
	{
		if (sharingKF->mnLoopWords > minCommonWords)
		{
			nscores++;

			const float score = static_cast<float>(voc_->score(keyframe->mBowVec, sharingKF->mBowVec));
			sharingKF->mLoopScore = score;
			if (score >= minScore)
				scoreAndMatches.push_back(std::make_pair(score, sharingKF));
		}
	}

	if (scoreAndMatches.empty())
		return std::vector<KeyFrame*>();

	std::list<std::pair<float, KeyFrame*>> accScoreAndMatches;
	float bestAccScore = minScore;

	// Lets now accumulate score by covisibility
	//for (list<pair<float, KeyFrame*> >::iterator it = scoreAndMatches.begin(), itend = scoreAndMatches.end(); it != itend; it++)
	for (const auto& v : scoreAndMatches)
	{
		KeyFrame* sharingKF = v.second;

		float bestScore = v.first;
		float accScore = v.first;
		KeyFrame* bestKF = sharingKF;

		for (KeyFrame* neighborKF : sharingKF->GetBestCovisibilityKeyFrames(10))
		{
			if (neighborKF->mnLoopQuery == keyframe->mnId && neighborKF->mnLoopWords > minCommonWords)
			{
				accScore += neighborKF->mLoopScore;
				if (neighborKF->mLoopScore > bestScore)
				{
					bestKF = neighborKF;
					bestScore = neighborKF->mLoopScore;
				}
			}
		}

		accScoreAndMatches.push_back(std::make_pair(accScore, bestKF));
		bestAccScore = std::max(bestAccScore, accScore);
	}

	// Return all those keyframes with a score higher than 0.75*bestScore
	const float minScoreToRetain = 0.75f * bestAccScore;

	std::set<KeyFrame*> spAlreadyAddedKF;
	std::vector<KeyFrame*> vpLoopCandidates;
	vpLoopCandidates.reserve(accScoreAndMatches.size());

	std::set<KeyFrame*> candidateKFs;

	//for (list<pair<float, KeyFrame*> >::iterator it = accScoreAndMatches.begin(), itend = accScoreAndMatches.end(); it != itend; it++)
	for (const auto& v : accScoreAndMatches)
	{
		if (v.first > minScoreToRetain)
			candidateKFs.insert(v.second);
	}

	return std::vector<KeyFrame*>(std::begin(candidateKFs), std::end(candidateKFs));
}

vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
	list<KeyFrame*> lKFsSharingWords;

	// Search all keyframes that share a word with current frame
	{
		LOCK_MUTEX_DATABASE();

		for (DBoW2::BowVector::const_iterator vit = F->bowVector.begin(), vend = F->bowVector.end(); vit != vend; vit++)
		{
			list<KeyFrame*> &lKFs = wordIdToKFs_[vit->first];

			for (list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
			{
				KeyFrame* pKFi = *lit;
				if (pKFi->mnRelocQuery != F->id)
				{
					pKFi->mnRelocWords = 0;
					pKFi->mnRelocQuery = F->id;
					lKFsSharingWords.push_back(pKFi);
				}
				pKFi->mnRelocWords++;
			}
		}
	}
	if (lKFsSharingWords.empty())
		return vector<KeyFrame*>();

	// Only compare against those keyframes that share enough words
	int maxCommonWords = 0;
	for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
	{
		if ((*lit)->mnRelocWords > maxCommonWords)
			maxCommonWords = (*lit)->mnRelocWords;
	}

	int minCommonWords = maxCommonWords*0.8f;

	list<pair<float, KeyFrame*> > lScoreAndMatch;

	int nscores = 0;

	// Compute similarity score.
	for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
	{
		KeyFrame* pKFi = *lit;

		if (pKFi->mnRelocWords > minCommonWords)
		{
			nscores++;
			float si = voc_->score(F->bowVector, pKFi->mBowVec);
			pKFi->mRelocScore = si;
			lScoreAndMatch.push_back(make_pair(si, pKFi));
		}
	}

	if (lScoreAndMatch.empty())
		return vector<KeyFrame*>();

	list<pair<float, KeyFrame*> > lAccScoreAndMatch;
	float bestAccScore = 0;

	// Lets now accumulate score by covisibility
	for (list<pair<float, KeyFrame*> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
	{
		KeyFrame* pKFi = it->second;
		vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

		float bestScore = it->first;
		float accScore = bestScore;
		KeyFrame* pBestKF = pKFi;
		for (vector<KeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
		{
			KeyFrame* pKF2 = *vit;
			if (pKF2->mnRelocQuery != F->id)
				continue;

			accScore += pKF2->mRelocScore;
			if (pKF2->mRelocScore > bestScore)
			{
				pBestKF = pKF2;
				bestScore = pKF2->mRelocScore;
			}

		}
		lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
		if (accScore > bestAccScore)
			bestAccScore = accScore;
	}

	// Return all those keyframes with a score higher than 0.75*bestScore
	float minScoreToRetain = 0.75f*bestAccScore;
	set<KeyFrame*> spAlreadyAddedKF;
	vector<KeyFrame*> vpRelocCandidates;
	vpRelocCandidates.reserve(lAccScoreAndMatch.size());
	for (list<pair<float, KeyFrame*> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
	{
		const float &si = it->first;
		if (si > minScoreToRetain)
		{
			KeyFrame* pKFi = it->second;
			if (!spAlreadyAddedKF.count(pKFi))
			{
				vpRelocCandidates.push_back(pKFi);
				spAlreadyAddedKF.insert(pKFi);
			}
		}
	}

	return vpRelocCandidates;
}

} //namespace ORB_SLAM
