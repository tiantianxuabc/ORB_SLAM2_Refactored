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

	for (const auto& word : keyframe->bowVector)
		wordIdToKFs_[word.first].push_back(keyframe);
}

void KeyFrameDatabase::erase(KeyFrame* keyframe)
{
	LOCK_MUTEX_DATABASE();

	// Erase elements in the Inverse File for the entry
	for (const auto& word : keyframe->bowVector)
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

		for (const auto& word : keyframe->bowVector)
		{
			for (KeyFrame* sharingKF : wordIdToKFs_[word.first])
			{
				if (sharingKF->loopQuery != keyframe->id)
				{
					sharingKF->loopWords = 0;
					if (!connectedKFs.count(sharingKF))
					{
						sharingKF->loopQuery = keyframe->id;
						wordSharingKFs.push_back(sharingKF);
					}
				}
				sharingKF->loopWords++;
			}
		}
	}

	if (wordSharingKFs.empty())
		return std::vector<KeyFrame*>();

	std::list<std::pair<float, KeyFrame*>> scoreAndMatches;

	// Only compare against those keyframes that share enough words
	int maxCommonWords = 0;
	for (KeyFrame* sharingKF : wordSharingKFs)
		maxCommonWords = std::max(maxCommonWords, sharingKF->loopWords);

	const int minCommonWords = static_cast<int>(0.8f * maxCommonWords);

	int nscores = 0;

	// Compute similarity score. Retain the matches whose score is higher than minScore
	for (KeyFrame* sharingKF : wordSharingKFs)
	{
		if (sharingKF->loopWords > minCommonWords)
		{
			nscores++;

			const float score = static_cast<float>(voc_->score(keyframe->bowVector, sharingKF->bowVector));
			sharingKF->loopScore = score;
			if (score >= minScore)
				scoreAndMatches.push_back(std::make_pair(score, sharingKF));
		}
	}

	if (scoreAndMatches.empty())
		return std::vector<KeyFrame*>();

	std::list<std::pair<float, KeyFrame*>> accScoreAndMatches;
	float bestAccScore = minScore;

	// Lets now accumulate score by covisibility
	for (const auto& v : scoreAndMatches)
	{
		KeyFrame* sharingKF = v.second;

		float bestScore = v.first;
		float accScore = v.first;
		KeyFrame* bestKF = sharingKF;

		for (KeyFrame* neighborKF : sharingKF->GetBestCovisibilityKeyFrames(10))
		{
			if (neighborKF->loopQuery == keyframe->id && neighborKF->loopWords > minCommonWords)
			{
				accScore += neighborKF->loopScore;
				if (neighborKF->loopScore > bestScore)
				{
					bestKF = neighborKF;
					bestScore = neighborKF->loopScore;
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
	for (const auto& v : accScoreAndMatches)
	{
		if (v.first > minScoreToRetain)
			candidateKFs.insert(v.second);
	}

	return std::vector<KeyFrame*>(std::begin(candidateKFs), std::end(candidateKFs));
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame* frame)
{
	std::list<KeyFrame*> wordSharingKFs;

	// Search all keyframes that share a word with current frame
	{
		LOCK_MUTEX_DATABASE();

		for (const auto& word : frame->bowVector)
		{
			for (KeyFrame* sharingKF : wordIdToKFs_[word.first])
			{
				if (sharingKF->relocQuery != frame->id)
				{
					sharingKF->relocWords = 0;
					sharingKF->relocQuery = frame->id;
					wordSharingKFs.push_back(sharingKF);
				}
				sharingKF->relocWords++;
			}
		}
	}

	if (wordSharingKFs.empty())
		return std::vector<KeyFrame*>();

	// Only compare against those keyframes that share enough words
	int maxCommonWords = 0;
	for (KeyFrame* sharingKF : wordSharingKFs)
		maxCommonWords = std::max(maxCommonWords, sharingKF->relocWords);

	const int minCommonWords = static_cast<int>(0.8f * maxCommonWords);

	std::list<std::pair<float, KeyFrame*>> scoreAndMatches;

	int nscores = 0;

	// Compute similarity score.
	for (KeyFrame* sharingKF : wordSharingKFs)
	{
		if (sharingKF->relocWords > minCommonWords)
		{
			nscores++;
			const float score = static_cast<float>(voc_->score(frame->bowVector, sharingKF->bowVector));
			sharingKF->relocScore = score;
			scoreAndMatches.push_back(std::make_pair(score, sharingKF));
		}
	}

	if (scoreAndMatches.empty())
		return std::vector<KeyFrame*>();

	std::list<std::pair<float, KeyFrame*>> accScoreAndMatches;
	float bestAccScore = 0;

	// Lets now accumulate score by covisibility
	for (const auto& v : scoreAndMatches)
	{
		KeyFrame* sharingKF = v.second;
		
		float bestScore = v.first;
		float accScore = bestScore;
		KeyFrame* bestKF = sharingKF;

		for (KeyFrame* neighborKF : sharingKF->GetBestCovisibilityKeyFrames(10))
		{
			if (neighborKF->relocQuery != frame->id)
				continue;

			accScore += neighborKF->relocScore;
			if (neighborKF->relocScore > bestScore)
			{
				bestKF = neighborKF;
				bestScore = neighborKF->relocScore;
			}
		}

		accScoreAndMatches.push_back(std::make_pair(accScore, bestKF));
		bestAccScore = std::max(bestAccScore, accScore);
	}

	// Return all those keyframes with a score higher than 0.75*bestScore
	const float minScoreToRetain = 0.75f * bestAccScore;

	std::set<KeyFrame*> candidateKFs;
	for (const auto& v : accScoreAndMatches)
	{
		if (v.first > minScoreToRetain)
			candidateKFs.insert(v.second);
	}

	return std::vector<KeyFrame*>(std::begin(candidateKFs), std::end(candidateKFs));
}

} //namespace ORB_SLAM
