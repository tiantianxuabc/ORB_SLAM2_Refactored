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

#include "Map.h"

#include <mutex>

#include "MapPoint.h"
#include "KeyFrame.h"

#define LOCK_MUTEX_MAP() std::unique_lock<std::mutex> lock(mutexMap_);

namespace ORB_SLAM2
{

Map::Map() : maxKFId_(0), bigChangeId_(0) {}

void Map::AddKeyFrame(KeyFrame* keyframe)
{
	LOCK_MUTEX_MAP();
	keyframes_.insert(keyframe);
	maxKFId_ = std::max(maxKFId_, keyframe->id);
}

void Map::AddMapPoint(MapPoint* mappoint)
{
	LOCK_MUTEX_MAP();
	mappoints_.insert(mappoint);
}

void Map::EraseMapPoint(MapPoint* mappoint)
{
	LOCK_MUTEX_MAP();
	mappoints_.erase(mappoint);

	// TODO: This only erase the pointer.
	// Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame* keyframe)
{
	LOCK_MUTEX_MAP();
	keyframes_.erase(keyframe);

	// TODO: This only erase the pointer.
	// Delete the MapPoint
}

void Map::SetReferenceMapPoints(const std::vector<MapPoint*>& mappoints)
{
	LOCK_MUTEX_MAP();
	referenceMapPoints_ = mappoints;
}

void Map::InformNewBigChange()
{
	LOCK_MUTEX_MAP();
	bigChangeId_++;
}

int Map::GetLastBigChangeIdx()
{
	LOCK_MUTEX_MAP();
	return bigChangeId_;
}

std::vector<KeyFrame*> Map::GetAllKeyFrames()
{
	LOCK_MUTEX_MAP();
	return std::vector<KeyFrame*>(keyframes_.begin(), keyframes_.end());
}

std::vector<MapPoint*> Map::GetAllMapPoints()
{
	LOCK_MUTEX_MAP();
	return std::vector<MapPoint*>(mappoints_.begin(), mappoints_.end());
}

long unsigned int Map::MapPointsInMap()
{
	LOCK_MUTEX_MAP();
	return mappoints_.size();
}

long unsigned int Map::KeyFramesInMap()
{
	LOCK_MUTEX_MAP();
	return keyframes_.size();
}

std::vector<MapPoint*> Map::GetReferenceMapPoints()
{
	LOCK_MUTEX_MAP();
	return referenceMapPoints_;
}

frameid_t Map::GetMaxKFid()
{
	LOCK_MUTEX_MAP();
	return maxKFId_;
}

void Map::clear()
{
	for (std::set<MapPoint*>::iterator sit = mappoints_.begin(), send = mappoints_.end(); sit != send; sit++)
		delete *sit;

	for (std::set<KeyFrame*>::iterator sit = keyframes_.begin(), send = keyframes_.end(); sit != send; sit++)
		delete *sit;

	mappoints_.clear();
	keyframes_.clear();
	maxKFId_ = 0;
	referenceMapPoints_.clear();
	mvpKeyFrameOrigins.clear();
}

} //namespace ORB_SLAM
