/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raul Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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

#include "Optimizer.h"

#include <mutex>

#include <Thirdparty/g2o/g2o/core/block_solver.h>
#include <Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h>
#include <Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h>
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>
#include <Thirdparty/g2o/g2o/core/robust_kernel_impl.h>
#include <Thirdparty/g2o/g2o/solvers/linear_solver_dense.h>
#include <Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h>

#include <Eigen/StdVector>

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

namespace ORB_SLAM2
{

static const double CHI2_MONO = 5.991;
static const double CHI2_STEREO = 7.815;
static const double DELTA_MONO = sqrt(CHI2_MONO);
static const double DELTA_STEREO = sqrt(CHI2_STEREO);

using VertexSE3 = g2o::VertexSE3Expmap;
using VertexSBA = g2o::VertexSBAPointXYZ;

template <template<class> class LinearSolver, class BlockSolver>
static void CreateOptimizer(g2o::SparseOptimizer& optimizer, double lambda = -1)
{
	using MatrixType = typename BlockSolver::PoseMatrixType;
	auto linearSolver = new LinearSolver<MatrixType>();
	auto solver = new BlockSolver(linearSolver);
	auto algorithm = new g2o::OptimizationAlgorithmLevenberg(solver);
	if (lambda >= 0)
		algorithm->setUserLambdaInit(lambda);
	optimizer.setAlgorithm(algorithm);
}

static VertexSE3* CreateVertexSE3(const VertexSE3::EstimateType& estimate, int id, bool fixed)
{
	VertexSE3* v = new VertexSE3();
	v->setEstimate(estimate);
	v->setId(id);
	v->setFixed(fixed);
	return v;
}

static VertexSBA* CreateVertexSBA(const VertexSBA::EstimateType& estimate, int id, bool fixed = false, bool marginalized = false)
{
	VertexSBA* v = new VertexSBA();
	v->setEstimate(estimate);
	v->setId(id);
	v->setMarginalized(marginalized);
	v->setFixed(fixed);
	return v;
}

template <class EDGE>
static void SetHuberKernel(EDGE* e, double delta)
{
	g2o::RobustKernelHuber* kernel = new g2o::RobustKernelHuber;
	kernel->setDelta(delta);
	e->setRobustKernel(kernel);
}

template <class EDGE>
static void SetMeasurement(EDGE* e, const cv::Point2f& pt)
{
	e->setMeasurement({ pt.x, pt.y });
}

template <class EDGE>
static void SetMeasurement(EDGE* e, const cv::Point2f& pt, float ur)
{
	e->setMeasurement({ pt.x, pt.y, ur });
}

template <int DIM, class EDGE>
static void SetInformation(EDGE* e, float invSigmaSq)
{
	e->setInformation(invSigmaSq * Eigen::Matrix<double, DIM, DIM>::Identity());
}

template <class EDGE>
static void SetCalibration(EDGE* e, const CameraParams& camera)
{
	e->fx = camera.fx;
	e->fy = camera.fy;
	e->cx = camera.cx;
	e->cy = camera.cy;
}

template <class EDGE>
static void SetCalibration(EDGE* e, const CameraParams& camera, float bf)
{
	SetCalibration(e, camera);
	e->bf = bf;
}

template <class EDGE>
static void SetXw(EDGE* e, const Point3D& Xw)
{
	e->Xw[0] = Xw(0);
	e->Xw[1] = Xw(1);
	e->Xw[2] = Xw(2);
}

template <class T> struct get_elem_type {};
template <> struct get_elem_type<Eigen::Matrix3d> { using type = double; };
template <> struct get_elem_type<CameraPose::Mat33> { using type = float; };
template <> struct get_elem_type<Eigen::Vector3d> { using type = double; };
template <> struct get_elem_type<CameraPose::Mat31> { using type = float; };

template <class R2, class R1>
static R2 ConvertRotation(const R1& src)
{
	using dst_elem_type = typename get_elem_type<R2>::type;
	R2 dst;
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) dst(i, j) = static_cast<dst_elem_type>(src(i, j));
	return dst;
}

template <class T2, class T1>
static T2 ConvertTranslation(const T1& src)
{
	using dst_elem_type = typename get_elem_type<T2>::type;
	T2 dst;
	for (int i = 0; i < 3; i++) dst(i) = static_cast<dst_elem_type>(src(i));
	return dst;
}

static CameraPose FromSE3Quat(const g2o::SE3Quat& quat)
{
	const auto R = ConvertRotation<CameraPose::Mat33>(quat.rotation().matrix());
	const auto t = ConvertTranslation<CameraPose::Mat31>(quat.translation());
	return CameraPose(R, t);
}

static g2o::SE3Quat ToSE3Quat(const CameraPose& pose)
{
	const auto R = ConvertRotation<Eigen::Matrix3d>(pose.R());
	const auto t = ConvertTranslation<Eigen::Vector3d>(pose.t());
	return g2o::SE3Quat(R, t);
}

static Eigen::Vector3d ToVector3d(const Point3D& v)
{
	return Eigen::Vector3d(v(0), v(1), v(2));
}

static Point3D FromVector3d(const Eigen::Vector3d& _v)
{
	Point3D v;
	for (int i = 0; i < 3; i++) v(i) = static_cast<float>(_v(i));
	return v;
}

static g2o::Sim3 ToG2OSim3(const Sim3& S)
{
	const Eigen::Matrix3d R = ConvertRotation<Eigen::Matrix3d>(S.R());
	const Eigen::Vector3d t = ConvertTranslation<Eigen::Vector3d>(S.t());
	return g2o::Sim3(R, t, static_cast<float>(S.Scale()));
}

static Sim3 FromG2OSim3(const g2o::Sim3& S)
{
	const CameraPose::Mat33 R = ConvertRotation<CameraPose::Mat33>(S.rotation().matrix());
	const CameraPose::Mat31 t = ConvertTranslation<CameraPose::Mat31>(S.translation());
	return Sim3(R, t, S.scale());
}

void Optimizer::GlobalBundleAdjustemnt(Map* map, int niterations, bool* stopFlag, frameid_t loopKFId, bool robust)
{
	std::vector<KeyFrame*> keyframes = map->GetAllKeyFrames();
	std::vector<MapPoint*> mappoints = map->GetAllMapPoints();
	BundleAdjustment(keyframes, mappoints, niterations, stopFlag, loopKFId, robust);
}

void Optimizer::BundleAdjustment(const std::vector<KeyFrame*>& keyframes, const std::vector<MapPoint*>& mappoints,
	int niterations, bool* stopFlag, frameid_t loopKFId, bool robust)
{
	g2o::SparseOptimizer optimizer;
	CreateOptimizer<g2o::LinearSolverEigen, g2o::BlockSolver_6_3>(optimizer);
	if (stopFlag)
		optimizer.setForceStopFlag(stopFlag);

	frameid_t maxKFId = 0;

	// Set KeyFrame vertices
	for (KeyFrame* keyframe : keyframes)
	{
		if (keyframe->isBad())
			continue;

		auto vertex = CreateVertexSE3(ToSE3Quat(keyframe->GetPose()), keyframe->id, keyframe->id == 0);
		optimizer.addVertex(vertex);
		maxKFId = std::max(maxKFId, keyframe->id);
	}

	// Set MapPoint vertices
	std::vector<bool> notIncludedMP;
	notIncludedMP.resize(mappoints.size());
	for (size_t i = 0; i < mappoints.size(); i++)
	{
		MapPoint* mappoint = mappoints[i];
		if (mappoint->isBad())
			continue;

		const int id = mappoint->id + maxKFId + 1;
		auto vertex = CreateVertexSBA(ToVector3d(mappoint->GetWorldPos()), id, false, true);
		optimizer.addVertex(vertex);

		int nedges = 0;
		//SET EDGES
		for (const auto& observation : mappoint->GetObservations())
		{
			KeyFrame* keyframe = observation.first;
			const size_t idx = observation.second;
			if (keyframe->isBad() || keyframe->id > maxKFId)
				continue;

			nedges++;

			const cv::KeyPoint& keypoint = keyframe->keypointsUn[idx];
			const float ur = keyframe->uright[idx];
			const float invSigmaSq = keyframe->pyramid.invSigmaSq[keypoint.octave];

			if (ur)
			{
				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

				e->setVertex(0, optimizer.vertex(id));
				e->setVertex(1, optimizer.vertex(keyframe->id));

				SetMeasurement(e, keypoint.pt);
				SetInformation<2>(e, invSigmaSq);
				if (robust)
					SetHuberKernel(e, DELTA_MONO);
				SetCalibration(e, keyframe->camera);

				optimizer.addEdge(e);
			}
			else
			{
				g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

				e->setVertex(0, optimizer.vertex(id));
				e->setVertex(1, optimizer.vertex(keyframe->id));

				SetMeasurement(e, keypoint.pt, ur);
				SetInformation<3>(e, invSigmaSq);
				if (robust)
					SetHuberKernel(e, DELTA_STEREO);
				SetCalibration(e, keyframe->camera, keyframe->camera.bf);

				optimizer.addEdge(e);
			}
		}

		if (nedges == 0)
		{
			optimizer.removeVertex(vertex);
			notIncludedMP[i] = true;
		}
		else
		{
			notIncludedMP[i] = false;
		}
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(niterations);

	// Recover optimized data

	//Keyframes
	for (KeyFrame* keyframe : keyframes)
	{
		if (keyframe->isBad())
			continue;

		const int id = keyframe->id;
		g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(id));
		if (loopKFId == 0)
		{
			keyframe->SetPose(FromSE3Quat(vertex->estimate()));
		}
		else
		{
			keyframe->TcwGBA = FromSE3Quat(vertex->estimate());
			keyframe->BAGlobalForKF = loopKFId;
		}
	}

	//Points
	for (size_t i = 0; i < mappoints.size(); i++)
	{
		MapPoint* mappoint = mappoints[i];

		if (notIncludedMP[i] || mappoint->isBad())
			continue;

		const int id = mappoint->id + maxKFId + 1;
		g2o::VertexSBAPointXYZ* vertex = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(id));

		if (loopKFId == 0)
		{
			mappoint->SetWorldPos(FromVector3d(vertex->estimate()));
			mappoint->UpdateNormalAndDepth();
		}
		else
		{
			mappoint->posGBA = FromVector3d(vertex->estimate());
			mappoint->BAGlobalForKF = loopKFId;
		}
	}
}

int Optimizer::PoseOptimization(Frame* frame)
{
	g2o::SparseOptimizer optimizer;
	CreateOptimizer<g2o::LinearSolverDense, g2o::BlockSolver_6_3>(optimizer);

	// Set Frame vertex
	auto vertex = CreateVertexSE3(ToSE3Quat(frame->pose), 0, false);
	optimizer.addVertex(vertex);

	// Set MapPoint vertices
	const int nkeypoints = frame->N;

	enum { EDGE_MONO = 0, EDGE_STEREO = 1 };
	std::vector<int> indices;
	std::vector<int> edgeTypes;
	std::vector<g2o::HyperGraph::Edge*> edges;

	{
		std::unique_lock<std::mutex> lock(MapPoint::GetGlobalMutex());

		for (int i = 0; i < nkeypoints; i++)
		{
			MapPoint* mappoint = frame->mappoints[i];
			if (!mappoint)
				continue;

			frame->outlier[i] = false;

			const cv::KeyPoint& keypoint = frame->keypointsUn[i];
			const float ur = frame->uright[i];
			const float invSigmaSq = frame->pyramid.invSigmaSq[keypoint.octave];

			// Monocular observation
			if (ur < 0)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

				e->setVertex(0, vertex);
				SetMeasurement(e, keypoint.pt);
				SetInformation<2>(e, invSigmaSq);
				SetHuberKernel(e, DELTA_MONO);
				SetCalibration(e, frame->camera);
				SetXw(e, mappoint->GetWorldPos());

				optimizer.addEdge(e);
				edges.push_back(e);
				edgeTypes.push_back(EDGE_MONO);
			}
			else  // Stereo observation
			{
				g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

				e->setVertex(0, vertex);
				SetMeasurement(e, keypoint.pt, ur);
				SetInformation<3>(e, invSigmaSq);
				SetHuberKernel(e, DELTA_STEREO);
				SetCalibration(e, frame->camera, frame->camera.bf);
				SetXw(e, mappoint->GetWorldPos());

				optimizer.addEdge(e);
				edges.push_back(e);
				edgeTypes.push_back(EDGE_STEREO);
			}

			indices.push_back(i);
		}
	}

	const int nedges = static_cast<int>(edges.size());
	if (nedges < 3)
		return 0;

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const int iterations = 10;
	const double maxChi2[2] = { CHI2_MONO, CHI2_STEREO };

	auto AsMonocular = [](g2o::HyperGraph::Edge* e) { return static_cast<g2o::EdgeSE3ProjectXYZOnlyPose*>(e); };
	auto AsStereo = [](g2o::HyperGraph::Edge* e) { return static_cast<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>(e); };

	int noutliers = 0;
	for (int k = 0; k < 4; k++)
	{
		vertex->setEstimate(ToSE3Quat(frame->pose));
		optimizer.initializeOptimization(0);
		optimizer.optimize(iterations);

		noutliers = 0;
		for (size_t i = 0; i < edges.size(); i++)
		{
			g2o::HyperGraph::Edge* e = edges[i];
			const int idx = indices[i];
			const int type = edgeTypes[i];
			const bool monocular = type == EDGE_MONO;

			if (frame->outlier[idx])
				monocular ? AsMonocular(e)->computeError() : AsStereo(e)->computeError();

			const double chi2 = monocular ? AsMonocular(e)->chi2() : AsStereo(e)->chi2();
			const bool outlier = chi2 > maxChi2[type];
			const int level = outlier ? 1 : 0;

			monocular ? AsMonocular(e)->setLevel(level) : AsStereo(e)->setLevel(level);

			frame->outlier[idx] = outlier;
			if (outlier)
				noutliers++;

			if (k == 2)
				monocular ? AsMonocular(e)->setRobustKernel(0) : AsStereo(e)->setRobustKernel(0);
		}
		if (optimizer.edges().size() < 10)
			break;
	}

	// Recover optimized pose and return number of inliers
	frame->SetPose(FromSE3Quat(vertex->estimate()));

	return nedges - noutliers;
}

void Optimizer::LocalBundleAdjustment(KeyFrame* currKeyFrame, bool* stopFlag, Map* map)
{
	// Local KeyFrames: First Breath Search from Current Keyframe
	std::list<KeyFrame*> localKFs;

	localKFs.push_back(currKeyFrame);
	currKeyFrame->BALocalForKF = currKeyFrame->id;

	for (KeyFrame* neighborKF : currKeyFrame->GetVectorCovisibleKeyFrames())
	{
		neighborKF->BALocalForKF = currKeyFrame->id;
		if (!neighborKF->isBad())
			localKFs.push_back(neighborKF);
	}

	// Local MapPoints seen in Local KeyFrames
	std::list<MapPoint*> localMPs;
	for (KeyFrame* localKF : localKFs)
	{
		for (MapPoint* mappoint : localKF->GetMapPointMatches())
		{
			if (!mappoint || mappoint->isBad())
				continue;

			if (mappoint->BALocalForKF != currKeyFrame->id)
			{
				localMPs.push_back(mappoint);
				mappoint->BALocalForKF = currKeyFrame->id;
			}
		}
	}

	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	std::list<KeyFrame*> fixedCameras;
	for (MapPoint* mappoint : localMPs)
	{
		for (const auto& observation : mappoint->GetObservations())
		{
			KeyFrame* fixedKF = observation.first;
			if (fixedKF->BALocalForKF != currKeyFrame->id && fixedKF->BAFixedForKF != currKeyFrame->id)
			{
				fixedKF->BAFixedForKF = currKeyFrame->id;
				if (!fixedKF->isBad())
					fixedCameras.push_back(fixedKF);
			}
		}
	}

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	CreateOptimizer<g2o::LinearSolverEigen, g2o::BlockSolver_6_3>(optimizer);

	if (stopFlag)
		optimizer.setForceStopFlag(stopFlag);

	frameid_t maxKFId = 0;

	// Set Local KeyFrame vertices
	for (KeyFrame* localKF : localKFs)
	{
		auto vertex = CreateVertexSE3(ToSE3Quat(localKF->GetPose()), localKF->id, localKF->id == 0);
		optimizer.addVertex(vertex);
		maxKFId = std::max(maxKFId, localKF->id);
	}

	// Set Fixed KeyFrame vertices
	for (KeyFrame* fixedKF : fixedCameras)
	{
		auto vertex = CreateVertexSE3(ToSE3Quat(fixedKF->GetPose()), fixedKF->id, true);
		optimizer.addVertex(vertex);
		maxKFId = std::max(maxKFId, fixedKF->id);
	}

	// Set MapPoint vertices
	const size_t expectedSize = (localKFs.size() + fixedCameras.size()) * localMPs.size();

	enum { EDGE_MONO = 0, EDGE_STEREO = 1 };
	std::vector<int> edgeTypes;
	std::vector<g2o::HyperGraph::Edge*> edges;
	std::vector<MapPoint*> mappoints;
	std::vector<KeyFrame*> keyframes;
	edges.reserve(expectedSize);
	mappoints.reserve(expectedSize);
	keyframes.reserve(expectedSize);

	for (MapPoint* mappoint : localMPs)
	{
		const int id = mappoint->id + maxKFId + 1;
		auto vertex = CreateVertexSBA(ToVector3d(mappoint->GetWorldPos()), id, false, true);
		optimizer.addVertex(vertex);

		//Set edges
		for (const auto& observation : mappoint->GetObservations())
		{
			KeyFrame* keyframe = observation.first;
			const size_t idx = observation.second;
			if (keyframe->isBad())
				continue;

			const cv::KeyPoint& keypoint = keyframe->keypointsUn[idx];
			const float ur = keyframe->uright[idx];
			const float invSigmaSq = keyframe->pyramid.invSigmaSq[keypoint.octave];

			// Monocular observation
			if (ur < 0)
			{
				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

				e->setVertex(0, optimizer.vertex(id));
				e->setVertex(1, optimizer.vertex(keyframe->id));

				SetMeasurement(e, keypoint.pt);
				SetInformation<2>(e, invSigmaSq);
				SetHuberKernel(e, DELTA_MONO);
				SetCalibration(e, keyframe->camera);

				optimizer.addEdge(e);
				edges.push_back(e);
				edgeTypes.push_back(EDGE_MONO);
			}
			else // Stereo observation
			{
				g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

				e->setVertex(0, optimizer.vertex(id));
				e->setVertex(1, optimizer.vertex(keyframe->id));

				SetMeasurement(e, keypoint.pt, ur);
				SetInformation<3>(e, invSigmaSq);
				SetHuberKernel(e, DELTA_STEREO);
				SetCalibration(e, keyframe->camera, keyframe->camera.bf);

				optimizer.addEdge(e);
				edges.push_back(e);
				edgeTypes.push_back(EDGE_STEREO);
			}

			mappoints.push_back(mappoint);
			keyframes.push_back(keyframe);
		}
	}

	if (stopFlag && *stopFlag)
		return;

	optimizer.initializeOptimization();
	optimizer.optimize(5);

	bool doMore = true;

	if (stopFlag && *stopFlag)
		doMore = false;

	auto AsMonocular = [](g2o::HyperGraph::Edge* e) { return static_cast<g2o::EdgeSE3ProjectXYZ*>(e); };
	auto AsStereo = [](g2o::HyperGraph::Edge* e) { return static_cast<g2o::EdgeStereoSE3ProjectXYZ*>(e); };
	const double maxChi2[2] = { CHI2_MONO, CHI2_STEREO };

	if (doMore)
	{
		// Check inlier observations
		for (size_t i = 0; i < edges.size(); i++)
		{
			if (mappoints[i]->isBad())
				continue;

			g2o::HyperGraph::Edge* e = edges[i];

			const int type = edgeTypes[i];
			const bool monocular = type == EDGE_MONO;

			const double chi2 = monocular ? AsMonocular(e)->chi2() : AsStereo(e)->chi2();
			const bool isDepthPositive = monocular ? AsMonocular(e)->isDepthPositive() : AsStereo(e)->isDepthPositive();

			if (chi2 > maxChi2[type] || !isDepthPositive)
			{
				monocular ? AsMonocular(e)->setLevel(1) : AsStereo(e)->setLevel(1);
			}

			monocular ? AsMonocular(e)->setRobustKernel(0) : AsStereo(e)->setRobustKernel(0);
		}

		// Optimize again without the outliers
		optimizer.initializeOptimization(0);
		optimizer.optimize(10);
	}

	std::vector<std::pair<KeyFrame*, MapPoint*>> toErase;
	toErase.reserve(edges.size());

	// Check inlier observations
	for (size_t i = 0; i < edges.size(); i++)
	{
		MapPoint* mappoint = mappoints[i];
		if (mappoint->isBad())
			continue;

		g2o::HyperGraph::Edge* e = edges[i];

		const int type = edgeTypes[i];
		const bool monocular = type == EDGE_MONO;

		const double chi2 = monocular ? AsMonocular(e)->chi2() : AsStereo(e)->chi2();
		const bool isDepthPositive = monocular ? AsMonocular(e)->isDepthPositive() : AsStereo(e)->isDepthPositive();

		if (chi2 > maxChi2[type] || !isDepthPositive)
		{
			KeyFrame* keyframe = keyframes[i];
			toErase.push_back(std::make_pair(keyframe, mappoint));
		}
	}

	// Get Map Mutex
	std::unique_lock<std::mutex> lock(map->mutexMapUpdate);

	if (!toErase.empty())
	{
		for (auto& erase : toErase)
		{
			KeyFrame* eraseKF = erase.first;
			MapPoint* eraseMP = erase.second;
			eraseKF->EraseMapPointMatch(eraseMP);
			eraseMP->EraseObservation(eraseKF);
		}
	}

	// Recover optimized data

	//Keyframes
	for (KeyFrame* localKF : localKFs)
	{
		const int id = localKF->id;
		g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(id));
		localKF->SetPose(FromSE3Quat(vertex->estimate()));
	}

	//Points
	for (MapPoint* localMP : localMPs)
	{
		const int id = localMP->id + maxKFId + 1;
		g2o::VertexSBAPointXYZ* vertex = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(id));
		localMP->SetWorldPos(FromVector3d(vertex->estimate()));
		localMP->UpdateNormalAndDepth();
	}
}

static std::pair<frameid_t, frameid_t> MakeMinMaxPair(frameid_t v1, frameid_t v2)
{
	return std::make_pair(std::min(v1, v2), std::max(v1, v2));
}

void Optimizer::OptimizeEssentialGraph(Map* map, KeyFrame* loopKF, KeyFrame* currKF,
	const KeyFrameAndPose& nonCorrectedSim3, const KeyFrameAndPose& correctedSim3,
	const LoopConnections& loopConnections, bool fixScale)
{
	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	CreateOptimizer<g2o::LinearSolverEigen, g2o::BlockSolver_7_3>(optimizer, 1e-16);
	optimizer.setVerbose(false);

	const std::vector<KeyFrame*> keyframes = map->GetAllKeyFrames();
	const std::vector<MapPoint*> mappoints = map->GetAllMapPoints();

	const frameid_t maxKFid = map->GetMaxKFid();

	std::vector<Sim3> nonCorrectedScw(maxKFid + 1);
	std::vector<Sim3> correctedSwc(maxKFid + 1);
	std::vector<g2o::VertexSim3Expmap*> vertices(maxKFid + 1);

	// Set KeyFrame vertices
	for (KeyFrame* keyframe : keyframes)
	{
		if (keyframe->isBad())
			continue;

		g2o::VertexSim3Expmap* vertex = new g2o::VertexSim3Expmap();

		const frameid_t id = keyframe->id;

		auto it = correctedSim3.find(keyframe);

		if (it != std::end(correctedSim3))
		{
			const Sim3& Scw = it->second;
			nonCorrectedScw[id] = Scw;
			vertex->setEstimate(ToG2OSim3(Scw));
		}
		else
		{
			Sim3 Scw(keyframe->GetPose());
			nonCorrectedScw[id] = Scw;
			vertex->setEstimate(ToG2OSim3(Scw));
		}

		if (keyframe == loopKF)
			vertex->setFixed(true);

		vertex->setId(id);
		vertex->setMarginalized(false);
		vertex->_fix_scale = fixScale;

		optimizer.addVertex(vertex);

		vertices[id] = vertex;
	}

	std::set<std::pair<frameid_t, frameid_t>> insertedEdges;

	const Eigen::Matrix<double, 7, 7> lambda = Eigen::Matrix<double, 7, 7>::Identity();

	// Set Loop edges
	const int minWeight = 100;
	for (const auto& connection : loopConnections)
	{
		KeyFrame* keyframe = connection.first;
		const frameid_t id1 = keyframe->id;
		const Sim3 Siw = nonCorrectedScw[id1];
		const Sim3 Swi = Siw.Inverse();

		for (KeyFrame* connectedKF : connection.second)
		{
			const frameid_t id2 = connectedKF->id;
			if ((id1 != currKF->id || id2 != loopKF->id) && keyframe->GetWeight(connectedKF) < minWeight)
				continue;

			const Sim3 Sjw = nonCorrectedScw[id2];
			const Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3* e = new g2o::EdgeSim3();
			e->setVertex(1, optimizer.vertex(id2));
			e->setVertex(0, optimizer.vertex(id1));
			e->setMeasurement(ToG2OSim3(Sji));
			e->information() = lambda;
			optimizer.addEdge(e);
			insertedEdges.insert(MakeMinMaxPair(id1, id2));
		}
	}

	// Set normal edges
	for (KeyFrame* keyframe : keyframes)
	{
		const frameid_t id1 = keyframe->id;
		const auto it1 = nonCorrectedSim3.find(keyframe);
		const Sim3 Siw = it1 != std::end(nonCorrectedSim3) ? it1->second : nonCorrectedScw[id1];
		const Sim3 Swi = Siw.Inverse();

		KeyFrame* parentKF = keyframe->GetParent();

		// Spanning tree edge
		if (parentKF)
		{
			const frameid_t id2 = parentKF->id;
			const auto it2 = nonCorrectedSim3.find(parentKF);
			const Sim3 Sjw = it2 != std::end(nonCorrectedSim3) ? it2->second : nonCorrectedScw[id2];
			const Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3* e = new g2o::EdgeSim3();
			e->setVertex(1, optimizer.vertex(id2));
			e->setVertex(0, optimizer.vertex(id1));
			e->setMeasurement(ToG2OSim3(Sji));
			e->information() = lambda;
			optimizer.addEdge(e);
		}

		// Loop edges
		const std::set<KeyFrame*> loopEdges = keyframe->GetLoopEdges();
		for (KeyFrame* loopEdge : loopEdges)
		{
			if (loopEdge->id >= keyframe->id)
				continue;

			const frameid_t id3 = loopEdge->id;
			const auto it3 = nonCorrectedSim3.find(loopEdge);
			const Sim3 Slw = it3 != std::end(nonCorrectedSim3) ? it3->second : nonCorrectedScw[id3];
			const Sim3 Sli = Slw * Swi;

			g2o::EdgeSim3* e = new g2o::EdgeSim3();
			e->setVertex(1, optimizer.vertex(id3));
			e->setVertex(0, optimizer.vertex(id1));
			e->setMeasurement(ToG2OSim3(Sli));
			e->information() = lambda;
			optimizer.addEdge(e);
		}

		// Covisibility graph edges
		for (KeyFrame* connectedKF : keyframe->GetCovisiblesByWeight(minWeight))
		{
			if (!connectedKF)
				continue;

			if (connectedKF == parentKF || keyframe->HasChild(connectedKF) || loopEdges.count(connectedKF))
				continue;

			if (connectedKF->isBad() || connectedKF->id >= keyframe->id)
				continue;

			if (insertedEdges.count(MakeMinMaxPair(keyframe->id, connectedKF->id)))
				continue;

			const frameid_t id4 = connectedKF->id;
			const auto it4 = nonCorrectedSim3.find(connectedKF);
			const Sim3 Snw = it4 != std::end(nonCorrectedSim3) ? it4->second : nonCorrectedScw[id4];
			const Sim3 Sni = Snw * Swi;

			g2o::EdgeSim3* e = new g2o::EdgeSim3();
			e->setVertex(1, optimizer.vertex(id4));
			e->setVertex(0, optimizer.vertex(id1));
			e->setMeasurement(ToG2OSim3(Sni));
			e->information() = lambda;
			optimizer.addEdge(e);
		}
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(20);

	std::unique_lock<std::mutex> lock(map->mutexMapUpdate);

	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
	for (KeyFrame* keyframe : keyframes)
	{
		const frameid_t id = keyframe->id;

		g2o::VertexSim3Expmap* vertex = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(id));
		const Sim3 correctedSiw = FromG2OSim3(vertex->estimate());
		correctedSwc[id] = correctedSiw.Inverse();
		const double invs = 1. / correctedSiw.Scale();
		const CameraPose Tiw(correctedSiw.R(), invs * correctedSiw.t());
		keyframe->SetPose(Tiw);
	}

	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for (MapPoint* mappoint : mappoints)
	{
		if (mappoint->isBad())
			continue;

		KeyFrame* referenceKF = mappoint->GetReferenceKeyFrame();
		const int id = mappoint->correctedByKF == currKF->id ? mappoint->correctedReference : referenceKF->id;

		const Sim3 Srw = nonCorrectedScw[id];
		const Sim3 correctedSwr = correctedSwc[id];
		const Sim3 correction = correctedSwr * Srw;

		const Point3D P3Dw = mappoint->GetWorldPos();
		const Point3D CorrectedP3Dw = correction.Map(P3Dw);
		mappoint->SetWorldPos(CorrectedP3Dw);
		mappoint->UpdateNormalAndDepth();
	}
}

int Optimizer::OptimizeSim3(KeyFrame* keyframe1, KeyFrame* keyframe2, std::vector<MapPoint*>& matches1, Sim3& S12,
	float maxChi2, bool fixScale)
{
	g2o::SparseOptimizer optimizer;
	CreateOptimizer<g2o::LinearSolverDense, g2o::BlockSolverX>(optimizer);

	// Calibration
	const CameraParams& camera1 = keyframe1->camera;
	const CameraParams& camera2 = keyframe2->camera;

	// Camera poses
	const auto R1w = keyframe1->GetPose().R();
	const auto t1w = keyframe1->GetPose().t();
	const auto R2w = keyframe2->GetPose().R();
	const auto t2w = keyframe2->GetPose().t();

	// Set Sim3 vertex
	g2o::VertexSim3Expmap * vertex = new g2o::VertexSim3Expmap();
	vertex->_fix_scale = fixScale;
	vertex->setEstimate(ToG2OSim3(S12));
	vertex->setId(0);
	vertex->setFixed(false);
	vertex->_principle_point1[0] = camera1.cx;
	vertex->_principle_point1[1] = camera1.cy;
	vertex->_focal_length1[0] = camera1.fx;
	vertex->_focal_length1[1] = camera1.fy;
	vertex->_principle_point2[0] = camera2.cx;
	vertex->_principle_point2[1] = camera2.cy;
	vertex->_focal_length2[0] = camera2.fx;
	vertex->_focal_length2[1] = camera2.fy;
	optimizer.addVertex(vertex);

	// Set MapPoint vertices
	const int nmatches = static_cast<int>(matches1.size());
	const std::vector<MapPoint*> mappoints1 = keyframe1->GetMapPointMatches();
	std::vector<g2o::EdgeSim3ProjectXYZ*> edges12;
	std::vector<g2o::EdgeInverseSim3ProjectXYZ*> edges21;
	std::vector<size_t> indices;

	indices.reserve(2 * nmatches);
	edges12.reserve(2 * nmatches);
	edges21.reserve(2 * nmatches);

	const double deltaHuber = sqrt(maxChi2);

	int ncorrespondences = 0;

	for (int i = 0; i < nmatches; i++)
	{
		MapPoint* mappoint1 = mappoints1[i];
		MapPoint* mappoint2 = matches1[i];

		if (!mappoint1 || !mappoint2)
			continue;

		const int i2 = mappoint2->GetIndexInKeyFrame(keyframe2);
		if (mappoint1->isBad() || mappoint2->isBad() || i2 < 0)
			continue;

		const int id1 = 2 * i + 1;
		const int id2 = 2 * (i + 1);

		const Point3D Xw1 = mappoint1->GetWorldPos();
		const Point3D Xc1 = R1w * Xw1 + t1w;
		optimizer.addVertex(CreateVertexSBA(ToVector3d(Xc1), id1, true));

		const Point3D Xw2 = mappoint2->GetWorldPos();
		const Point3D Xc2 = R2w * Xw2 + t2w;
		optimizer.addVertex(CreateVertexSBA(ToVector3d(Xc2), id2, true));

		ncorrespondences++;

		// Set edge x1 = S12*X2
		g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
		e12->setVertex(0, optimizer.vertex(id2));
		e12->setVertex(1, optimizer.vertex(0));

		const cv::KeyPoint& keypoint1 = keyframe1->keypointsUn[i];
		const float invSigmaSq1 = keyframe1->pyramid.invSigmaSq[keypoint1.octave];
		SetMeasurement(e12, keypoint1.pt);
		SetInformation<2>(e12, invSigmaSq1);
		SetHuberKernel(e12, deltaHuber);
		optimizer.addEdge(e12);

		// Set edge x2 = S21*X1
		g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();
		e21->setVertex(0, optimizer.vertex(id1));
		e21->setVertex(1, optimizer.vertex(0));

		const cv::KeyPoint& keypoint2 = keyframe2->keypointsUn[i2];
		const float invSigmaSq2 = keyframe2->pyramid.invSigmaSq[keypoint2.octave];
		SetMeasurement(e21, keypoint2.pt);
		SetInformation<2>(e21, invSigmaSq2);
		SetHuberKernel(e21, deltaHuber);
		optimizer.addEdge(e21);

		edges12.push_back(e12);
		edges21.push_back(e21);
		indices.push_back(i);
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(5);

	// Check inliers
	int nbad = 0;
	for (size_t i = 0; i < edges12.size(); i++)
	{
		g2o::EdgeSim3ProjectXYZ* e12 = edges12[i];
		g2o::EdgeInverseSim3ProjectXYZ* e21 = edges21[i];
		if (!e12 || !e21)
			continue;

		if (e12->chi2() > maxChi2 || e21->chi2() > maxChi2)
		{
			size_t idx = indices[i];
			matches1[idx] = nullptr;
			optimizer.removeEdge(e12);
			optimizer.removeEdge(e21);
			edges12[i] = nullptr;
			edges21[i] = nullptr;
			nbad++;
		}
	}

	if (ncorrespondences - nbad < 10)
		return 0;

	// Optimize again only with inliers
	const int iterations = nbad > 0 ? 10 : 5;
	optimizer.initializeOptimization();
	optimizer.optimize(iterations);

	int ninliers = 0;
	for (size_t i = 0; i < edges12.size(); i++)
	{
		g2o::EdgeSim3ProjectXYZ* e12 = edges12[i];
		g2o::EdgeInverseSim3ProjectXYZ* e21 = edges21[i];
		if (!e12 || !e21)
			continue;

		if (e12->chi2() > maxChi2 || e21->chi2() > maxChi2)
		{
			matches1[indices[i]] = nullptr;
		}
		else
		{
			ninliers++;
		}
	}

	// Recover optimized Sim3
	S12 = FromG2OSim3(vertex->estimate());

	return ninliers;
}

} //namespace ORB_SLAM
