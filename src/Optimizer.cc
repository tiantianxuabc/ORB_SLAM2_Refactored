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
#include "Converter.h"

namespace ORB_SLAM2
{

static const double CHI2_MONO = 5.991;
static const double CHI2_STEREO = 7.815;
static const double DELTA_MONO = sqrt(CHI2_MONO);
static const double DELTA_STEREO = sqrt(CHI2_STEREO);

void Optimizer::GlobalBundleAdjustemnt(Map* map, int niterations, bool* stopFlag, frameid_t loopKFId, bool robust)
{
	std::vector<KeyFrame*> keyframes = map->GetAllKeyFrames();
	std::vector<MapPoint*> mappoints = map->GetAllMapPoints();
	BundleAdjustment(keyframes, mappoints, niterations, stopFlag, loopKFId, robust);
}

void Optimizer::BundleAdjustment(const std::vector<KeyFrame*>& keyframes, const std::vector<MapPoint*>& mappoints,
	int niterations, bool* stopFlag, frameid_t loopKFId, bool robust)
{
	vector<bool> vbNotIncludedMP;
	vbNotIncludedMP.resize(mappoints.size());

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	if (stopFlag)
		optimizer.setForceStopFlag(stopFlag);

	long unsigned int maxKFid = 0;

	// Set KeyFrame vertices
	for (size_t i = 0; i < keyframes.size(); i++)
	{
		KeyFrame* pKF = keyframes[i];
		if (pKF->isBad())
			continue;
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
		vSE3->setId(pKF->id);
		vSE3->setFixed(pKF->id == 0);
		optimizer.addVertex(vSE3);
		if (pKF->id > maxKFid)
			maxKFid = pKF->id;
	}

	const float thHuber2D = sqrt(5.99);
	const float thHuber3D = sqrt(7.815);

	// Set MapPoint vertices
	for (size_t i = 0; i < mappoints.size(); i++)
	{
		MapPoint* pMP = mappoints[i];
		if (pMP->isBad())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		const int id = pMP->id + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const map<KeyFrame*, size_t> observations = pMP->GetObservations();

		int nEdges = 0;
		//SET EDGES
		for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
		{

			KeyFrame* pKF = mit->first;
			if (pKF->isBad() || pKF->id > maxKFid)
				continue;

			nEdges++;

			const cv::KeyPoint &kpUn = pKF->keypointsUn[mit->second];

			if (pKF->uright[mit->second] < 0)
			{
				Eigen::Matrix<double, 2, 1> obs;
				obs << kpUn.pt.x, kpUn.pt.y;

				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->id)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKF->pyramid.invSigmaSq[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				if (robust)
				{
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuber2D);
				}

				e->fx = pKF->camera.fx;
				e->fy = pKF->camera.fy;
				e->cx = pKF->camera.cx;
				e->cy = pKF->camera.cy;

				optimizer.addEdge(e);
			}
			else
			{
				Eigen::Matrix<double, 3, 1> obs;
				const float kp_ur = pKF->uright[mit->second];
				obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

				g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->id)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKF->pyramid.invSigmaSq[kpUn.octave];
				Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
				e->setInformation(Info);

				if (robust)
				{
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuber3D);
				}

				e->fx = pKF->camera.fx;
				e->fy = pKF->camera.fy;
				e->cx = pKF->camera.cx;
				e->cy = pKF->camera.cy;
				e->bf = pKF->camera.bf;

				optimizer.addEdge(e);
			}
		}

		if (nEdges == 0)
		{
			optimizer.removeVertex(vPoint);
			vbNotIncludedMP[i] = true;
		}
		else
		{
			vbNotIncludedMP[i] = false;
		}
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(niterations);

	// Recover optimized data

	//Keyframes
	for (size_t i = 0; i < keyframes.size(); i++)
	{
		KeyFrame* pKF = keyframes[i];
		if (pKF->isBad())
			continue;
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->id));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		if (loopKFId == 0)
		{
			pKF->SetPose(Converter::toCvMat(SE3quat));
		}
		else
		{
			pKF->TcwGBA.create(4, 4, CV_32F);
			Converter::toCvMat(SE3quat).copyTo(pKF->TcwGBA);
			pKF->BAGlobalForKF = loopKFId;
		}
	}

	//Points
	for (size_t i = 0; i < mappoints.size(); i++)
	{
		if (vbNotIncludedMP[i])
			continue;

		MapPoint* pMP = mappoints[i];

		if (pMP->isBad())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->id + maxKFid + 1));

		if (loopKFId == 0)
		{
			pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
			pMP->UpdateNormalAndDepth();
		}
		else
		{
			pMP->posGBA.create(3, 1, CV_32F);
			Converter::toCvMat(vPoint->estimate()).copyTo(pMP->posGBA);
			pMP->BAGlobalForKF = loopKFId;
		}
	}

}

template <template<class> class LinearSolver, class BlockSolver>
static void CreateOptimizer(g2o::SparseOptimizer& optimizer)
{
	using MatrixType = typename BlockSolver::PoseMatrixType;
	auto linearSolver = new LinearSolver<MatrixType>();
	auto solver = new BlockSolver(linearSolver);
	auto algorithm = new g2o::OptimizationAlgorithmLevenberg(solver);
	optimizer.setAlgorithm(algorithm);
}

using VertexSE3 = g2o::VertexSE3Expmap;
static VertexSE3* CreateVertexSE3(const VertexSE3::EstimateType& estimate, int id, bool fixed)
{
	VertexSE3* v = new VertexSE3();
	v->setEstimate(estimate);
	v->setId(id);
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
static void SetXw(EDGE* e, const cv::Mat1f& Xw)
{
	e->Xw[0] = Xw(0);
	e->Xw[1] = Xw(1);
	e->Xw[2] = Xw(2);
}

int Optimizer::PoseOptimization(Frame* frame)
{
	g2o::SparseOptimizer optimizer;
	CreateOptimizer<g2o::LinearSolverDense, g2o::BlockSolver_6_3>(optimizer);

	// Set Frame vertex
	auto vertex = CreateVertexSE3(Converter::toSE3Quat(frame->pose.Tcw), 0, false);
	optimizer.addVertex(vertex);

	// Set MapPoint vertices
	const int nkeypoints = frame->N;

	enum { EDGE_MONO = 0, EDGE_STEREO = 1 };
	std::vector<int> indices;
	std::vector<int> edgeTypes;
	std::vector<g2o::HyperGraph::Edge*> edges;

	{
		unique_lock<mutex> lock(MapPoint::globalMutex_);

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
		vertex->setEstimate(Converter::toSE3Quat(frame->pose.Tcw));
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
	frame->SetPose(Converter::toCvMat(vertex->estimate()));

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
		auto vertex = CreateVertexSE3(Converter::toSE3Quat(localKF->GetPose()), localKF->id, localKF->id == 0);
		optimizer.addVertex(vertex);
		maxKFId = std::max(maxKFId, localKF->id);
	}

	// Set Fixed KeyFrame vertices
	for (KeyFrame* fixedKF : fixedCameras)
	{
		auto vertex = CreateVertexSE3(Converter::toSE3Quat(fixedKF->GetPose()), fixedKF->id, true);
		optimizer.addVertex(vertex);
		maxKFId = std::max(maxKFId, fixedKF->id);
	}

	// Set MapPoint vertices
	const int expectedSize = (localKFs.size() + fixedCameras.size()) * localMPs.size();

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
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(mappoint->GetWorldPos()));
		int id = mappoint->id + maxKFId + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		//Set edges
		for (const auto& observation : mappoint->GetObservations())
		{
			KeyFrame* keyframe = observation.first;
			const int idx = observation.second;
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
	unique_lock<mutex> lock(map->mutexMapUpdate);

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
		localKF->SetPose(Converter::toCvMat(vertex->estimate()));
	}

	//Points
	for (MapPoint* localMP : localMPs)
	{
		const int id = localMP->id + maxKFId + 1;
		g2o::VertexSBAPointXYZ* vertex = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(id));
		localMP->SetWorldPos(Converter::toCvMat(vertex->estimate()));
		localMP->UpdateNormalAndDepth();
	}
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
	const KeyFrameAndPose &NonCorrectedSim3,
	const KeyFrameAndPose &CorrectedSim3,
	const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
		new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
	g2o::BlockSolver_7_3 * solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

	solver->setUserLambdaInit(1e-16);
	optimizer.setAlgorithm(solver);

	const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
	const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

	const unsigned int nMaxKFid = pMap->GetMaxKFid();

	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid + 1);
	vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid + 1);
	vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid + 1);

	const int minFeat = 100;

	// Set KeyFrame vertices
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
	{
		KeyFrame* pKF = vpKFs[i];
		if (pKF->isBad())
			continue;
		g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

		const int nIDi = pKF->id;

		KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

		if (it != CorrectedSim3.end())
		{
			vScw[nIDi] = it->second;
			VSim3->setEstimate(it->second);
		}
		else
		{
			Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
			Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
			g2o::Sim3 Siw(Rcw, tcw, 1.0);
			vScw[nIDi] = Siw;
			VSim3->setEstimate(Siw);
		}

		if (pKF == pLoopKF)
			VSim3->setFixed(true);

		VSim3->setId(nIDi);
		VSim3->setMarginalized(false);
		VSim3->_fix_scale = bFixScale;

		optimizer.addVertex(VSim3);

		vpVertices[nIDi] = VSim3;
	}


	set<pair<long unsigned int, long unsigned int> > sInsertedEdges;

	const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

	// Set Loop edges
	for (map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
	{
		KeyFrame* pKF = mit->first;
		const long unsigned int nIDi = pKF->id;
		const set<KeyFrame*> &spConnections = mit->second;
		const g2o::Sim3 Siw = vScw[nIDi];
		const g2o::Sim3 Swi = Siw.inverse();

		for (set<KeyFrame*>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
		{
			const long unsigned int nIDj = (*sit)->id;
			if ((nIDi != pCurKF->id || nIDj != pLoopKF->id) && pKF->GetWeight(*sit) < minFeat)
				continue;

			const g2o::Sim3 Sjw = vScw[nIDj];
			const g2o::Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3* e = new g2o::EdgeSim3();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
			e->setMeasurement(Sji);

			e->information() = matLambda;

			optimizer.addEdge(e);

			sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
		}
	}

	// Set normal edges
	for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
	{
		KeyFrame* pKF = vpKFs[i];

		const int nIDi = pKF->id;

		g2o::Sim3 Swi;

		KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

		if (iti != NonCorrectedSim3.end())
			Swi = (iti->second).inverse();
		else
			Swi = vScw[nIDi].inverse();

		KeyFrame* pParentKF = pKF->GetParent();

		// Spanning tree edge
		if (pParentKF)
		{
			int nIDj = pParentKF->id;

			g2o::Sim3 Sjw;

			KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

			if (itj != NonCorrectedSim3.end())
				Sjw = itj->second;
			else
				Sjw = vScw[nIDj];

			g2o::Sim3 Sji = Sjw * Swi;

			g2o::EdgeSim3* e = new g2o::EdgeSim3();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
			e->setMeasurement(Sji);

			e->information() = matLambda;
			optimizer.addEdge(e);
		}

		// Loop edges
		const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
		for (set<KeyFrame*>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
		{
			KeyFrame* pLKF = *sit;
			if (pLKF->id < pKF->id)
			{
				g2o::Sim3 Slw;

				KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

				if (itl != NonCorrectedSim3.end())
					Slw = itl->second;
				else
					Slw = vScw[pLKF->id];

				g2o::Sim3 Sli = Slw * Swi;
				g2o::EdgeSim3* el = new g2o::EdgeSim3();
				el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->id)));
				el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
				el->setMeasurement(Sli);
				el->information() = matLambda;
				optimizer.addEdge(el);
			}
		}

		// Covisibility graph edges
		const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
		for (vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
		{
			KeyFrame* pKFn = *vit;
			if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
			{
				if (!pKFn->isBad() && pKFn->id < pKF->id)
				{
					if (sInsertedEdges.count(make_pair(min(pKF->id, pKFn->id), max(pKF->id, pKFn->id))))
						continue;

					g2o::Sim3 Snw;

					KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

					if (itn != NonCorrectedSim3.end())
						Snw = itn->second;
					else
						Snw = vScw[pKFn->id];

					g2o::Sim3 Sni = Snw * Swi;

					g2o::EdgeSim3* en = new g2o::EdgeSim3();
					en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->id)));
					en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
					en->setMeasurement(Sni);
					en->information() = matLambda;
					optimizer.addEdge(en);
				}
			}
		}
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(20);

	unique_lock<mutex> lock(pMap->mutexMapUpdate);

	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
	for (size_t i = 0; i < vpKFs.size(); i++)
	{
		KeyFrame* pKFi = vpKFs[i];

		const int nIDi = pKFi->id;

		g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
		g2o::Sim3 CorrectedSiw = VSim3->estimate();
		vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
		Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
		Eigen::Vector3d eigt = CorrectedSiw.translation();
		double s = CorrectedSiw.scale();

		eigt *= (1. / s); //[R t/s;0 1]

		cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

		pKFi->SetPose(Tiw);
	}

	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
	{
		MapPoint* pMP = vpMPs[i];

		if (pMP->isBad())
			continue;

		int nIDr;
		if (pMP->correctedByKF == pCurKF->id)
		{
			nIDr = pMP->correctedReference;
		}
		else
		{
			KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
			nIDr = pRefKF->id;
		}


		g2o::Sim3 Srw = vScw[nIDr];
		g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

		cv::Mat P3Dw = pMP->GetWorldPos();
		Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
		Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

		cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
		pMP->SetWorldPos(cvCorrectedP3Dw);

		pMP->UpdateNormalAndDepth();
	}
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

	g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// Calibration
	const cv::Mat &K1 = pKF1->camera.Mat();
	const cv::Mat &K2 = pKF2->camera.Mat();

	// Camera poses
	const cv::Mat R1w = pKF1->GetRotation();
	const cv::Mat t1w = pKF1->GetTranslation();
	const cv::Mat R2w = pKF2->GetRotation();
	const cv::Mat t2w = pKF2->GetTranslation();

	// Set Sim3 vertex
	g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();
	vSim3->_fix_scale = bFixScale;
	vSim3->setEstimate(g2oS12);
	vSim3->setId(0);
	vSim3->setFixed(false);
	vSim3->_principle_point1[0] = K1.at<float>(0, 2);
	vSim3->_principle_point1[1] = K1.at<float>(1, 2);
	vSim3->_focal_length1[0] = K1.at<float>(0, 0);
	vSim3->_focal_length1[1] = K1.at<float>(1, 1);
	vSim3->_principle_point2[0] = K2.at<float>(0, 2);
	vSim3->_principle_point2[1] = K2.at<float>(1, 2);
	vSim3->_focal_length2[0] = K2.at<float>(0, 0);
	vSim3->_focal_length2[1] = K2.at<float>(1, 1);
	optimizer.addVertex(vSim3);

	// Set MapPoint vertices
	const int N = vpMatches1.size();
	const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
	vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
	vector<size_t> vnIndexEdge;

	vnIndexEdge.reserve(2 * N);
	vpEdges12.reserve(2 * N);
	vpEdges21.reserve(2 * N);

	const float deltaHuber = sqrt(th2);

	int nCorrespondences = 0;

	for (int i = 0; i < N; i++)
	{
		if (!vpMatches1[i])
			continue;

		MapPoint* pMP1 = vpMapPoints1[i];
		MapPoint* pMP2 = vpMatches1[i];

		const int id1 = 2 * i + 1;
		const int id2 = 2 * (i + 1);

		const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

		if (pMP1 && pMP2)
		{
			if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0)
			{
				g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D1w = pMP1->GetWorldPos();
				cv::Mat P3D1c = R1w*P3D1w + t1w;
				vPoint1->setEstimate(Converter::toVector3d(P3D1c));
				vPoint1->setId(id1);
				vPoint1->setFixed(true);
				optimizer.addVertex(vPoint1);

				g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
				cv::Mat P3D2w = pMP2->GetWorldPos();
				cv::Mat P3D2c = R2w*P3D2w + t2w;
				vPoint2->setEstimate(Converter::toVector3d(P3D2c));
				vPoint2->setId(id2);
				vPoint2->setFixed(true);
				optimizer.addVertex(vPoint2);
			}
			else
				continue;
		}
		else
			continue;

		nCorrespondences++;

		// Set edge x1 = S12*X2
		Eigen::Matrix<double, 2, 1> obs1;
		const cv::KeyPoint &kpUn1 = pKF1->keypointsUn[i];
		obs1 << kpUn1.pt.x, kpUn1.pt.y;

		g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
		e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
		e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		e12->setMeasurement(obs1);
		const float &invSigmaSquare1 = pKF1->pyramid.invSigmaSq[kpUn1.octave];
		e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

		g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
		e12->setRobustKernel(rk1);
		rk1->setDelta(deltaHuber);
		optimizer.addEdge(e12);

		// Set edge x2 = S21*X1
		Eigen::Matrix<double, 2, 1> obs2;
		const cv::KeyPoint &kpUn2 = pKF2->keypointsUn[i2];
		obs2 << kpUn2.pt.x, kpUn2.pt.y;

		g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

		e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
		e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		e21->setMeasurement(obs2);
		float invSigmaSquare2 = pKF2->pyramid.invSigmaSq[kpUn2.octave];
		e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

		g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
		e21->setRobustKernel(rk2);
		rk2->setDelta(deltaHuber);
		optimizer.addEdge(e21);

		vpEdges12.push_back(e12);
		vpEdges21.push_back(e21);
		vnIndexEdge.push_back(i);
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(5);

	// Check inliers
	int nBad = 0;
	for (size_t i = 0; i < vpEdges12.size(); i++)
	{
		g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
		g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
		if (!e12 || !e21)
			continue;

		if (e12->chi2() > th2 || e21->chi2() > th2)
		{
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint*>(NULL);
			optimizer.removeEdge(e12);
			optimizer.removeEdge(e21);
			vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
			vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
			nBad++;
		}
	}

	int nMoreIterations;
	if (nBad > 0)
		nMoreIterations = 10;
	else
		nMoreIterations = 5;

	if (nCorrespondences - nBad < 10)
		return 0;

	// Optimize again only with inliers

	optimizer.initializeOptimization();
	optimizer.optimize(nMoreIterations);

	int nIn = 0;
	for (size_t i = 0; i < vpEdges12.size(); i++)
	{
		g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
		g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
		if (!e12 || !e21)
			continue;

		if (e12->chi2() > th2 || e21->chi2() > th2)
		{
			size_t idx = vnIndexEdge[i];
			vpMatches1[idx] = static_cast<MapPoint*>(NULL);
		}
		else
			nIn++;
	}

	// Recover optimized Sim3
	g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
	g2oS12 = vSim3_recov->estimate();

	return nIn;
}


} //namespace ORB_SLAM
