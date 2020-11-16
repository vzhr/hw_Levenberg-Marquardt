//
// Created by zhang on 2020/11/12.
//

#include "optimizer_lm.h"
#include <fstream>
#include <Eigen/Sparse>
namespace hw_lm {
void OptimizerLM::solve(int it_nums) {
	buildHessian();
	double current_lambda = computeLambdaInit();
	double ni = 2;

	double current_chi2 = computeCh2();
	std::cout << "current_lambda: " << current_lambda << std::endl;
	std::cout << "current_chi2: " << current_chi2 << std::endl;

	for (int j = 0; j < it_nums; ++j) {

		buildHessian();
		storeHessianDiagonal();
		double rho = 0.;
		int qmax = 0;
		do {
			push();
			restoreHessianDiagonal();
			setLambda(current_lambda);
			Eigen::VectorXd deltaX = solveX();
//		std::cout << deltaX.transpose() << std::endl;
			update(deltaX);
			computeError();
			double temp_chi2 = computeCh2();
			rho = current_chi2 - temp_chi2;

			//compute scale
			double scale = 0.;
			for (int j = 0; j < deltaX.rows(); ++j) {
				scale += deltaX(j) * (current_lambda * deltaX(j) + b_(j));
			}
			scale += double (1e-3);
			rho /= scale;

			if (rho > 0){// the step is good
				double alpha = 1. - pow(2 * rho - 1, 3);
				alpha = std::min(alpha, 2./3);
				double scale_factor = std::max(1./3, alpha);
				current_lambda *= scale_factor;
				ni = 2;
				current_chi2 = temp_chi2;
			}
			else{
				current_lambda *=ni;
				ni *= 2;
				pop();
			}
			qmax++;
		}
		while (rho < 0 && qmax < 10);
		std::cout << "iteration: " << j\
		<< "\tcurrent_chi2: " << current_chi2\
		<< "\tlambda: " << current_lambda\
		<< "\tlevenbergIter: " << qmax\
		<< std::endl;
	}

}

void OptimizerLM::addVertexSE3(VertexSE3 *v) {
	pose_vertex_map_.emplace(v->Id(), v);
	if (!v->fixed()) {
		assert(v->Id() > -1 && "not set id!");
		unsigned int hessian_id = pose_id_++;
		pose_in_hessian_to_id_.emplace(hessian_id, v->Id());
		pose_id_to_hessian_.emplace(v->Id(), hessian_id);
	}
}
void OptimizerLM::addVertexPoint(VertexSBAPointXYZ *v) {
	point_vertex_map_.emplace(v->Id(), v);
	if (!v->fixed()) {
		unsigned int hessian_id = point_id_++;
		point_in_hessian_to_id_.emplace(hessian_id, v->Id());
		point_id_to_hessian_.emplace(v->Id(), hessian_id);
	}
}
void OptimizerLM::addEdge(EdgeSE3ProjectXYZ *edge) {
	edge_vec_.push_back(edge);
}
void OptimizerLM::buildHessian() {
	hessian_.resize(6 * pose_id_ + 3 * point_id_, 6 * pose_id_ + 3 * point_id_);
	hessian_.setZero();
	b_.resize(6 * pose_id_ + 3 * point_id_);
	b_.setZero();
	recorder_set_.clear();

	std::fstream f("hw_hessian.txt");
	for (const auto &e : edge_vec_) {
		e->computeError();
		e->linearize();
		if (!e->vj_->fixed()) {
			unsigned int pi = pose_id_to_hessian_.at(e->vj_->Id());
			hessian_.block<6, 6>(6 * pi, 6 * pi) +=
				e->jacobian_xj_.transpose() * e->information() * e->jacobian_xj_;
			b_.block<6, 1>(6 * pi, 0) -=
				e->jacobian_xj_.transpose() * e->information() * e->error();

			if (!e->vi_->fixed()) {
				unsigned int li = point_id_to_hessian_.at(e->vi_->Id());
				recorder_set_.emplace(pi, li);
				auto pl = e->jacobian_xj_.transpose() * e->information() * e->jacobian_xi_;
				hessian_.block<6, 3>(6 * pi, 6 * pose_id_ + 3 * li) += pl;
				hessian_.block<3, 6>(6 * pose_id_ + 3 * li, 6 * pi) += pl.transpose();
			}
		}
		if (!e->vi_->fixed()) {
			unsigned int li = point_id_to_hessian_.at(e->vi_->Id());
			hessian_.block<3, 3>(6 * pose_id_ + 3 * li, 6 * pose_id_ + 3 * li) +=
				e->jacobian_xi_.transpose() * e->information() * e->jacobian_xi_;
			b_.block<3, 1>(6 * pose_id_ + 3 * li, 0) -=
				e->jacobian_xi_.transpose() * e->information() * e->error();
		}

	}
//	f << hessian_.block(0, 0, 6 * pose_id_, 6 * pose_id_) << std::endl;

//	f<< "hessian endl" << std::endl;

}
double OptimizerLM::computeLambdaInit() {
	double max_diagonal = 0.;
	for (int i = 0; i < hessian_.cols(); ++i) {
		if (hessian_(i, i) > max_diagonal) max_diagonal = hessian_(i, i);
	}
	return 1e-5 * max_diagonal;
}
double OptimizerLM::computeCh2() {
	double chi2 = 0.;
	for (const auto &e : edge_vec_) {
		chi2 += e->chi2();
	}
	return chi2;
}
void OptimizerLM::setLambda(double lambda) {
	hessian_.diagonal().array() += lambda;
}
void OptimizerLM::push() {
	for (auto &it : pose_in_hessian_to_id_) {
		pose_vertex_map_.at(it.second)->push();
	}
	for (auto &it : point_in_hessian_to_id_) {
		point_vertex_map_.at(it.second)->push();
	}
}
void OptimizerLM::pop() {
	for (auto &it : pose_in_hessian_to_id_) {
		pose_vertex_map_.at(it.second)->pop();
	}
	for (auto &it : point_in_hessian_to_id_) {
		point_vertex_map_.at(it.second)->pop();
	}
}
Eigen::VectorXd OptimizerLM::solveX() {
	Eigen::SparseMatrix<double> B(6 * pose_id_, 6 * pose_id_);
	Eigen::SparseMatrix<double> E(6 * pose_id_, 3 * point_id_);
	Eigen::SparseMatrix<double> C_inv(3 * point_id_, 3 * point_id_);
	std::vector<Eigen::Triplet<double>> triplets;


	triplets.resize(6 * 6 * pose_id_);
	for (int i = 0; i < pose_id_; ++i) {
		for (int j = 0; j < 6; ++j) {
			for (int k = 0; k < 6; ++k) {
				triplets.emplace_back(6 * i + j,
									  6 * i + k,
									  hessian_(6 * i + j, 6 * i + k));
			}
		}
	}
	B.setFromTriplets(triplets.begin(), triplets.end());

	triplets.clear();
	triplets.resize(3 * 3 * point_id_);
	for (int i = 0; i < point_id_; ++i) {
		auto block_inv = hessian_.block<3, 3>(6 * pose_id_ + 3 * i, 6 * pose_id_ + 3 * i).inverse();
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				triplets.emplace_back(3 * i + j,
									  3 * i + k,
									  block_inv(j, k));
			}
		}
	}
	C_inv.setFromTriplets(triplets.begin(), triplets.end());

	triplets.clear();
	triplets.resize(recorder_set_.size() * 6 * 3);
	for (const auto &it : recorder_set_) {
		auto pi = it.first;
		auto li = it.second;
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 3; ++j) {
				triplets.emplace_back(6 * pi + i,
									  3 * li + j,
									  hessian_(6 * pi + i, 6 * pose_id_ + 3 * li + j));
			}
		}
	}
	E.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::SparseMatrix<double> E_C_inv = E * C_inv;
	Eigen::SparseMatrix<double> A = B - E_C_inv * E.transpose();
	Eigen::VectorXd b(6 * pose_id_);
	b = b_.block(0, 0, 6 * pose_id_, 1) - E_C_inv * b_.block(6 * pose_id_, 0, 3 * point_id_, 1);

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> ldlt;
	ldlt.compute(A);
	Eigen::VectorXd x(6 * pose_id_ + 3 * point_id_);
	x.block(0, 0, 6 * pose_id_, 1) = ldlt.solve(b);
	x.block(6 * pose_id_, 0, 3 * point_id_, 1) =
		C_inv * (b_.block(6 * pose_id_, 0, 3 * point_id_, 1)
			- E.transpose() * x.block(0, 0, 6 * pose_id_, 1));
	return x;
}
void OptimizerLM::update(Eigen::VectorXd &x) {
	for (auto &it : pose_in_hessian_to_id_) {
		pose_vertex_map_.at(it.second)->plus(x.data() + 6 * it.first);
	}
	for (auto &it : point_in_hessian_to_id_) {
		point_vertex_map_.at(it.second)->plus(x.data() + 6 * pose_id_ + 3 * it.first);
	}
}
void OptimizerLM::computeError() {
	for (auto& it : edge_vec_){
		it->computeError();
	}
}
void OptimizerLM::storeHessianDiagonal() {
	hessian_diagonal_.resize(hessian_.rows());
	hessian_diagonal_ = hessian_.diagonal();
}
void OptimizerLM::restoreHessianDiagonal() {
	hessian_.diagonal() = hessian_diagonal_;
}

}

