//
// Created by zhang on 2020/11/12.
//

#ifndef OPTIMIZER_LM_H
#define OPTIMIZER_LM_H
#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include "types.h"
namespace hw_lm {
class OptimizerLM {
public:
	OptimizerLM() = default;
	void solve(int it_nums);
	void addVertexSE3(VertexSE3 *);
	void addVertexPoint(VertexSBAPointXYZ *);
	void addEdge(EdgeSE3ProjectXYZ *);

public:
	int max_it = 10;
	int variable_dimension_;

	std::map<unsigned int, unsigned int> pose_in_hessian_to_id_;
	std::map<unsigned int, unsigned int> point_in_hessian_to_id_;
	std::map<unsigned int, unsigned int> pose_id_to_hessian_;
	std::map<unsigned int, unsigned int> point_id_to_hessian_;

	std::unordered_map<unsigned int, VertexSE3 *> pose_vertex_map_;
	std::unordered_map<unsigned int, VertexSBAPointXYZ *> point_vertex_map_;
	std::vector<EdgeSE3ProjectXYZ *> edge_vec_;
private:
	void buildHessian();
	double computeLambdaInit();
	double computeCh2();
	void computeError();
	void setLambda(double);
	void push();
	void pop();
	void update(Eigen::VectorXd& x);
	void storeHessianDiagonal();
	void restoreHessianDiagonal();
	Eigen::VectorXd solveX();

	unsigned int pose_id_ = 0;
	unsigned int point_id_ = 0;
	Eigen::MatrixXd hessian_;
	Eigen::VectorXd b_;
	Eigen::VectorXd hessian_diagonal_;
	std::set<std::pair<unsigned int, unsigned int>> recorder_set_;
};
}


#endif //OPTIMIZER_LM_H
