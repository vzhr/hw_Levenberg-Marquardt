//
// Created by zhang on 2020/11/10.
//

#ifndef VERTEX_H
#define VERTEX_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
namespace hw_lm {
template<int D, typename T>
class vertex {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
	typedef T EstimateType;
	static const int dimension_ = D;
	void setEstimate(const EstimateType &et) {
		estimate_ = et;
		estimate_bk_ = estimate_;
	};
	inline const EstimateType &estimate() const { return estimate_; };
	inline int Id() const { return id_; };
	inline void setId(int id) { id_ = id; };
	virtual void plus(const double *update_) = 0;
	virtual void push() { estimate_bk_ = estimate_; }
	virtual void pop() { estimate_ = estimate_bk_; };
	void setFixed(bool fix) { fix_ = fix; };
	bool fixed() const { return fix_; };

protected:
	EstimateType estimate_;
	EstimateType estimate_bk_;
	int id_ = -2;//default for check
	bool fix_ = false;
};
}


#endif //VERTEX_H
