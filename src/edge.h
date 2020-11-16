//
// Created by zhang on 2020/11/10.
//

#ifndef EDGE_H
#define EDGE_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
namespace hw_lm {
template<int D, int VNumber, typename E>
class BaseEdge {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
	BaseEdge() = default;
	static const int dimension_ = D;
	using Measurement = E;
	using ErrorVector = Eigen::Matrix<double, D, 1, Eigen::ColMajor>;
	using InformationType = Eigen::Matrix<double, D, D, Eigen::ColMajor>;

	inline virtual void setMeasurement(const Measurement &m) { measurement_ = m; };
	inline const Measurement &measurement() const { return measurement_; };

	inline virtual void setInformation(const InformationType &info) { information_ = info; };
	inline const InformationType &information() const { return information_; };

	virtual void setVertex(int, void *) = 0;
	virtual void computeError() = 0;
	virtual void linearize() = 0;
	virtual double chi2() {return error().transpose() * information() * error();};
	const ErrorVector &error() const { return error_; }
	ErrorVector &error() { return error_; }
protected:
	Measurement measurement_;
	InformationType information_;
	ErrorVector error_;
	using JacobianType = Eigen::Matrix<double, D, VNumber, Eigen::ColMajor>;
	JacobianType jacobian_;

};
template<int D, typename E, typename VertexXi, typename VertexXj>
class BinaryEdge: public BaseEdge<D, VertexXi::dimension_ + VertexXj::dimension_, E> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	using VertexXiType = VertexXi;
	using VertexXjType = VertexXj;
	static const int Di = VertexXiType::dimension_;
	static const int Dj = VertexXjType::dimension_;
	using JacobianXiType = typename Eigen::Matrix<double, D, Di, Eigen::ColMajor>::AlignedMapType;
	using JacobianXjType = typename Eigen::Matrix<double, D, Dj, Eigen::ColMajor>::AlignedMapType;


	BinaryEdge()
		: BaseEdge<D, Di + Dj, E>(),
		  jacobian_xi_(nullptr, D, Di), jacobian_xj_(nullptr, D, Dj) {};
	using BaseEdge<D, Di + Dj, E>::jacobian_;
	using BaseEdge<D, Di + Dj, E>::dimension_;
	void linearize() override {
		new(&jacobian_xi_) JacobianXiType(jacobian_.block(0, 0, dimension_, Di).data(),
										  dimension_, Di);
		new(&jacobian_xj_) JacobianXjType(jacobian_.block(0, Di, dimension_, Dj).data(),
										  dimension_, Dj);
		linearizeOplus();
//		std::cout << " jacobianxi: " <<jacobian_xi_(0,0) << ", jacobian_: " << jacobian_(0,0) << std::endl;
	}
	void setVertex(int i, void *p) override {
		assert(i > -1 && i < 2 && "index out of bound");
		switch (i) {
			case 0: vi_ = (VertexXi *)p;
				break;
			case 1: vj_ = (VertexXj *)p;
				break;
			default: std::cerr << "setVertex index out bound." << std::endl;

		}

	};
	virtual void linearizeOplus() = 0;
	JacobianXiType jacobian_xi_;
	JacobianXjType jacobian_xj_;
	VertexXi *vi_;
	VertexXj *vj_;

};
};


#endif //EDGE_H
