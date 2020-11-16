//
// Created by zhang on 2020/11/11.
//

#ifndef TYPES_H
#define TYPES_H
#include "edge.h"
#include "vertex.h"
#include "eigen_types.h"
#include "se3quat.h"
namespace hw_lm {
class VertexSE3: public vertex<6, ::g2o::SE3Quat> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
	VertexSE3() = default;
	void plus(const double *update) override {
		estimate_bk_ = estimate_;
		Eigen::Map<const Vector6> v(update);
		estimate_ = ::g2o::SE3Quat::exp(v) * estimate();
	}
};
class VertexSBAPointXYZ: public vertex<3, Vector3> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	VertexSBAPointXYZ() = default;
	void plus(const double *update) override {
		Eigen::Map<const Vector3> v(update);
		estimate_ += v;
	}
	void pop() override{ estimate_ = estimate_bk_;};
};
class EdgeSE3ProjectXYZ: public BinaryEdge<2, Vector2, VertexSBAPointXYZ, VertexSE3> {
public:
	void computeError() override {
		const VertexSE3 *v1 = vj_;
		VertexSBAPointXYZ *v2 = vi_;
		Vector2 obs(measurement_);
		error_ = obs - cam_project(v1->estimate().map(v2->estimate()));
	};
	double fx{}, fy{}, cx{}, cy{};
private:
	Vector2 cam_project(const Vector3 &trans_xyz) const;
	void linearizeOplus() override;

};
void EdgeSE3ProjectXYZ::linearizeOplus() {
	VertexSE3 *vj = vj_;
	const ::g2o::SE3Quat& T(vj->estimate());
	VertexSBAPointXYZ *vi = vi_;
	const Vector3& xyz = vi->estimate();
	Vector3 xyz_trans = T.map(xyz);

	number_t x = xyz_trans[0];
	number_t y = xyz_trans[1];
	number_t z = xyz_trans[2];
	number_t z_2 = z * z;

	Eigen::Matrix<number_t, 2, 3> tmp;
	tmp(0, 0) = fx;
	tmp(0, 1) = 0;
	tmp(0, 2) = -x / z * fx;

	tmp(1, 0) = 0;
	tmp(1, 1) = fy;
	tmp(1, 2) = -y / z * fy;

	jacobian_xi_ = -1. / z * tmp * T.rotation().toRotationMatrix();

	jacobian_xj_(0, 0) = x * y / z_2 * fx;
	jacobian_xj_(0, 1) = -(1 + (x * x / z_2)) * fx;
	jacobian_xj_(0, 2) = y / z * fx;
	jacobian_xj_(0, 3) = -1. / z * fx;
	jacobian_xj_(0, 4) = 0;
	jacobian_xj_(0, 5) = x / z_2 * fx;

	jacobian_xj_(1, 0) = (1 + y * y / z_2) * fy;
	jacobian_xj_(1, 1) = -x * y / z_2 * fy;
	jacobian_xj_(1, 2) = -x / z * fy;
	jacobian_xj_(1, 3) = 0;
	jacobian_xj_(1, 4) = -1. / z * fy;
	jacobian_xj_(1, 5) = y / z_2 * fy;
}
Vector2 EdgeSE3ProjectXYZ::cam_project(const Vector3 &trans_xyz) const {
	Vector2 proj = ::g2o::project(trans_xyz);
	Vector2 res;
	res[0] = proj[0] * fx + cx;
	res[1] = proj[1] * fy + cy;
	return res;
}
}
#endif //TYPES_H
