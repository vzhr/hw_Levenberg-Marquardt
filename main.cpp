#include <iostream>
#include <string>
#include <fstream>
#include "optimizer_lm.h"
int main(int agc, char **argv)
{
	std::string f_path{argv[1]};
	std::ifstream f(f_path);
	std::stringstream ss;
	if (!(f.is_open())) {
		std::cout << "file error" << std::endl;
		return EXIT_FAILURE;
	}

	float fx = 0, fy = 0, cx = 0, cy = 0;
	{
		char line[1024];
		f.getline(line, 1024);
		ss << line;
		ss >> fx >> fy >> cx >> cy;
	}
	printf("%f,%f,%f,%f\n", fx, fy, cx, cy);

	std::map<int, hw_lm::VertexSE3 *> kf_map;
	std::set<hw_lm::EdgeSE3ProjectXYZ *> kf_edge_set;
	std::map<int, ::g2o::VertexSBAPointXYZ *> mp_map;
	std::set<::g2o::EdgeSE3ProjectXYZ *> mp_edge_set;

	char line[1024];
	f.getline(line, 1024);
	while (std::strcmp(line, "kf-edges") != 0) {
		if (strlen(line) == 0) {
			f.getline(line, 1024);
			continue;
		}


		ss.clear();
		ss << line;
		int id = 0;
		ss >> id;

		Eigen::Matrix4d eigen_pose;
		eigen_pose.setIdentity();
		for (int i = 0; i < 4; ++i) {
			f.getline(line, 1024);
			ss.clear();
			ss << line;
			for (int j = 0; j < 4; ++j) {
				double tmp = 0;
				ss >> tmp;
				eigen_pose(i, j) = tmp;
			}
		}
//			std::cout << eigen_pose << std::endl;
		::g2o::SE3Quat pose_cw(eigen_pose.block<3, 3>(0, 0), eigen_pose.block<3, 1>(0, 3));
		auto kf_vtx = new hw_lm::VertexSE3();
		if (id < 0) {
			kf_vtx->setFixed(true);
			id = -id;
		}
		kf_vtx->setId(id);
		kf_vtx->setEstimate(pose_cw);

		kf_map.emplace(id, kf_vtx);
		f.getline(line, 1024);
	}
	f.getline(line, 1024);//skip title string
	while (std::strcmp(line, "mp-vs") != 0) {
		if (strlen(line) == 0) {
			f.getline(line, 1024);
			continue;
		}
		ss.clear();
		ss << line;
		int id1 = 0, id2 = 0;
		float weight = 0.f;
		ss >> id1 >> id2 >> weight;

		Eigen::Matrix4d eigen_pose;
		eigen_pose.setIdentity();
		for (int i = 0; i < 4; ++i) {
			f.getline(line, 1024);
			ss.clear();
			ss << line;
			for (int j = 0; j < 4; ++j) {
				double tmp = 0;
				ss >> tmp;
				eigen_pose(i, j) = tmp;
			}
		}

		//printf("%d,%d,%f\n", id1, id2, weight);
		//printf("%f,%f\n",  eigen_pose.block<3,1>(0,3).squaredNorm(), weight*eigen_pose.block<3,1>(0,3).squaredNorm());
		auto c1 = kf_map.find(id1);
		auto c2 = kf_map.find(id2);
		if (c1 != kf_map.end() && c2 != kf_map.end()) {
			Eigen::Matrix<double, 3, 3, Eigen::ColMajor> r = eigen_pose.block<3, 3>(0, 0);
			Eigen::Matrix<double, 3, 1, Eigen::ColMajor> t = eigen_pose.block<3, 1>(0, 3);
			const ::g2o::SE3Quat
				pose_c2c1(r, t);
//			auto edge_kf = new ::g2o::EdgeSE3Expmap();
			Eigen::Matrix<double, 6, 6>
				information = Eigen::Matrix<double, 6, 6>::Identity() * weight;
//				std::cout << information << std::endl;
//			edge_kf->setInformation(information);
//			edge_kf->setVertex(0, c1->second);
//			edge_kf->setVertex(1, c2->second);
//			edge_kf->setMeasurement(pose_c2c1);
//			kf_edge_set.insert(edge_kf);
		}
		f.getline(line, 1024);
	}

	int last_flag_id = -1;
	f.getline(line, 1024);
	ss.clear();
	ss << line;
	ss >> last_flag_id;
	while (f.good() && !f.eof()) {

		int mp_id = last_flag_id;
		float x = 0, y = 0, z = 0;
		ss >> x;
		f.getline(line, 1024);
		ss.clear();
		ss << line;
		ss >> y;
		f.getline(line, 1024);
		ss.clear();
		ss << line;
		ss >> z;
		auto mp_vtx = new hw_lm::VertexSBAPointXYZ();
		::g2o::Vector3 mp_pose(x, y, z);
		mp_vtx->setEstimate(mp_pose);
//		mp_vtx->setMarginalized(true);
		mp_vtx->setId(mp_id);
		mp_map.emplace(mp_id, mp_vtx);

		while (last_flag_id == mp_id) {
			int kf_id = 0;
			float u, v, weight;
			f.getline(line, 1024);
			ss.clear();
			ss << line;
			ss >> last_flag_id;
			if (last_flag_id != mp_id || f.eof() || !f.good()) break;

			ss >> kf_id >> u >> v >> weight;
			auto it = kf_map.find(kf_id);
			if (it != kf_map.end()) {
				auto mp_kf_edge = new ::g2o::EdgeSE3ProjectXYZ();
				::g2o::Vector2 measurement(u, v);
				::g2o::Matrix2 information = ::g2o::Matrix2::Identity() * weight;
				mp_kf_edge->setVertex(0, mp_vtx);
				mp_kf_edge->setVertex(1, it->second);
				mp_kf_edge->setMeasurement(measurement);
				mp_kf_edge->setInformation(information);
				mp_kf_edge->cx = cx;
				mp_kf_edge->cy = cy;
				mp_kf_edge->fx = fx;
				mp_kf_edge->fy = fy;
				mp_edge_set.insert(mp_kf_edge);
			}
		}
	}
	f.close();


	hw_lm::OptimizerLM optimizer;
	for (auto &it_kf : kf_map){
		optimizer.addVertexSE3(it_kf.second);
	}
	for (auto &it_mp: mp_map){
		optimizer.addVertexPoint(it_mp.second);
	}
	for (auto &it_edge : mp_edge_set){
		optimizer.addEdge(it_edge);
	}
	optimizer.solve(50);

	std::cout << "Hello World!" << std::endl;
}