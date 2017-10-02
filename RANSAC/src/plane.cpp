#include "acq/impl/primitive.hpp"

namespace acq {

    /// ------- computeScore() ------
    void Plane::computeScore(Eigen::Matrix3d var, DecoratedCloud& cloud, double T, double alpha) {

        // --- Compute the Plane Inliers ---
        Eigen::MatrixXi inliers_idx =  this->computeInliers(cloud,T,alpha) ;
        if(inliers_idx.rows() > 0) {
            this->setInliers_idx(inliers_idx);


            // --- Estimate the density of our plane ---
            //double inliersDensity = inliers_idx.rows()/this->findInliersBoundingBox(var, cloud, inliers_idx);

            // --- Compute the plane score ---
            double density_max = 150, score = 0.0;
            int inliers_min = 40;
            const int n = inliers_idx.rows();
            if(n > inliers_min){
                score = 80.0 + 0.2*(100.0 - (std::abs(density_max - n)) /
                                double(std::max(density_max, double(n))) * 100.0);
            }

            // --- Set the score for this primitive ---
            this->setScore(score);

        }
    }

    /// ------- computeInliers() ------
    Eigen::MatrixXi Plane::computeInliers(DecoratedCloud& cloud, double T, double alpha) {

        int numberPoint = cloud.getVertices().rows();
        Eigen::MatrixXi inliers_idx(numberPoint, 1);

        Eigen::Matrix<double, 1, 3> N = this->getNormal().normalized();
        Eigen::Matrix<double, 1, 3> P = this->getRefPoint();


        if( N.norm() > 0 && numberPoint > 0) {
            const long n = cloud.getVertices().rows();

            Eigen::Matrix<double, 1, 3> _V, _N;

            int idx_counter = 0; double dist = 0;
            for (int i = 0; i < n; i++) {
                _V = cloud.getVertices().row(i);
                _N = cloud.getNormals().row(i).normalized();
                if(_N.dot(N) < 0) _N = -_N;

                // --- Check if in range and if normals match ---
                dist = std::abs((N.dot(_V - P)) / N.norm());
                if (dist < T && std::abs(_N.dot(N)) > alpha) {
                    inliers_idx(idx_counter, 0) = i;
                    idx_counter++;
                }
            }

            if (idx_counter == 0) inliers_idx = inliers_idx.topRows(1);
            else inliers_idx = inliers_idx.topRows(idx_counter - 1);
        }

        return inliers_idx ;
    }

    Primitive* Plane::clone(){
        Primitive* thisPlane = new Plane(this->getRefPoint(), this->getNormal()) ;
        thisPlane->setType(2) ;
        return thisPlane;
    }
}