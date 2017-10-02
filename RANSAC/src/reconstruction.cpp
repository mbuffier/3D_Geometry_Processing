#include "acq/reconstruction.h"

namespace acq {
    /// ---- RECONSTRUCTION ----
    void reconstruct(PrimitiveManager& best_primitives, DecoratedCloud& cloud, int nbSamples, double T, double alpha, double T2) {
        const int n = best_primitives.getCloudSize();
        for (int i = 0; i < n; i++) { // For every primitive

            if (best_primitives.getPrimitive(i)->getType() == 2) { // If we found a plane
                Primitive *this_prim = best_primitives.getPrimitive(i);
                Eigen::MatrixXi inliers_idx = static_cast<Plane*>(this_prim)->computeInliers(cloud, T, alpha); // Retrieve the inliers
                sampleFromPrimitive(cloud, inliers_idx, this_prim, nbSamples, T2);
            }
        }
    }



    void sampleFromPrimitive(DecoratedCloud& cloud, Eigen::MatrixXi inliers_idx, Primitive* plane, int nbSample, double T) {
        /* Compute the optimal number of points for this plane, given the cloud variance and the inliers */

        Eigen::Matrix<double, 1000, 3> new_V, new_C, new_N, new_V_XY;

        // --- Find an orthonormal basis of the plane ---
        Eigen::Matrix<double, Eigen::Dynamic, 3> N(1, 3); N = static_cast<Plane*>(plane)->getNormal().normalized();
        Eigen::Matrix<double, Eigen::Dynamic, 3> u(1, 3);

        // Compute u so that it's orthogonal to N
        if(N(0,1)!=0 || N(0,0)!=0)
            u << -N(0,1), N(0,0), 0.0;
        else if(N(0,2)!=0 || N(0,0)!=0)
            u << -N(0,2), 0.0, N(0,0);
        else u << 0.0, -N(0,2), N(0,1);
        if(u.norm()!=0.0) u.normalize();

        // Compute v to make a basis (u,v,N)
        Eigen::MatrixXd v = u.row(0).cross(-N.row(0));
        if(v.norm()!=0.0) v.normalize();

        // --- Retrieve 3D Inliers and project on (u,v) basis ---
        const int n = inliers_idx.rows();
        Eigen::MatrixXd inliers2D(n, 2);
        Eigen::MatrixXd this_vertex;
        int idx;

        std::cout << "OKK3" << std::endl;


        for(int i = 0; i<n; i++){
            idx = inliers_idx(i,0);
            this_vertex = cloud.getVertices().row(idx);

            inliers2D(i,0) = u.row(0).dot(this_vertex.row(0));
            inliers2D(i,1) = v.row(0).dot(this_vertex.row(0));
        }

        // --- Compute Bounding Box Area ---
        double x_min = inliers2D.col(0).minCoeff();
        double x_max = inliers2D.col(0).maxCoeff();
        double y_min = inliers2D.col(1).minCoeff();
        double y_max = inliers2D.col(1).maxCoeff();


        // --- Generate new 2D samples in (u,v) coordinates ---
        srand(time(NULL));
        double this_x, this_y, f_x, f_y;
        double x, y, z;

        int current_idx = 0;

        for(int i = 0; i < nbSample; i++) {
            f_x = (double) rand() / RAND_MAX;
            f_y = (double) rand() / RAND_MAX;

            this_x = x_min + f_x * (x_max - x_min);
            this_y = y_min + f_y * (y_max - y_min);

            bool isValid = true;

            // Check if point is too close to pre-existing point in cloud
            for (int k = 0; k < inliers2D.rows(); k++) {
                if (sqrt(pow(inliers2D(k, 0) - this_x, 2.0) + pow(inliers2D(k, 1) - this_y, 2.0)) < T) {
                    isValid = false;
                    break;
                }
            }

            // Check if point is too close to preoivously added point
            for (int k = 0; k < current_idx; k++) {
                if (sqrt(pow(new_V_XY(k, 0) - this_x, 2.0) + pow(new_V_XY(k, 1) - this_y, 2.0)) < T) {
                    isValid = false;
                    break;
                }
            }


            // We found a valid sample, compute its world coordinates
            if (isValid) {
                x = this_x*u(0,0) +  this_y*v(0,0);
                y = this_x*u(0,1) +  this_y*v(0,1);
                z = this_x*u(0,2) +  this_y*v(0,2);

                Eigen::MatrixXd new_vertex(1,3); new_vertex << x, y, z;
                Eigen::MatrixXd new_vertex_XY(1,3); new_vertex_XY << this_x, this_y;
                Eigen::MatrixXd new_color(1,3); new_color << 1.0, 0.0, 0.0;
                new_vertex += static_cast<Plane*>(plane)->getRefPoint().dot(static_cast<Plane*>(plane)->getNormal().normalized())*static_cast<Plane*>(plane)->getNormal().normalized();

                new_V.row(current_idx) = new_vertex.row(0);
                new_V_XY.row(current_idx) = new_vertex_XY.row(0);
                new_C.row(current_idx) = new_color.row(0);
                new_N.row(current_idx) = static_cast<Plane*>(plane)->getNormal().normalized();

                current_idx++;

            }
        }


        if(current_idx > 0) {

            Eigen::MatrixXd V = new_V.topRows(current_idx);
            Eigen::MatrixXd C = new_C.topRows(current_idx);
            Eigen::MatrixXd N = new_N.topRows(current_idx);

            const int n_rows = current_idx + cloud.getVertices().rows();
            Eigen::MatrixXd joined_V(n_rows, 3), joined_C(n_rows, 3), joined_N(n_rows, 3);

            joined_V << cloud.getVertices(), V;
            joined_C << cloud.getColors()  , C;
            joined_N << cloud.getNormals() , N;

            cloud.setVertices(joined_V);
            cloud.setColors(joined_C);
            cloud.setNormals(joined_N);

        }
    }
}


