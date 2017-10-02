#ifndef ACQ_EVALUATION_H
#define ACQ_EVALUATION_H

#include "acq/cloudManager.h"
#include "acq/primitiveManager.h"

#include <ANN/ANN.h>					// ANN declarations

namespace acq {
// ***********************$ Function to add noise and test smoothing ******************
void computeBoundingBox(float &Xmax,float & Xmin,float & Ymax,float & Ymin,float &Zmax, float & Zmin, DecoratedCloud& cloud) ;

// with 1 add noise on the vertices, with 2 add noise on the normals 
Eigen::MatrixXd addNoise(float noise, DecoratedCloud& cloud, int typeMatrix) ;


// ***********************$ Function to perform connected component algorithm ******************
// apply the algorithm to each cloud the plane of a cloudmanager
void connectedComponentManager(CloudManager& thisCloudManager, PrimitiveManager& best_primitives, double threshold) ;

// recursive connected component algorithm to separate distant part of a cloud 
void connectedComponent(DecoratedCloud& cloud, double threshold) ;

// label a given vertices if its neighbors aren't too far 
void labelVertices(Eigen::RowVector3d thisColor, ANNpointArray verticesArray, Eigen::MatrixXd& colors, 
                    int this_idx, Eigen::MatrixXd& visited, ANNkd_tree*	kdTree, double threshold, int connectivity) ;

// use to convert a matrix in an array for ANN librairy 
ANNpointArray matrixToANNArray(Eigen::MatrixXd const& points) ;
}

#endif

// do connected component 