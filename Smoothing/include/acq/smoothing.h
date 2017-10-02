#ifndef SMOOTHING_H
#define SMOOTHING_H

#include "acq/decoratedCloud.h"
#include "acq/discreteCurvature.h"

// eigen solver
#include <Eigen/Sparse>
 #include<Eigen/IterativeLinearSolvers> 
#include<Eigen/SparseCholesky>

 // IGL function used 
#include "igl/jet.h"
#include "igl/unique.h"
#include "igl/triangle_triangle_adjacency.h"
#include "igl/bounding_box_diagonal.h"

namespace acq {
// ******* function to help to perform smoothing ********
double findLambda(double pourcentage, DecoratedCloud &cloud) ;

/// ******* function to produce noise to perform smoothing ********
Eigen::MatrixXd addNoise(float noise, DecoratedCloud& cloud) ;
void computeBoundingBox(float &Xmax,float & Xmin,float & Ymax,float & Ymin,float &Zmax, float & Zmin, DecoratedCloud& cloud);

// *********$ Function to perform smoothing *********
Eigen::MatrixXd explicitSmoothing(DecoratedCloud &cloud, double lambda, bool typeDiscretization) ;
Eigen::MatrixXd implicitSmoothing(DecoratedCloud& cloud, double lambda) ;

// *********$ Function to test denoising *********
float computeError(DecoratedCloud &cloud1, DecoratedCloud &cloud2) ;
}
#endif 