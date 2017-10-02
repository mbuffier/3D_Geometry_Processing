        
#ifndef DISCRETECURVATURE_H
#define DISCRETECURVATURE_H

#include "acq/decoratedCloud.h"

#include <Eigen/Sparse>
 #include<Eigen/SparseCholesky>

#include "igl/jet.h"
#include "igl/unique.h"
#include "igl/triangle_triangle_adjacency.h"

#define _USE_MATH_DEFINES

namespace acq {

// ************* function to find area and angle of a triangle **************
double computeArea(Eigen::Vector3d &P1, Eigen::Vector3d &P2, Eigen::Vector3d &P3) ;
double computeAngle(Eigen::Vector3d &P1, Eigen::Vector3d &P2, Eigen::Vector3d &P3) ;
double computeCotan(Eigen::Vector3d &P1, Eigen::Vector3d &P2, Eigen::Vector3d &P3) ;

// ************* Function which return the L matrix for a mesh **************
// uniform discretization
Eigen::SparseMatrix<double> uniformLaplacian(DecoratedCloud & cloud) ;

// cotan discretization
Eigen::SparseMatrix<double> diagonalArea(DecoratedCloud & cloud) ;
Eigen::SparseMatrix<double> weightCotan(DecoratedCloud & cloud) ;
Eigen::SparseMatrix<double> inverseSparse(Eigen::SparseMatrix<double> M) ; 

Eigen::SparseMatrix<double> computeCotanDiscretization(DecoratedCloud & cloud) ;

// ************* Functions for curvature estimation **************
Eigen::MatrixXd meanCurvature(DecoratedCloud & cloud, int typeDiscretization) ;
Eigen::MatrixXd gaussianCurvUnifom(DecoratedCloud & cloud) ;
}
#endif 