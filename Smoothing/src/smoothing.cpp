#include "acq/smoothing.h"

namespace acq {

double findLambda(double pourcentage, DecoratedCloud &cloud) {
    double sizeBBDiag = igl::bounding_box_diagonal(cloud.getVertices()) ;
    double result = (pourcentage/100.0)*sizeBBDiag ;

    return result ; 
}

// ***********************$ Function to add noise and test smoothing ******************
void computeBoundingBox(float &Xmax,float & Xmin,float & Ymax,float & Ymin,float &Zmax, float & Zmin, DecoratedCloud& cloud) {
    Eigen::MatrixXd maximum(1, 3)  ;
    maximum = cloud.getVertices().colwise().maxCoeff() ;
    Xmax = maximum(0) ;
    Ymax= maximum(1) ;
    Zmax  = maximum(2) ;

    Eigen::MatrixXd minimum(1, 3)  ;
    minimum = cloud.getVertices().colwise().minCoeff() ;
    Xmin = minimum(0) ;
    Ymin = minimum(1) ;
    Zmin = minimum(2) ;
}

Eigen::MatrixXd addNoise(float noise, DecoratedCloud& cloud) {
        // compute the value of the boundingBox for the second cloud 
        float Xmax, Xmin, Ymax, Ymin, Zmax, Zmin ;
        computeBoundingBox(Xmax, Xmin, Ymax, Ymin,Zmax, Zmin, cloud) ;

        // set the variance to sigma = noise% of the bouding box size in each direction
        float sigmaX =(Xmax-Xmin)*noise  ;
        float sigmaY = (Ymax-Ymin)*noise ;
        float sigmaZ = (Zmax-Zmin)*noise ;

        int M = cloud.getVertices().rows() ;

        // initialization 
        Eigen::MatrixXd random(M, 3)  ;

         // construct noise 
          for(int i = 0; i< M; i++) {
             random(i, 0) = std::rand()*sigmaX /RAND_MAX ; 
             random(i, 1) = std::rand()*sigmaY /RAND_MAX ;
             random(i, 2) = std::rand()*sigmaZ /RAND_MAX ;
         }
        
        Eigen::MatrixXd newVertices(M,3) ;
        // add noise to the vertices 
        newVertices = cloud.getVertices() + random ;

        return newVertices ;
}

// *********************** Smoothing explicit  ******************
Eigen::MatrixXd explicitSmoothing(DecoratedCloud& cloud, double lambda, bool typeDiscretization) {
        int numbVertices = cloud.getVertices().rows() ;

        //****** Create the L matrix *******
        Eigen::SparseMatrix<double> L(numbVertices,numbVertices);

        if (typeDiscretization) L = computeCotanDiscretization(cloud) ;
        else L = uniformLaplacian(cloud) ;

        // ***** Move the vertices ******
        Eigen::MatrixXd vertices_new(numbVertices,3) ;   
        Eigen::SparseMatrix<double> eye(numbVertices,numbVertices) ;
        eye.setIdentity() ;

        vertices_new = (eye + L*lambda)*cloud.getVertices() ;

        return vertices_new ;
}

// *********************** Smoothing implicit  ******************
Eigen::MatrixXd implicitSmoothing(DecoratedCloud& cloud, double lambda) {
        int numbVertices = cloud.getVertices().rows() ;

        //****** Create the Lw matrix *******
        Eigen::SparseMatrix<double> Lw(numbVertices,numbVertices);
        Lw = weightCotan(cloud) ;

        //****** Create the M matrix *******
        Eigen::SparseMatrix<double> M(numbVertices,numbVertices);
        M = diagonalArea(cloud) ;

        // ***** Create the equation Ax = b ******
        Eigen::SparseMatrix<double> A(numbVertices,numbVertices);
        A = M - lambda*Lw ;

        Eigen::MatrixXd b(numbVertices,3) ;   
        b = M*cloud.getVertices() ;

        // ***** Compute the solver ******
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver ;
        solver.compute(A); 

        // **** Solve the system *****
        Eigen::MatrixXd vertices_new(numbVertices,3) ; 
        vertices_new = solver.solve(b).eval();
        return vertices_new ;
}

// ********* Function to test denoising *********
float computeError(DecoratedCloud &cloud1, DecoratedCloud &cloud2) {
        float error = (cloud1.getVertices()-cloud2.getVertices()).rowwise().norm().sum() ;
        return error ;
}


} // namespace acq