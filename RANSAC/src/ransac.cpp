#include "acq/impl/ransac.hpp"

namespace acq {

    bool ransac(DecoratedCloud &cloud, PrimitiveManager &best_primitives, CloudManager &cloudManager, 
                double thresh, double alpha, double thresh_best, int iterationsTotal, int numberSample) {

            // ****************** INITIALISATION   ***********
            int numberOfPoint = cloud.getVertices().rows() ; 
            Eigen::Matrix<int, 3,1> thisSample ;
            Eigen::MatrixXi thisInliers ; 
            bool prim_detected = false, test_thisSphere, test_thisPlane, test ;
            int bestPrim_idx, nbAllPrim, i ;
            double best_score ;
            int newSize = 4, n_inliers ;
            bool primitiveFound = false ;
            // compute the variance 
            Eigen::Matrix3d variance = computeVariance(cloud.getVertices()) ;

            // will contain all the primitives created 
            PrimitiveManager allPrimitive ;

            // create the primitive for this iteration 
            for (i=0 ; i<iterationsTotal; i++) {
                // sample the right amount of point 
                for (int j=0; j<numberSample; j++) {
                    // sample the point 
                    thisSample = sample(numberOfPoint) ;

                    // test for the primitive, if they exist : add them in the cloud primitive 
                    computeSphere(thisSample, variance, cloud, allPrimitive, thresh, alpha) ;
                    computePlane(thisSample, variance, cloud, allPrimitive, thresh, alpha);
                }
                nbAllPrim = allPrimitive.getCloudSize() ;

                // if a primitive has been created in the turn 
                if (nbAllPrim>0) {
                    // get back the best primitive and its score
                    bestPrim_idx = allPrimitive.findBestScore() ;
                    Primitive* best_prim = allPrimitive.getPrimitive(bestPrim_idx) ;
                    best_score = best_prim->getScore() ;

                    // keep it if good enough
                    if (best_score > thresh_best) {
                        // the inliers of this cloud 
                        thisInliers = best_prim->computeInliers(cloud, thresh, alpha) ;                     
                        n_inliers = thisInliers.rows();

                        // extra test 
                        if(n_inliers > 1) {
                            // copy the primitive to store and add it to the newCloud                           
                            Primitive* prim_Storage = best_prim->clone() ;
                            best_primitives.addPrimitive(prim_Storage) ;

                            // clean the cloud and store the inliers in the cloud manager
                            cleanCloud(cloud, cloudManager, thisInliers) ;
                            numberOfPoint = cloud.getVertices().rows() ;
                            primitiveFound = true ;
                        }
                        // clean the primitive 
                        allPrimitive.deletePrimitive(bestPrim_idx) ;
                    }
                    // if the primitive isn't good enough, not take into account
                    else {
                        allPrimitive.deletePrimitive(bestPrim_idx) ;
                    }

                    // if there is not enough points in the cloud 
                    if (numberOfPoint < 3) {
                        break ;
                    }
                }
            }            
            // free the memory allocated with all the primitives not used 
            allPrimitive.clearAllPrimitives() ;

            // cloudManager and cloudPrimitive contains the result of the function
            return primitiveFound ; // Just return a bool if ransac finds something
    };

}


