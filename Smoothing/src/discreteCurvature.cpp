#include "acq/discreteCurvature.h"

namespace acq {

// ******************* Functions to compute angle and area ***************

double Pi = 3.14159265359 ;

double computeArea(Eigen::Vector3d &P1, Eigen::Vector3d &P2, Eigen::Vector3d &P3) {
    Eigen::Vector3d vec1 = P2-P1 ;
    Eigen::Vector3d vec2 = P2-P3 ;
    Eigen::Vector3d vec3 = P3-P1 ;

    double side1 = vec1.norm() ;
    double side2 = vec2.norm() ;
    double side3 = vec3.norm() ;

    double halfP = (side1+side2+side3)/2.0 ;

    double area_value = sqrt(halfP*(halfP-side1)*(halfP-side2)*(halfP-side3)) ;
    return area_value ;
}

double computeAngle(Eigen::Vector3d &P1, Eigen::Vector3d &P2, Eigen::Vector3d &P3) {
    Eigen::Vector3d vec1 = P2-P1 ;
    Eigen::Vector3d vec2 = P3-P1 ;

    double norm1 = vec1.norm() ;
    double norm2 = vec2.norm() ;

    double angle = acos(vec1.dot(vec2)/(norm1*norm2)) ;
    return angle ;
}

double computeCotan(Eigen::Vector3d &P1, Eigen::Vector3d &P2, Eigen::Vector3d &P3) {
    Eigen::Vector3d vec1 = P2-P1 ;
    Eigen::Vector3d vec2 = P3-P1 ;

    double norm1 = vec1.norm() ;
    double norm2 = vec2.norm() ;

    double angle = acos(vec1.dot(vec2)/(norm1*norm2)) ;
    double cotan = tan(M_PI_2-angle) ;
    return cotan ; 
}

// ************* Function which return the L matrix for a mesh **************
//*************  uniform discretization **************

Eigen::SparseMatrix<double> uniformLaplacian(DecoratedCloud & cloud) {
        // initialization of matrices of faces, vertices and normals 
        int numbVertices = cloud.getVertices().rows() ;
        Eigen::MatrixXd normals(numbVertices, 3) ;
        Eigen::MatrixXd vertices(numbVertices, 3) ;
        Eigen::MatrixXi faces(numbVertices, 3) ;
        normals = cloud.getNormals() ;
        vertices = cloud.getVertices() ;
        faces = cloud.getFaces() ;

        // save the memory for the resulting matrix 
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(8*numbVertices);
        Eigen::SparseMatrix<double> L(numbVertices,numbVertices);

        // initialization for the function 
         std::vector<int> neighborIndexUnique ;
         bool findThisVer ;
         int position, valence ;
         float test ;

         // go through all the vertices of the mesh 
         for (int i=0; i < vertices.rows(); i++) {

             // go through all the face of the mesh 
             for (int j=0 ; j < faces.rows() ; j++ ) {
                 findThisVer = false ;
                 
                 // go throught the vertices of the face of find the current one 
                 for (int k = 0 ; k < 3 ; k++) {
                    if (faces(j,k) == i) {
                        position = k ;
                        findThisVer = true ;
                        break ;
                    }
                 }

                 // push back the neighbor vertex (all the faces in the same sens)
                 if (findThisVer) {
                    neighborIndexUnique.push_back(faces(j,(position +1)%3)) ;
                 }
             }

            valence = neighborIndexUnique.size() ;

            // fill in the matrix with the found neighboring vertices 
             for (int neigh =0; neigh < valence ; neigh++) {
                 tripletList.push_back(Eigen::Triplet<double>(i, neighborIndexUnique[neigh],1.0/double(valence)));
             }

             tripletList.push_back(Eigen::Triplet<double>(i, i,-1));
             neighborIndexUnique.clear() ;
         }
        L.setFromTriplets(tripletList.begin(), tripletList.end());

        return L ;
}

//*************  cotan discretization **************

Eigen::SparseMatrix<double> diagonalArea(DecoratedCloud & cloud) {
        // initialization
        int numbVertices = cloud.getVertices().rows() ;
        Eigen::MatrixXd vertices(numbVertices, 3) ;
        Eigen::MatrixXi faces(numbVertices, 3) ;
        vertices = cloud.getVertices() ;
        faces = cloud.getFaces() ;

        // reserve for the triplet list 
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(numbVertices);
        Eigen::SparseMatrix<double> M(numbVertices,numbVertices);

         float thisArea, areaTotal ;
         bool findThisVer ;
         int position ;
         Eigen::Vector3d P1, P2, P3 ;

        // loop over the vertices 
         for (int i=0; i < vertices.rows(); i++) {
             areaTotal = 0.0 ;
             // loop over the faces  
             for (int j=0 ; j < faces.rows() ; j++ ) {
                 findThisVer = false ;
                 for (int k = 0 ; k < 3 ; k++) {
                    if (faces(j,k) == i) {
                        position = k ;
                        findThisVer = true ;
                        break ;
                    }
                 }

                 // if the vertex is in the faces, compute the area of the triangle 
                 if (findThisVer) {
                     // the vertices of this triangle 
                    P1 = vertices.row(i) ;
                    P2 = vertices.row(faces(j,(position +1)%3)) ;
                    P3 = vertices.row(faces(j,(position +2)%3)) ;

                    // compute the angle for the triangle and the area 
                    thisArea = computeArea(P1, P2, P3) ;
                    areaTotal += thisArea ;
                 }
                thisArea = 0.0 ;
             }
             // compute the area for this vertex as 1/3 of the total area of the triangles
             areaTotal = areaTotal/3.0 ;
             // add the diagonal term  
             tripletList.push_back(Eigen::Triplet<double>(i, i,2.0*areaTotal));
         }
         M.setFromTriplets(tripletList.begin(), tripletList.end());
         return M ;
}


Eigen::SparseMatrix<double> weightCotan(DecoratedCloud & cloud) {
        // initialization 
        int numbVertices = cloud.getVertices().rows() ;

        // faces, normals anf vertices 
        Eigen::MatrixXd vertices(numbVertices, 3) ;
        Eigen::MatrixXi faces(numbVertices, 3) ;
        vertices = cloud.getVertices() ;
        faces = cloud.getFaces() ;

        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(10*numbVertices); // more places than needed in case 
        Eigen::SparseMatrix<double> C(numbVertices,numbVertices);

        // compute the adjacency informations for each triangle 
        Eigen::MatrixXd TT(numbVertices,3) ;
        Eigen::MatrixXd TTi(numbVertices,3) ;
        igl::triangle_triangle_adjacency(faces, TT, TTi);

         // contains the information of position and weight for 1 neighbor
         std::vector<double> thisNeighborInfo ;
         thisNeighborInfo.reserve(2) ;
         // contains the information for all the neighbors 
         std::vector<std::vector<double>> neighborInformation ; 
         neighborInformation.reserve(10) ;

         bool findThisVer ;
         double sumWeight, weight, thisWeight;
         int position, thisNeight, triangle2, edge2, valence ;
         Eigen::Vector3d PCurrent, PNeighCurrent, PTriangle1, PTriangle2 ;

         // loop over the vertices 
         for (int i=0; i < vertices.rows(); i++) {
             sumWeight = 0.0 ;
             // loop over the faces 
             for (int j=0 ; j < faces.rows() ; j++ ) {
                 findThisVer = false ;
                 for (int k = 0 ; k < 3 ; k++) {
                    if (faces(j,k) == i) {
                        position = k ;
                        findThisVer = true ;
                        break ;
                    }
                 }

                 if (findThisVer) {
                    // vertices of the first triangle 
                    PCurrent = vertices.row(i) ; 
                    PNeighCurrent = vertices.row(faces(j, (position+2)%3)) ; 
                    PTriangle1 = vertices.row(faces(j, (position+1)%3));

                    // second triangle information 
                    triangle2 = TT(j, (position+2)%3) ;
                    edge2 = TTi(j,(position+2)%3) ;
                    PTriangle2 = vertices.row(faces(triangle2, (edge2+2)%3)) ;

                    // compute the sum of the cotan 
                    weight = computeCotan(PTriangle2, PNeighCurrent, PCurrent) + computeCotan(PTriangle1, PNeighCurrent, PCurrent) ;

                    // create and push the vector with indices and weight for this neighbor 
                    thisNeighborInfo.push_back(double(faces(j,(position +2)%3))) ;
                    thisNeighborInfo.push_back(weight) ;
                    neighborInformation.push_back(thisNeighborInfo) ;

                    // clear all for the next neighbor
                    thisNeighborInfo.clear() ;
                    weight = 0.0 ;
                 }
             }
                
             valence = neighborInformation.size() ;

             for (int index =0; index < valence ; index++) {
                 thisNeight = neighborInformation[index][0] ;
                 thisWeight = neighborInformation[index][1] ;

                 // add the weight to the sum 
                 sumWeight += thisWeight ;

                 tripletList.push_back(Eigen::Triplet<double>(i, thisNeight,thisWeight)) ;
             }

             tripletList.push_back(Eigen::Triplet<double>(i, i,-sumWeight));

             // clear all for the next vertex 
             neighborInformation.clear() ;
             sumWeight = 0.0 ; 
             valence = 0 ;
         }
        C.setFromTriplets(tripletList.begin(), tripletList.end());
        return C ; 
}

// compute the inverse of a sparse Matrix 
Eigen::SparseMatrix<double> inverseSparse(Eigen::SparseMatrix<double> M) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(M);
    Eigen::SparseMatrix<double> I(M.rows(),M.cols());
    I.setIdentity();
    Eigen::SparseMatrix<double>  M_inv = solver.solve(I);
    return M_inv ;
}

// compute the laplace operator from C and M matrices 
Eigen::SparseMatrix<double> computeCotanDiscretization(DecoratedCloud & cloud) {
        int numbVertices = cloud.getVertices().rows() ;
        Eigen::SparseMatrix<double> L(numbVertices,numbVertices);
        Eigen::SparseMatrix<double> C(numbVertices,numbVertices);
        Eigen::SparseMatrix<double> M(numbVertices,numbVertices);
        Eigen::SparseMatrix<double> M_inv(numbVertices,numbVertices);

        C = weightCotan(cloud) ;
        M = diagonalArea(cloud) ;
        M_inv = inverseSparse(M) ;

        L = M_inv*C ;
        return L ; 
}


// ******************* Mean curvature function ***************
Eigen::MatrixXd meanCurvature(DecoratedCloud & cloud, int typeDiscretization) {
        // initialization for vertices and normals 
        int numbVertices = cloud.getVertices().rows() ;
        Eigen::MatrixXd normals(numbVertices, 3) ;
        Eigen::MatrixXd vertices(numbVertices, 3) ;
        normals = cloud.getNormals() ;
        vertices = cloud.getVertices() ;

        // compute the L matrix in function of the discretization asked 
        // can extend this if we compute new types of discretization
        Eigen::SparseMatrix<double> L(numbVertices,numbVertices);
        if (typeDiscretization == 0) L = uniformLaplacian(cloud)  ;
        else if  (typeDiscretization == 1) L = computeCotanDiscretization(cloud) ;

        // compute the laplacian vectors 
        Eigen::MatrixXd laplacian_vectors(numbVertices,3) ;   
        laplacian_vectors = L*vertices ;

        // Compute the magnitude of the mean curvature 
        Eigen::MatrixXd meancurv(numbVertices,1) ;
        meancurv = laplacian_vectors.rowwise().norm()/2.0 ;

        // look for its orientation 
        Eigen::MatrixXd signCurv(numbVertices,1) ; 
        // compute the dot product 
        signCurv = (laplacian_vectors.cwiseProduct(normals)).rowwise().sum();
        // only interested on the sign 
        signCurv = signCurv.array() / signCurv.array().abs().array() ;

        // invert the curvature if needed 
        meancurv = meancurv.cwiseProduct(-1.f*signCurv) ;

        // apply this mean curvature as a color for the resulting mesh
        Eigen::MatrixXd color(numbVertices,3) ;
        igl::jet(meancurv,true,color);

        return color ;
}

// ******************* Gaussian curvature function ***************                
Eigen::MatrixXd gaussianCurvUnifom(DecoratedCloud & cloud) {
        // initialization of matrices and variable 
        int numbVertices =  cloud.getVertices().rows() ;
        Eigen::MatrixXd gaussianCurv(numbVertices,1) ;
        Eigen::MatrixXd vertices(numbVertices, 3) ;
        Eigen::MatrixXi faces(numbVertices, 3) ;
        vertices = cloud.getVertices() ;
        faces = cloud.getFaces() ;
         bool findThisVer ;
         int position ;
         double thisArea, thisAngle, thisGausCurv, sumAngle, areaTotal ;
         Eigen::Vector3d P1, P2, P3 ;

         // loop over the vertices 
         for (int i=0; i < vertices.rows(); i++) {
             sumAngle = 0.0 ;
             areaTotal = 0.0 ;
             thisGausCurv = 0.0 ;
             // loop over the faces  
             for (int j=0 ; j < faces.rows() ; j++ ) {
                 findThisVer = false ;
                 for (int k = 0 ; k < 3 ; k++) {
                    if (faces(j,k) == i) {
                        position = k ;
                        findThisVer = true ;
                        break ;
                    }
                 }

                 // if the vertex is in the faces, compute the area of the triangle 
                 if (findThisVer) {
                     // the vertices of this triangle 
                    P1 = vertices.row(i) ;
                    P2 = vertices.row(faces(j,(position +1)%3)) ;
                    P3 = vertices.row(faces(j,(position +2)%3)) ;

                    // compute the angle for the triangle and the area 
                    thisArea = computeArea(P1, P2, P3) ;
                    thisAngle = computeAngle(P1, P2, P3) ;

                    areaTotal += thisArea ;
                    sumAngle += thisAngle ;
                 }
             }
             // compute the gauss curvature 
            thisGausCurv = (2.0*Pi-sumAngle)/(areaTotal/3.0) ;
            gaussianCurv(i,0) = thisGausCurv ;
         }
        // apply this gaussian curvature as a color for the resulting mesh
         Eigen::MatrixXd color(numbVertices,3) ;
        igl::jet(gaussianCurv,true ,color);

        return color ;
}
} // namespace acq