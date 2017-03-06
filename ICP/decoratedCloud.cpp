//
// Created by bontius on 20/01/17.
//

#include "acq/impl/decoratedCloud.hpp"
//#include namespace std;
namespace acq {

/*** Old constructors ***/
DecoratedCloud::DecoratedCloud(CloudT const& vertices)
    : _vertices(vertices) {}

DecoratedCloud::DecoratedCloud(CloudT const& vertices, FacesT const& faces)
    : _vertices(vertices), _faces(faces)
{}

DecoratedCloud::DecoratedCloud(CloudT const& vertices, FacesT const& faces, NormalsT const& normals)
    : _vertices(vertices), _faces(faces), _normals(normals)
{}

DecoratedCloud::DecoratedCloud(CloudT const& vertices, NormalsT const& normals)
    : _vertices(vertices), _normals(normals)
{}


/* ********* New constructors : from 2 and 6 meshes ************ */
DecoratedCloud::DecoratedCloud(DecoratedCloud const& cloud1, DecoratedCloud const& cloud2, bool overlap) 
 {
    // Concatenate (V1,F1, N1) and (V2,F2, N2) into (V,F, N)
    Eigen::MatrixXd const &V1 = cloud1.getVertices() ;
    Eigen::MatrixXd const &V2 = cloud2.getVertices();

    Eigen::MatrixXi const &F1 = cloud1.getFaces() ;
    Eigen::MatrixXi const &F2 = cloud2.getFaces() ;

    Eigen::MatrixXd const &N1 = cloud1.getNormals() ;
    Eigen::MatrixXd const &N2 = cloud2.getNormals() ;

    // compute new vertices matrix 
    Eigen::MatrixXd V(V1.rows()+V2.rows(),V1.cols());
    V<<V1,V2;
    // compute new faces matrix 
    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F<<F1,(F2.array()+V1.rows());
    // compute new normals matrix 
    Eigen::MatrixXd N(N1.rows()+N2.rows(),N1.cols());
    N<<N1,N2;

    // create the color for the first mesh  
    Eigen::MatrixXd C1(F1.rows(),3);
    C1<< Eigen::RowVector3d(0.2,0.3,0.8).replicate(F1.rows(),1);

    // create the color for the second mesh  
    Eigen::MatrixXd C2(F2.rows(),3);
    C2<< Eigen::RowVector3d(1.0,0.7,0.2).replicate(F2.rows(),1);

    // compute the color of the overlap if overlap true  
    if (overlap) {
        // initiazation  
        Eigen::MatrixXi IF;
        const bool first_only = false;

        // find the faces in common
       igl::copyleft::cgal::intersect_other(V1,F1,V2,F2,first_only,IF);

       // color for the intersection 
       Eigen::RowVector3d resultColor ;
       resultColor << 0.1,0.9,0.2 ;

       // go through the intersection and change the color for those faces in both meshes 
      for(int f = 0;f<IF.rows();f++) {
            C1.row(IF(f,0)) = resultColor;
            C2.row(IF(f,1)) = resultColor;
       }

       // compute the pourcentage of intersection 
       float success = (double(IF.rows())/double(F.rows()))*100.0 ;
       std::cout << "Pourcentage of intersection : " << success << " % "<< std::endl ;
    }

    // compute the nex colors matrix
    Eigen::MatrixXd C(F.rows(),3);
    C << C1, C2 ;

    // create the resulting mesh 
    _vertices = V ;
    _faces = F ;
    _color = C ;
    _normals = N ;
}

// construct 1 mesh with all the 6 meshes 
DecoratedCloud::DecoratedCloud(DecoratedCloud const& cloud1, DecoratedCloud const& cloud2, 
                                DecoratedCloud const& cloud3, DecoratedCloud const& cloud4, 
                                DecoratedCloud const& cloud5, DecoratedCloud const& cloud6) {
    // construct clouds 2 by 2 
    DecoratedCloud const& cloudInter1 = DecoratedCloud(cloud1, cloud2, false) ;
    DecoratedCloud const& cloudInter2 = DecoratedCloud(cloud3, cloud4, false) ;
    DecoratedCloud const& cloudInter3 = DecoratedCloud(cloud5, cloud6, false) ;

    DecoratedCloud const& cloudInter4 = DecoratedCloud(cloudInter1, cloudInter2, false) ;

    // compute the final cloud 
    DecoratedCloud const& finalCloud = DecoratedCloud(cloudInter4, cloudInter3, false) ;

    _vertices = finalCloud.getVertices() ;
    _faces = finalCloud.getFaces() ; ;
    _normals = finalCloud.getNormals() ;

    // create the color for the meshes  
    Eigen::MatrixXd C1(cloud1.getFaces().rows(),3);
    C1<< Eigen::RowVector3d(0.2,0.3,0.8).replicate(cloud1.getFaces().rows(),1);

    Eigen::MatrixXd C2(cloud2.getFaces().rows(),3);
    C2<< Eigen::RowVector3d(1.0,0.7,0.2).replicate(cloud2.getFaces().rows(),1);

    Eigen::MatrixXd C3(cloud3.getFaces().rows(),3);
    C3<< Eigen::RowVector3d(0.1,0.4,0.5).replicate(cloud3.getFaces().rows(),1);

    Eigen::MatrixXd C4(cloud4.getFaces().rows(),3);
    C4<< Eigen::RowVector3d(0.2,0.4,0.2).replicate(cloud4.getFaces().rows(),1);
   
    Eigen::MatrixXd C5(cloud5.getFaces().rows(),3);
    C5<< Eigen::RowVector3d(0.5,0.4,0.8).replicate(cloud5.getFaces().rows(),1);

    Eigen::MatrixXd C6(cloud6.getFaces().rows(),3);
    C6 << Eigen::RowVector3d(0.8,0.7,0.2).replicate(cloud6.getFaces().rows(),1);

   // compute the new colors matrix
    Eigen::MatrixXd C(_faces.rows(),3);
    C << C1, C2, C3, C4, C5, C6 ;

    // create the resulting mesh 
    _color = C ;
}


/* ********* Friend functions ************ */
Eigen::VectorXi sample(int const numberPoint, int sizeMatrix) {
    Eigen::VectorXi sampleInd(numberPoint,1) ;
    // add a random indices between 0 and sizeMatrix in a numberPoint sized vector 
    for (int i=0; i<numberPoint; i++) {
        int newIndex = rand() % (sizeMatrix + 1) ;
        sampleInd(i) = newIndex ;
    }
    return sampleInd ;
}

// take an array and gives back a matrix 
Eigen::MatrixXd arrayToMatrix(ANNpointArray arrayPoint, int size) {
    Eigen::MatrixXd pointsMatrix(size, 3) ;
    // construct the matrix row by row 
    for (int i=0; i<size ; i++) {
        for (int j =0 ; j<3; j++) {
            pointsMatrix(i,j) = arrayPoint[i][j] ;            
        }
    }
    return pointsMatrix ;
}

 // give back a rotation matrix from 3 angles on principal axis
Eigen::Matrix3d constructR(float rx, float ry, float rz) {
    // rotation on the x axis
    Eigen::Matrix3d Rx ;
    Rx << 1, 0, 0, 
         0, cos(rx), -sin(rx), 
         0,  sin(rx), cos(rx) ;
    // rotation on the y axis
    Eigen::Matrix3d Ry ;
    Ry << cos(ry), 0, sin(ry), 
         0, 1, 0, 
         -sin(ry),  0, cos(ry) ;
    // rotation on the z axis
    Eigen::Matrix3d Rz ;
    Rz << cos(rz), -sin(rz), 0,  
         sin(rz), cos(rz), 0,
         0, 0, 1 ;

    // final rotation 
    Eigen::Matrix3d R ;
    R = Rz*Ry*Rx ;
    return R ;
}

/* *********  Functions to sample the clouds ************ */
// from an vector of sampled indices, extract the array of vertices 
ANNpointArray DecoratedCloud::sampleVertex(Eigen::VectorXi sampleIndices) const {
    int nbPoint = sampleIndices.size() ;

    // allocation of memory for the array 
    ANNpointArray dataPts ;
    int dim = 3 ;			
	dataPts = annAllocPts(nbPoint, dim);			
    // create the array 
    int index ;
    for (int i=0; i<nbPoint ; i++) {
        index = sampleIndices(i);
        ANNpoint point ;
        point = annAllocPt(dim) ;

        for (int j=0; j<dim; j++) {
            point[j] = _vertices(index,j) ;
        }
        dataPts[i] = point ;
    }
    // return the array 
    return dataPts ;
}

// from an vector of sampled indices, extract the array of normal : same as above  
ANNpointArray DecoratedCloud::sampleNormal(Eigen::VectorXi sampleIndices) const {
    int nbPoint = sampleIndices.size() ;

    ANNpointArray normalPts ;
    int dim = 3 ;		
	normalPts = annAllocPts(nbPoint, dim);		
    int index ;
    for (int i=0; i<nbPoint ; i++) {
        index = sampleIndices(i);
        ANNpoint point ;
        point = annAllocPt(dim) ;

        for (int j=0; j<dim; j++) {
            point[j] = _normals(index,j) ;
        }
        normalPts[i] = point ;
    }
    return normalPts ;
}

/* *********  Functions to move meshes ************ */

// move a mesh from a rotation matrix T and a translation vector 
void DecoratedCloud::transformation(Eigen::Matrix3d R, Eigen::Matrix<double, 3, 1> T) {  
     // construct the transformation matrix   
     Eigen::Matrix4d transform = Eigen::MatrixXd::Zero(4,4) ; 
     transform.topLeftCorner(3, 3) = R ;
     transform.topRightCorner(3, 1) = T ;
     transform.bottomLeftCorner(1, 3) = Eigen::MatrixXd::Zero(1,3) ;
     transform(3,3) =1 ;

     // transform coordinates in homogenous coordinates
     Eigen::MatrixXd homogenousVert(_vertices.rows(),_vertices.cols()+1) ;
     homogenousVert.leftCols(3) = _vertices ;
     homogenousVert.col(3) = Eigen::MatrixXd::Ones(_vertices.rows(),1) ;                            

     // apply the transformation and gives back the transformated points 
    Eigen::MatrixXd newVertices(_vertices.cols()+1,_vertices.rows()) ;
    newVertices =  transform*homogenousVert.transpose() ;

    // move the mesh 
     _vertices = newVertices.transpose().topLeftCorner(_vertices.rows(), _vertices.cols()) ;
}

// move a mesh with an angle theta around y and a translation (x,y,z)
void DecoratedCloud::install(double theta, float x, float y, float z) {
    // rotation 
     Eigen::Matrix3d R ;
        R << cos(theta), 0, sin(theta), 
              0, 1, 0, 
            -sin(theta),  0, cos(theta) ;

    // translation matrix
    Eigen::Matrix<double, 3,1> T ;
     T << x,y,z ;
     
     // transform the mesh 
     this->transformation(R,T) ; 
}

/* *********  Functions to add noise ************ */

// compute the bounding box of a mesh 
void DecoratedCloud::boundingBox(float &Xmax,float & Xmin,float & Ymax,float & Ymin,float &Zmax, float & Zmin) {
    Eigen::MatrixXd maximum(1, 3)  ;
    maximum = _vertices.colwise().maxCoeff() ;
    Xmax = maximum(0) ;
    Ymax= maximum(1) ;
    Zmax  = maximum(2) ;

    Eigen::MatrixXd minimum(1, 3)  ;
    minimum = _vertices.colwise().minCoeff() ;
    Xmin = minimum(0) ;
    Ymin = minimum(1) ;
    Zmin = minimum(2) ;

}

// add noise 
void DecoratedCloud::addNoise(float sigmaX, float sigmaY, float sigmaZ) {
    int M = _vertices.size() ;

    // initialization 
    Eigen::MatrixXd random(M, 3)  ;

    // construct noise 
    for(int i = 0; i< M; i++) {
        random(i, 0) = std::rand()*sigmaX /RAND_MAX ; 
        random(i, 1) = std::rand()*sigmaY /RAND_MAX ;
        random(i, 2) = std::rand()*sigmaZ /RAND_MAX ;
    }
    // add noise to the vertices 
    _vertices += random ;
}

/* *********  ICP function ************ */

void DecoratedCloud::icpAlgo(DecoratedCloud const& cloud,int const nbP, ANNkd_tree*	kdTreeConst) {

    // get back the dimensions of the matrix
    int nPts = cloud.getVertices().rows() ;
    int dim = cloud.getVertices().cols() ; 
    int M = _vertices.rows() ;

    // sample the indice for the given number of points
    Eigen::VectorXi sampleIndices(nbP, 1);// indices sample
    sampleIndices = sample(nbP, M) ;

    // create the array for the kd-tree points to sample 
    ANNpointArray sampleVertices ; // array of points 
    sampleVertices = annAllocPts(nbP, dim);
    ANNpointArray sampleNormals ; // array of normals
    sampleNormals = annAllocPts(nbP, dim);

    // sample the cloud which move 
    sampleVertices = this->sampleVertex(sampleIndices) ;
    sampleNormals = this->sampleNormal(sampleIndices) ;

    // initialization for the search into the kd-tree
    ANNpoint queryPt;
    queryPt = annAllocPt(dim);
        
    // to check the normals : point from the fixed cloud 
    Eigen::Matrix<double, 1,3> normalPt ;
    Eigen::Matrix<double, 1,3> verticeNormal ;

    // to store the result 
    Eigen::MatrixXd resultpoint(nbP, 3) ;
    Eigen::MatrixXd resultmatching(nbP, 3) ;

    int index = 0 ;
    // fill the neighbor distance and indices 
    for (int i=0; i<nbP ; i++) {
        // allocate variable for the kdTree
        ANNidxArray	nnIdx;	
        nnIdx = new ANNidx[1]; 
	    ANNdistArray dists;					// near neighbor distances
	    dists = new ANNdist[1];           // allocate near neighbor dists
        queryPt = sampleVertices[i] ;

        kdTreeConst->annkSearch(				// search
				queryPt,						// query point
				1,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				0.0);							// error bound

        // take the result 
        int indexFind =  nnIdx[0] ;
        double distance =  dists[0] ;

        // to test the dot product, transformation into a vector
        for(int j=0 ; j<3 ; j++) {  
            normalPt(0,j) = sampleNormals[i][j] ;
        }

        verticeNormal = cloud.getNormals().row(indexFind) ;

        // compute the dot product 
        float test = std::abs(normalPt.dot(verticeNormal)) ;

        // test to elimanate bad matches 
        if (distance < 0.001 && test > 0.98 ) {
            for(int j=0 ; j<3 ; j++) {
                resultpoint(index,j) = sampleVertices[i][j] ;
            }          
  
            resultmatching.row(index) = cloud.getVertices().row(indexFind) ;
            index +=1 ;
        }
         // free memory 
        delete [] nnIdx ;
        delete [] dists;
    }
    // free memory 
    annDeallocPts(sampleVertices) ;
    annDeallocPts(sampleNormals) ;
    annDeallocPt(queryPt) ;

    // print the result for this search 
    float pour = (float(index)/nbP)*100.0 ;
    std::cout<< "Number of points used :  " << index << " over " <<  nbP << " points sampled (" << pour <<"%)"<< std::endl ;

    index -= 1 ;
    Eigen::MatrixXd resultPoint(index, 3) ;
    Eigen::MatrixXd resultMatching(index, 3) ;

    // only take the points which matched 
    resultPoint = resultpoint.topRows(index) ;
    resultMatching = resultmatching.topRows(index) ;

    // barycenter for the moving cloud 
    Eigen::Matrix<double, 1, 3> barySampleP  ; 
    barySampleP = resultPoint.colwise().sum()/index ;
    Eigen::MatrixXd barySamplePArray(index, 3) ;
    barySamplePArray = barySampleP.replicate(index,1) ;

    // barycenter for the fixed cloud 
    Eigen::Matrix<double, 1, 3> baryCouldP  ; 
    baryCouldP = resultMatching.colwise().sum()/index ;
    Eigen::MatrixXd baryCouldPArray(index, 3) ;
    baryCouldPArray = baryCouldP.replicate(index,1) ;

    // centered points for moving cloud 
    resultPoint = resultPoint-barySamplePArray ;

    // centered points for fixed cloud 
    resultMatching = resultMatching - baryCouldPArray ;

    // creation of R
    Eigen::Matrix3d A = (resultMatching.transpose())*resultPoint ; 
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);    
    Eigen::Matrix3d U =  svd.matrixU() ; 
    Eigen::Matrix3d V =  svd.matrixV() ; 
    Eigen::Matrix3d R ; 
    R = U*(V.transpose()) ; 

    // creation of T
    Eigen::Matrix<double, 3, 1> T ; 
    T = baryCouldP.transpose() - R*barySampleP.transpose() ;

    // transformation of the cloud 
    this->transformation(R,T) ;
}


// apply ICP point to plane 
void DecoratedCloud::icpPointToPlane(DecoratedCloud const& cloud,int const nbP, ANNkd_tree*	kdTreeConst) {

    // dimensions of the matrix
    int nPts = cloud.getVertices().rows() ;
    int dim = cloud.getVertices().cols() ; 
    int M = _vertices.rows() ;

    // sample nbp points from the matrix 
    Eigen::VectorXi sampleIndices(nbP, 1);// indices sample
    sampleIndices = sample(nbP, M) ;

    // create the array for the kd-tree points to sample 
    ANNpointArray sampleVertices ; // array of points 
    sampleVertices = annAllocPts(nbP, dim);
    ANNpointArray sampleNormals ; // array of normals
    sampleNormals = annAllocPts(nbP, dim);

    sampleVertices = this->sampleVertex(sampleIndices) ;
    sampleNormals = this->sampleNormal(sampleIndices) ;

    // initialization
    ANNpoint queryPt;
    queryPt = annAllocPt(dim);
        
    // initialization to fill A  
    Eigen::Matrix<double, 1,3> normalPt ;
    Eigen::Matrix<double, 1,3> coordPt ;
    Eigen::Matrix<double, 1,3> coordPtFixed ;
    Eigen::Matrix<double, 1,3> normalPtFixed ;

    // to solve the system 
    Eigen::MatrixXd A(nbP, 6) ;
    Eigen::MatrixXd b(nbP, 1) ;

    int index = 0 ;
    // fill the neighbor distance and indices 
    for (int i=0; i<nbP ; i++) {

        // allocate variable for the kdTree
        ANNidxArray	nnIdx;	
        nnIdx = new ANNidx[1]; 
	    ANNdistArray dists;					// near neighbor distances
	    dists = new ANNdist[1];           // allocate near neighbor dists
        queryPt = sampleVertices[i] ;

        kdTreeConst->annkSearch(				// search
				queryPt,						// query point
				1,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				0.0);							// error bound

        // take the result 
        int indexFind =  nnIdx[0] ;
        double distance =  dists[0] ;

        // transformation into a matrix for algebric computation 
        for(int j=0 ; j<3 ; j++) {  
            normalPt(0,j) = sampleNormals[i][j] ;
            coordPt(0,j) = sampleVertices[i][j] ;
        }
        coordPtFixed = cloud.getVertices().row(indexFind) ;
        normalPtFixed = cloud.getNormals().row(indexFind) ;

        // compute the dot product 
        float test = std::abs(normalPt.dot(normalPtFixed)) ;

        // test to elimanate bad matches 
        if (distance < 0.001 && test > 0.5 ) {
            // fill A matrix 
            A.row(index) << coordPt.cross(normalPt), normalPt ;

            // fill b vector 
            b(index) = -normalPt.dot(coordPt-coordPtFixed) ;
            index +=1 ;
        }

         // free memory 
        delete [] nnIdx ;
        delete [] dists;
    }
    // free memory 
    annDeallocPts(sampleVertices) ;
    annDeallocPts(sampleNormals) ;
    annDeallocPt(queryPt) ;

    // print the result for this search 
    float pour = (float(index)/nbP)*100.0 ;
    std::cout<< "Number of points used :  " << index << " over " <<  nbP << " points sampled (" << pour <<"%)"<< std::endl ;

    // only take the points which matched 
    index -= 1 ;
    A = A.topRows(index) ;
    b = b.topRows(index) ;

    // save Ax = b to find R and T 
    Eigen::VectorXd result = A.colPivHouseholderQr().solve(b);

    // construct the rotation from the resulting vector 
    Eigen::Matrix3d R ;
    R = constructR(result(0), result(1), result(2)) ;

    Eigen::Matrix<double, 3,1> T ;
    T << result(3), result(4), result(5) ;
    
    // apply the transformation 
    this->transformation(R,T) ;
}

} //...ns acq

