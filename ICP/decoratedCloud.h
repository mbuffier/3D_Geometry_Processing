//
// Created by bontius on 20/01/17.
//

#ifndef ACQ_DECORATEDCLOUD_H
#define ACQ_DECORATEDCLOUD_H

#include "acq/typedefs.h"
#include <iostream>
#include <ANN/ANN.h>					// ANN declarations
#include <Eigen/SVD>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include "igl/copyleft/cgal/intersect_other.h"
#include "acq/impl/normalEstimation.hpp" // Templated functions
#include <Eigen/Geometry>

namespace acq {

/** \brief Simple class to keep track of points normals and faces for a point cloud or mesh. */
class DecoratedCloud {
public:
    /**Old Constructors */
    explicit DecoratedCloud() {}
    explicit DecoratedCloud(CloudT const& vertices);
    explicit DecoratedCloud(CloudT const& vertices, FacesT const& faces);
    explicit DecoratedCloud(CloudT const& vertices, NormalsT const& normals);
    explicit DecoratedCloud(CloudT const& vertices, FacesT const& faces, NormalsT const& normals);

    /** new Constructors to obtain a cloud from several to visualize (with a boolean to show overlap) */
    explicit DecoratedCloud(DecoratedCloud const& cloud1, DecoratedCloud const& cloud2, bool overlap) ;

    explicit DecoratedCloud(DecoratedCloud const& cloud1, DecoratedCloud const& cloud2, 
                            DecoratedCloud const& cloud3, DecoratedCloud const& cloud4, 
                            DecoratedCloud const& cloud5, DecoratedCloud const& cloud6) ;

    /** Old getters/seters  */
    CloudT const& getVertices() const { return _vertices; }
    void setVertices(CloudT const& vertices) { _vertices = vertices; }
    bool hasVertices() const { return static_cast<bool>(_vertices.size()); }

    FacesT const& getFaces() const { return _faces;}
    void setFaces(FacesT const& faces) { _faces = faces; }
    bool hasFaces() const { return static_cast<bool>(_faces.size()); }

    NormalsT      & getNormals() { return _normals; }
    NormalsT const& getNormals() const { return _normals; }
    void setNormals(NormalsT const& normals) { _normals = normals; }
    bool hasNormals() const { return static_cast<bool>(_normals.size()); }

    /** New getter for the color   */
    ColorsT const& getColors() const { return _color;}

    /** Friend functions  */
    // transform an array to a matrix 
    friend Eigen::MatrixXd arrayToMatrix(ANNpointArray arrayPoint, int size);
    // sample numberPoint integer from a sizeMatrix sized matrix 
    friend Eigen::VectorXi sample(int const numberPoint, int sizeMatrix) ; 
    // give back a rotation matrix from 3 angles on principal axis
    friend Eigen::Matrix3d constructR(float rx, float ry, float rz) ;

    /** Functions to sample the clouds */
    // give back an array of vertex from a vector of indices 
    ANNpointArray sampleVertex(Eigen::VectorXi sampleIndices) const ;
    // same for normals 
    ANNpointArray sampleNormal(Eigen::VectorXi sampleIndices ) const ;

    /** Functions to move meshes */
    // move a mesh from a rotation matrix T and a translation vector 
    void transformation( Eigen::Matrix3d R, Eigen::Matrix<double, 3, 1> T) ;
    // move a mesh with a rotation  theta around y and a translation(x,y,z)
    void install(double theta, float x, float y, float z) ;

    /** Functions to add noise */
    // compute the bounding box 
    void boundingBox(float &Xmax,float & Xmin,float & Ymax,float & Ymin,float &Zmax, float & Zmin);
    // add noise to a mesh according to standart deviations computed from the bounding box 
    void addNoise(float sigmaX, float sigmaY, float sigmaZ) ;

    /** ICP function */
    void icpAlgo(DecoratedCloud const& cloud, int const nPts, ANNkd_tree*	kdTreeConst) ;
    void icpPointToPlane(DecoratedCloud const& cloud, int const nPts, ANNkd_tree*	kdTreeConst) ;
    
protected:
    CloudT   _vertices; //!< Point cloud, N x 3 matrix where N is the number of points.
    FacesT   _faces;    //!< Faces stored as rows of vertex indices (referring to \ref _vertices).
    NormalsT _normals;  //!< Per-vertex normals, associated with \ref _vertices by row ID.
    ColorsT _color ;
public:
    // See https://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; //...class DecoratedCloud()

} //...ns acq

#endif //ACQ_DECORATEDCLOUD_H