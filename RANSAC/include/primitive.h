#ifndef ACQ_PRIMITIVE_H
#define ACQ_PRIMITIVE_H

#include "acq/typedefs.h"
#include "acq/decoratedCloud.h"

#include <math.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>


namespace acq {

    /** ---- PRIMITIVE ---- */
    class Primitive {

    public:
        // constructor/destructor
        Primitive() {} ;
        virtual ~Primitive() {} ;

        // getters.setters 
        double getScore()const{return _score;}
        int getType()const{return _type;}
        Eigen::MatrixXi getInliers_idx()const{return _inliers_idx;}

        void setType(int type) {_type = type ; }
        void setScore(double score){_score = score;}
        void setInliers_idx(Eigen::MatrixXi inliers_idx){_inliers_idx = inliers_idx;}

        // Virtual Functions
        virtual double getRadius(){};
        virtual Eigen::Matrix<double, 1,3> getCenter(){};
        virtual Eigen::Matrix<double, 1,3> getNormal(){};
        virtual Eigen::Matrix<double, 1,3> getRefPoint(){};
        virtual Eigen::MatrixXi computeInliers(DecoratedCloud& cloud, double threshold, double alpha){};
        virtual void computeScore(Eigen::Matrix3d variance, DecoratedCloud& pointCloud, double threshold, double alpha){};
        virtual int findBestNumberPoints(Eigen::Matrix3d variance){};        
        virtual Primitive* clone(){};

    protected:
        double _score; 
        int _type; // 1: Sphere, 2: Plane
        Eigen::MatrixXi _inliers_idx;
    };

    /** ---- SPHERE ---- */
    class Sphere : public Primitive {

    public:
        // Constructor and Destructor
        Sphere(double radius, Eigen::Matrix<double, 1,3> center) : _radius(radius), _center(center) {} ;
        ~Sphere(){};

        // Getters/Setters
        double getRadius() const {return _radius;}
        Eigen::Matrix<double, 1,3> getCenter()const{return _center;}
        void setRadius(double radius){_radius = radius;}
        void setCenter(Eigen::Matrix<double, 1,3> center){_center = center;}

        // functions specific to spheres
        void computeScore(Eigen::Matrix3d variance, DecoratedCloud& cloud, double threshold, double alpha);
        Eigen::MatrixXi computeInliers(DecoratedCloud& cloud, double threshold, double alpha);
        int findBestNumberPoints(Eigen::Matrix3d variance) ;
        Primitive* clone() ;

    protected:
        // radius and center of the sphere 
        double _radius;
        Eigen::Matrix<double, 1,3> _center;
    };

    /** ---- PLANE ---- */
    class Plane : public Primitive {

    public:
        // Constructor and Destructor
        Plane(Eigen::Matrix<double, 1,3> refPoint, Eigen::Matrix<double, 1,3> normal) : _refPoint(refPoint), _normal(normal) {} ;
        ~Plane(){};

        // getters/setters
        Eigen::Matrix<double, 1,3> getNormal() const {return _normal;}
        Eigen::Matrix<double, 1,3> getRefPoint()const {return _refPoint;}
        void setNormal(Eigen::Matrix<double, 1,3> normal) {_normal = normal;}
        void setRefPoint(Eigen::Matrix<double, 1,3> refPoint) {_refPoint = refPoint;}

        // this one is override because we use it one a primitive object
        Eigen::MatrixXi computeInliersPlane(DecoratedCloud& cloud, double threshold, double alpha) ;

        void computeScore(Eigen::Matrix3d variance, DecoratedCloud& cloud, double T, double alpha);
        Eigen::MatrixXi computeInliers(DecoratedCloud& cloud, double T, double alpha);
        int findBestNumberPoints(Eigen::Matrix3d var, DecoratedCloud& cloud,Eigen::MatrixXi inliers_idx);
        Primitive* clone() ;

    protected:
        Eigen::Matrix<double, 1,3> _refPoint;
        Eigen::Matrix<double, 1,3> _normal;
    };
}

#endif
