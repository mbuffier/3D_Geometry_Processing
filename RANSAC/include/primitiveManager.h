#ifndef ACQ_PRIMITIVEMANAGER_H
#define ACQ_PRIMITIVEMANAGER_H

#include "acq/primitive.h"
#include <vector>

namespace acq {

/** Small class to keep track of multiple primitives */
    class PrimitiveManager {
    public:

        // test constructor 
        PrimitiveManager() {} ;
        ~PrimitiveManager() {} ;

        /** add a primitive to the vector */
        void addPrimitive(Primitive* primitive);

        /** set the primitive to a fixed place */
        void setPrimitive(Primitive*, int index);

        /** get back the primitive from a vector  */
        Primitive* getPrimitive(int index);
        int getCloudSize() {return _primitives.size();};

        // find the primitive with the best score : return the index
        int findBestScore() ;

        // delete primitive
        void deletePrimitive(int index) ;

        // clean all the primitive in the cloud 
        void clearAllPrimitives() ;

        // delete from a given index to the end 
        void deleteCloudFromIndex(int indexStart) ;
    protected:
        std::vector<Primitive*> _primitives ;

    public:
        // See https://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    }; 

} 

#endif 
