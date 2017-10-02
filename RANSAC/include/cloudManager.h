//
// Created by bontius on 20/01/17.
//

#ifndef ACQ_CLOUDMANAGER_H
#define ACQ_CLOUDMANAGER_H

#include "acq/decoratedCloud.h"
#include <vector>

namespace acq {

/** \brief Small class to keep track of multiple point clouds. */
class CloudManager {
public:
    /** \brief Append a cloud to the list of clouds. */
    void addCloud(DecoratedCloud const& cloud);

    /** \brief Overwrite a cloud at a specific index. */
    void setCloud(DecoratedCloud const& cloud, int index);

    /** \brief Get cloud with specific index. */
    DecoratedCloud& getCloud(int index);

    /** \brief Get cloud with specific index (const version). */
    DecoratedCloud const& getCloud(int index) const;

    /** \brief Delete cloud with specific index (const version). */
    void deleteCloud(int index);

    // clear all the cloud in the cloudManager
    void clearCloud() ;

    // get cloud size 
    int getCloudSize(){return _clouds.size();};

    // delete the primitives from the given index 
    void deleteCloudFromIndex(int indexStart);

protected:
    std::vector<DecoratedCloud> _clouds; //!< List of clouds possibly with normals and faces.

public:
    // See https://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; //...class CloudManager

} //...ns acq

#endif // ACQ_CLOUDMANAGER_H
