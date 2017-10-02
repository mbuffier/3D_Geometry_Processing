#ifndef ACQ_RECONSTRUCTION_H
#define ACQ_RECONSTRUCTION_H

#include "acq/cloudManager.h"
#include "acq/primitiveManager.h"

namespace acq {
    void reconstruct(PrimitiveManager &best_primitives, DecoratedCloud &cloud, int nbSamples, double T, double alpha, double T2);
    void sampleFromPrimitive(DecoratedCloud &cloud, Eigen::MatrixXi inliers_idx, Primitive *plane, int nbSample, double T);
}
#endif

