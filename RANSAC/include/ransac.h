#ifndef IGLFRAMEWORK_RANSAC_H
#define IGLFRAMEWORK_RANSAC_H

#include "acq/primitiveManager.h"
#include "acq/decoratedCloud.h"
#include "acq/gestion.h"
#include "acq/cloudManager.h"

namespace acq {
    // main function apply to a cloud which modifies best_primitives and cloudManager with the best primitives found
    bool ransac(DecoratedCloud& cloud, PrimitiveManager& best_primitives, CloudManager& cloudManager, 
                double thresh, double alpha, double thresh_best, int iterationsTotal, int numberSample);
}

#endif

