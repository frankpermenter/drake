#include "drake/multibody/plant/implicit_stribeck_solver_results.h"

#include "drake/common/default_scalars.h"

// Explicitly instantiates on the most common scalar types.
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    struct ::drake::multibody::internal::ImplicitStribeckSolverResults)
