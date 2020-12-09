#pragma once

namespace drake {
namespace solvers {

enum class SolverType {
  kConex,
  kCsdp,
  kDReal,
  kEqualityConstrainedQP,
  kGurobi,
  kIpopt,
  kLinearSystem,
  kMobyLCP,
  kMosek,
  kNlopt,
  kOsqp,
  kSnopt,
  kScs,
  kUnrevisedLemke
};

}  // namespace solvers
}  // namespace drake
