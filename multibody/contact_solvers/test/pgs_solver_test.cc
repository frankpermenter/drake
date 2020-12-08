#include "drake/multibody/contact_solvers/test/pgs_solver.h"

#include <gtest/gtest.h>

#include "drake/multibody/contact_solvers/test/stack_of_objects_test.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename ContactSolverType>
class StackOfBoxesWithPgsTest : public StackOfBoxesTest<ContactSolverType> {
  void SetParams(NormalConstraintType,
                 ContactSolver<double>* solver) const override {
    PgsSolverParameters params;
    params.relaxation = 1;
    params.max_iterations = 25000;
    params.abs_tolerance = 1.0e-4;
    params.rel_tolerance = 1.0e-5;
    auto pgs_solver = dynamic_cast<PgsSolver<double>*>(solver);
    DRAKE_DEMAND(pgs_solver);
    pgs_solver->set_parameters(params);
  }
};

typedef ::testing::Types<PgsSolver<double>> ContactSolverTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(ContactSolvers, StackOfBoxesWithPgsTest,
                               ContactSolverTypes);

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
