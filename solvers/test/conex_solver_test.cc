#include "drake/solvers/conex_solver.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/test/mathematical_program_test_util.h"


#include "drake/solvers/test/linear_program_examples.h"
#include "drake/solvers/test/mathematical_program_test_util.h"
#include "drake/solvers/test/quadratic_program_examples.h"
#include "drake/solvers/test/second_order_cone_program_examples.h"
#include "drake/solvers/test/semidefinite_program_examples.h"
#include "drake/solvers/test/sos_examples.h"

namespace drake {
namespace solvers {
namespace test {

namespace {

// Conex uses `eps = 1e-5` by default.  For testing, we'll allow for some
// small cumulative error beyond that.
constexpr double kTol = 1e-4;

}  // namespace

#if 1
GTEST_TEST(TestConex, QuadraticMinimization0) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();

  Eigen::MatrixXd Q(2, 2);
  Eigen::MatrixXd c(2, 1);
  // clang-format off
  Q << 1, .1,
      .1, 4;
  c << 1,
       2;
  // clang-format on

  prog.AddQuadraticCost(Q, c, x);

  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    EXPECT_NEAR((Q * result.GetSolution(x) + c).norm(), 0, kTol);
  }
}

GTEST_TEST(TestConex, QuadraticMinimization1) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddQuadraticCost(x(0) * x(0) + x(1) * x(1) + x(0) + x(1));

  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    const Eigen::Vector2d x_expected(-.5, -.5);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(x), x_expected, kTol,
                                  MatrixCompareType::absolute));
  }
}

GTEST_TEST(TestConex, QuadraticMinimization2) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddQuadraticCost(x(0) * x(0) + x(1) * x(1) + x(0) + x(1));
  prog.AddConstraint(x(0)  <= 10);
  prog.AddConstraint(x(1)  >= -10);

  prog.AddLorentzConeConstraint(2, x(0) * x(0) + x(1) * x(1));

  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    const Eigen::Vector2d x_expected(-.5, -.5);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(x), x_expected, kTol,
                                  MatrixCompareType::absolute));
  }
}


GTEST_TEST(LinearProgramTest, Test0) {
  // Test a linear program with only equality constraint.
  // min x(0) + 2 * x(1)
  // s.t x(0) + x(1) = 2
  // The problem is unbounded.
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>("x");
  prog.AddLinearCost(x(0) + 2 * x(1));
  prog.AddLinearConstraint(x(0) + x(1) == 2);
  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    EXPECT_EQ(result.get_solution_result(), SolutionResult::kUnbounded);
  }

  // Now add the constraint x(1) <= 1. The problem is
  // min x(0) + 2 * x(1)
  // s.t x(0) + x(1) = 2
  //     x(1) <= 1
  // the problem should still be unbounded.
  prog.AddBoundingBoxConstraint(-std::numeric_limits<double>::infinity(), 1,
                                x(1));
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    EXPECT_EQ(result.get_solution_result(), SolutionResult::kUnbounded);
  }

  // Now add the constraint x(0) <= 5. The problem is
  // min x(0) + 2x(1)
  // s.t x(0) + x(1) = 2
  //     x(1) <= 1
  //     x(0) <= 5
  // the problem should be feasible. The optimal cost is -1, with x = (5, -3)
  prog.AddBoundingBoxConstraint(-std::numeric_limits<double>::infinity(), 5,
                                x(0));
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    EXPECT_TRUE(result.is_success());
    EXPECT_NEAR(result.get_optimal_cost(), -1, kTol);
    const Eigen::Vector2d x_expected(5, -3);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(x), x_expected, kTol,
                                MatrixCompareType::absolute));
  }

  // Now change the cost to 3x(0) - x(1) + 5, and add the constraint 2 <= x(0)
  // The problem is
  // min 3x(0) - x(1) + 5
  // s.t x(0) + x(1) = 2
  //     2 <= x(0) <= 5
  //          x(1) <= 1
  // The optimal cost is 11, the optimal solution is x = (2, 0)
  prog.AddLinearCost(2 * x(0) - 3 * x(1) + 5);
  prog.AddBoundingBoxConstraint(2, 6, x(0));
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    EXPECT_TRUE(result.is_success());
    EXPECT_NEAR(result.get_optimal_cost(), 11, kTol);
    const Eigen::Vector2d x_expected(2, 0);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(x), x_expected, kTol,
                                MatrixCompareType::absolute));
  }
}

GTEST_TEST(LinearProgramTest, Test1) {
  // Test a linear program with only equality constraints
  // min x(0) + 2 * x(1)
  // s.t x(0) + x(1) = 1
  //     2x(0) + x(1) = 2
  //     x(0) - 2x(1) = 3
  // This problem is infeasible.
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>("x");
  prog.AddLinearCost(x(0) + 2 * x(1));
  prog.AddLinearEqualityConstraint(x(0) + x(1) == 1 && 2 * x(0) + x(1) == 2);
  prog.AddLinearEqualityConstraint(x(0) - 2 * x(1) == 3);
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    auto result = scs_solver.Solve(prog, {}, {});
    EXPECT_EQ(result.get_solution_result(),
              SolutionResult::kInfeasibleConstraints);
  }
}

GTEST_TEST(LinearProgramTest, Test2) {
  // Test a linear program with bounding box, linear equality and inequality
  // constraints
  // min x(0) + 2 * x(1) + 3 * x(2) + 2
  // s.t  x(0) + x(1) = 2
  //      x(0) + 2x(2) = 3
  //      -2 <= x(0) + 4x(1) <= 10
  //      -5 <= x(1) + 2x(2) <= 9
  //      -x(0) + 2x(2) <= 7
  //      -x(1) + 3x(2) >= -10
  //       x(0) <= 10
  //      1 <= x(2) <= 9
  // The optimal cost is 8, with x = (1, 1, 1)
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<3>();
  prog.AddLinearCost(x(0) + 2 * x(1) + 3 * x(2) + 2);
  prog.AddLinearEqualityConstraint(x(0) + x(1) == 2 && x(0) + 2 * x(2) == 3);
  Eigen::Matrix<double, 3, 3> A;
  // clang-format off
  A << 1, 4, 0,
       0, 1, 2,
       -1, 0, 2;
  // clang-format on
  prog.AddLinearConstraint(
      A, Eigen::Vector3d(-2, -5, -std::numeric_limits<double>::infinity()),
      Eigen::Vector3d(10, 9, 7), x);
  prog.AddLinearConstraint(-x(1) + 3 * x(2) >= -10);
  prog.AddBoundingBoxConstraint(-std::numeric_limits<double>::infinity(), 10,
                                x(0));
  prog.AddBoundingBoxConstraint(1, 9, x(2));

  ConexSolver scs_solver;
  if (scs_solver.available()) {
    auto result = scs_solver.Solve(prog, {}, {});
    EXPECT_TRUE(result.is_success());
    EXPECT_NEAR(result.get_optimal_cost(), 8, kTol);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(x), Eigen::Vector3d(1, 1, 1),
                                kTol, MatrixCompareType::absolute));
  }
}

TEST_P(LinearProgramTest, TestLP) {
  ConexSolver solver;
  prob()->RunProblem(&solver);
}

INSTANTIATE_TEST_SUITE_P(
    ConexTest, LinearProgramTest,
    ::testing::Combine(::testing::ValuesIn(linear_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(linear_problems())));

TEST_F(InfeasibleLinearProgramTest0, TestInfeasible) {
  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(*prog_, {}, {});
    EXPECT_EQ(result.get_solution_result(),
              SolutionResult::kInfeasibleConstraints);
    EXPECT_EQ(result.get_optimal_cost(),
              MathematicalProgram::kGlobalInfeasibleCost);
  }
}

TEST_F(UnboundedLinearProgramTest0, TestUnbounded) {
  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(*prog_, {}, {});
    EXPECT_EQ(result.get_solution_result(), SolutionResult::kUnbounded);
    EXPECT_EQ(result.get_optimal_cost(), MathematicalProgram::kUnboundedCost);
  }
}

TEST_P(TestEllipsoidsSeparation, TestSOCP) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    SolveAndCheckSolution(scs_solver, kTol);
  }
}

INSTANTIATE_TEST_SUITE_P(ConexTest, TestEllipsoidsSeparation,
                        ::testing::ValuesIn(GetEllipsoidsSeparationProblems()));

TEST_P(TestQPasSOCP, TestSOCP) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    SolveAndCheckSolution(scs_solver, kTol);
  }
}

INSTANTIATE_TEST_SUITE_P(ConexTest, TestQPasSOCP,
                        ::testing::ValuesIn(GetQPasSOCPProblems()));


GTEST_TEST(TestSOCP, MaximizeGeometricMeanTrivialProblem1) {
  MaximizeGeometricMeanTrivialProblem1 prob;
  ConexSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(prob.prog(), {}, {});
    prob.CheckSolution(result, 4E-6);
  }
}

GTEST_TEST(TestSOCP, MaximizeGeometricMeanTrivialProblem2) {
  MaximizeGeometricMeanTrivialProblem2 prob;
  ConexSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(prob.prog(), {}, {});
    // On Mac the accuracy is about 2.1E-6. On Linux it is about 2E-6.
    prob.CheckSolution(result, 2.1E-6);
  }
}

GTEST_TEST(TestSOCP, SmallestEllipsoidCoveringProblem) {
  ConexSolver solver;
  SolveAndCheckSmallestEllipsoidCoveringProblems(solver, 4E-6);
}

TEST_P(QuadraticProgramTest, TestQP) {
  ConexSolver solver;
  if (solver.available()) {
    prob()->prog()->SetSolverOption(ConexSolver::id(), "verbose", 0);
    prob()->RunProblem(&solver);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConexTest, QuadraticProgramTest,
    ::testing::Combine(::testing::ValuesIn(quadratic_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(quadratic_problems())));

GTEST_TEST(QPtest, TestUnitBallExample) {
  ConexSolver solver;
  if (solver.available()) {
    TestQPonUnitBallExample(solver);
  }
}


GTEST_TEST(TestSemidefiniteProgram, EigenvalueProblem) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    SolveEigenvalueProblem(scs_solver, kTol);
  }
}

GTEST_TEST(TestSemidefiniteProgram, TrivialSDP) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    TestTrivialSDP(scs_solver, kTol);
  }
}

GTEST_TEST(TestSemidefiniteProgram, OuterEllipsoid) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    FindOuterEllipsoid(scs_solver, kTol);
  }
}

GTEST_TEST(TestSemidefiniteProgram, SolveSDPwithSecondOrderConeExample1) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    SolveSDPwithSecondOrderConeExample1(scs_solver, kTol);
  }
}

GTEST_TEST(TestSemidefiniteProgram, SolveSDPwithSecondOrderConeExample2) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    SolveSDPwithSecondOrderConeExample2(scs_solver, kTol);
  }
}

GTEST_TEST(TestSemidefiniteProgram, SolveSDPwithOverlappingVariables) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    SolveSDPwithOverlappingVariables(scs_solver, kTol);
  }
}







GTEST_TEST(TestConex, UnivariateQuarticSos) {
  UnivariateQuarticSos dut;
  ConexSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-6);
  }
}

GTEST_TEST(TestConex, BivariateQuarticSos) {
  BivariateQuarticSos dut;
  ConexSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-6);
  }
}


#endif

GTEST_TEST(TestConex, SimpleSos1) {
  SimpleSos1 dut;
  ConexSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-5);
  }
}

GTEST_TEST(TestConex, UnivariateNonnegative1) {
  UnivariateNonnegative1 dut;
  ConexSolver solver;
  if (solver.is_available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-6);
  }
}




#if 0
GTEST_TEST(TestConex, MotzkinPolynomial) {
  MotzkinPolynomial dut;
  ConexSolver solver;
  if (solver.is_available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-4);
  }
}

GTEST_TEST(TestSemidefiniteProgram, CommonLyapunov) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    FindCommonLyapunov(scs_solver, kTol);
  }
}


TEST_P(TestFindSpringEquilibrium, TestSOCP) {
  ConexSolver scs_solver;
  if (scs_solver.available()) {
    SolveAndCheckSolution(scs_solver, kTol);
  }
}
INSTANTIATE_TEST_SUITE_P(
    ConexTest, TestFindSpringEquilibrium,
    ::testing::ValuesIn(GetFindSpringEquilibriumProblems()));

#endif


#if 0
// Update this to use proper options.
GTEST_TEST(TestConex, SetOptions) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddLinearConstraint(x(0) + x(1) >= 1);
  prog.AddQuadraticCost(x(0) * x(0) + x(1) * x(1));

  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    const int iter_solve = result.get_solver_details<ConexSolver>().iter;
    const int solved_status =
        result.get_solver_details<ConexSolver>().scs_status;
    DRAKE_DEMAND(iter_solve >= 2);
    SolverOptions solver_options;
    // Now we require that Conex can only take half of the iterations before
    // termination. We expect now Conex cannot solve the problem.
    solver_options.SetOption(solver.solver_id(), "max_iters", iter_solve / 2);
    solver.Solve(prog, {}, solver_options, &result);
    EXPECT_NE(result.get_solver_details<ConexSolver>().scs_status,
              solved_status);
  }
}
#endif

#if 0

#endif

}  // namespace test
}  // namespace solvers
}  // namespace drake