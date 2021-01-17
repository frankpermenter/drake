#include "drake/solvers/conex_solver.h"

#include <Eigen/Sparse>
#include <fmt/format.h>

#include <numeric>      
#include <algorithm>

// clang-format off
// scs.h should be included before linsys/amatrix.h, since amatrix.h uses types
// scs_float, scs_int, etc, defined in scs.h
// clang-format on

#include "drake/common/text_logging.h"
#include "drake/math/eigen_sparse_triplet.h"
#include "drake/math/quadratic_form.h"
#include "drake/solvers/mathematical_program.h"


#include "conex/cone_program.h"
#include "conex/debug_macros.h"
#include "conex/linear_constraint.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/quadratic_cone_constraint.h"

namespace drake {
namespace solvers {

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& P) {
  for (auto e : P) {
    os << e << "\n\n";
  }
  return os;
}

using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;

bool ConexSolver::is_available() { return true; }

namespace {

vector<int> sort_indices(const vector<int> &v) {
  // initialize original index locations
  vector<int> idx(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    idx[i] = i;
  }

  sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2];});
  return idx;
}


std::vector<int> GetDecisionVarIndex(const MathematicalProgram& prog, 
                                     const VectorXDecisionVariable& x) {
  std::vector<int> y(x.rows());
  for (int i = 0 ; i < x.rows(); i++) {
    y[i] = prog.FindDecisionVariableIndex(x(i));
  }
  return y;
}

Eigen::MatrixXd ExtractRows(const Eigen::MatrixXd A, const std::vector<int> rows) {
  Eigen::MatrixXd B(rows.size(), A.cols());
  int row_count = 0;
  for (auto i : rows) {
    B.row(row_count++) = A.row(i);  
  }
  return B;
}

void ParseLinearCost(const MathematicalProgram& prog, Eigen::VectorXd* c,
                     double* constant) {
  for (const auto& linear_cost : prog.linear_costs()) {
    // Each linear cost is in the form of aᵀx + b
    const auto& a = linear_cost.evaluator()->a();
    const VectorXDecisionVariable& x = linear_cost.variables();
    for (int i = 0; i < a.rows(); ++i) {
      (*c)(prog.FindDecisionVariableIndex(x(i))) += a(i);
    }
    (*constant) += linear_cost.evaluator()->b();
  }
}

void ParseLinearConstraint(const MathematicalProgram& prog, conex::Program* conex_prog) {
  for (const auto& linear_constraint : prog.linear_constraints()) {
    const Eigen::VectorXd& ub = linear_constraint.evaluator()->upper_bound();
    const Eigen::VectorXd& lb = linear_constraint.evaluator()->lower_bound();
    const std::vector<int>& variables = GetDecisionVarIndex(prog, linear_constraint.variables());
    const Eigen::MatrixXd& Ai = linear_constraint.evaluator()->A();

    std::vector<int> upper_bound_rows;
    std::vector<int> lower_bound_rows;

    for (int i = 0; i < linear_constraint.evaluator()->num_constraints(); ++i) {
      if (!std::isinf(ub(i))) {
        upper_bound_rows.push_back(i);
      }
      if (!std::isinf(lb(i))) {
        lower_bound_rows.push_back(i);
      }
    }

    if (upper_bound_rows.size() > 0) {
      conex_prog->AddConstraint(conex::LinearConstraint(ExtractRows(Ai, upper_bound_rows),
                               ExtractRows(ub, upper_bound_rows)), variables);
    }

    if (lower_bound_rows.size() > 0) {
      conex_prog->AddConstraint(conex::LinearConstraint(-ExtractRows(Ai, lower_bound_rows),
                               -ExtractRows(lb, lower_bound_rows)), variables);
    }
  }
}

void ParseBoundingBoxConstraint(const MathematicalProgram& prog,
                                     conex::Program* conex_prog) {
  for (const auto& bounding_box_constraint : prog.bounding_box_constraints()) {
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    std::vector<int> lower_bounded_variables;
    std::vector<int> upper_bounded_variables;
    const std::vector<int> variable = GetDecisionVarIndex(prog, bounding_box_constraint.variables());
    for (size_t i = 0; i < variable.size(); ++i) {
      if (!std::isinf(bounding_box_constraint.evaluator()->upper_bound()(i))) {
        upper_bounds.emplace_back(bounding_box_constraint.evaluator()->upper_bound()(i));
        upper_bounded_variables.push_back(variable.at(i));
      }
      if (!std::isinf(bounding_box_constraint.evaluator()->lower_bound()(i))) {
        lower_bounds.emplace_back(bounding_box_constraint.evaluator()->lower_bound()(i));
        lower_bounded_variables.push_back(variable.at(i));
      }
    }
    if (upper_bounds.size() > 0) {
      //DRAKE_DEMAND(conex_prog->AddConstraint(conex::UpperBound(Eigen::Map<VectorXd>(upper_bounds.data(), upper_bounds.size())), 
      //                          upper_bounded_variables));
      conex_prog->AddConstraint(conex::UpperBound(Eigen::Map<VectorXd>(upper_bounds.data(), upper_bounds.size())), 
                                upper_bounded_variables);
    }
    if (lower_bounds.size() > 0) {
      //DRAKE_DEMAND(conex_prog->AddConstraint(conex::LowerBound(Eigen::Map<VectorXd>(lower_bounds.data(), lower_bounds.size())), 
      //                          lower_bounded_variables));
      conex_prog->AddConstraint(conex::LowerBound(Eigen::Map<VectorXd>(lower_bounds.data(), lower_bounds.size())), 
                                lower_bounded_variables);
    }
  }
}

void ParseSecondOrderConeConstraints(const MathematicalProgram& prog,
                                     conex::Program* conex_prog) {
  for (const auto& lorentz_cone_constraint : prog.lorentz_cone_constraints()) {
    const VectorXDecisionVariable& x = lorentz_cone_constraint.variables();
    const std::vector<int> x_indices = prog.FindDecisionVariableIndices(x);
    const MatrixXd Ai = -lorentz_cone_constraint.evaluator()->A();
    const MatrixXd bi = lorentz_cone_constraint.evaluator()->b();
    conex_prog->AddConstraint(conex::QuadraticConstraint(Ai, bi), x_indices);
  }

  for (const auto& constraint :
       prog.rotated_lorentz_cone_constraints()) {
    const VectorXDecisionVariable& x = constraint.variables();
    const std::vector<int> x_indices = prog.FindDecisionVariableIndices(x);
    MatrixXd Atemp = constraint.evaluator()->A();
    MatrixXd btemp = constraint.evaluator()->b();
    MatrixXd A = Atemp; MatrixXd b = btemp;
    A.row(0) = .5*(Atemp.row(0) + Atemp.row(1));
    b.row(0) = .5*(b.row(0) + b.row(1));
    A.row(1) = .5*(Atemp.row(0) - Atemp.row(1));
    b.row(1) = .5*(b.row(0) - b.row(1));
    A.array() *= -1;
    conex_prog->AddConstraint(conex::QuadraticConstraint(A, b), x_indices);
  }
}

void ParseLinearEqualityConstraint(const MathematicalProgram& prog,
                                     conex::Program* conex_prog) {

  for (const auto& constraint : prog.linear_equality_constraints()) {
    const VectorXDecisionVariable& x = constraint.variables();
    const std::vector<int> x_indices = prog.FindDecisionVariableIndices(x);
    const MatrixXd Atemp = constraint.evaluator()->A();
    const MatrixXd btemp = constraint.evaluator()->lower_bound();
    conex_prog->AddConstraint(conex::EqualityConstraints(Atemp, btemp), x_indices);
  }
}

void ParsePositiveSemidefiniteConstraint(const MathematicalProgram& prog,
                                     conex::Program* conex_prog) {
  DRAKE_DEMAND(prog.positive_semidefinite_constraints().size() == 0);
  // TODO(FrankPermenter): Add support for these constraints.
  //  for (const auto& psd_constraint : prog.positive_semidefinite_constraints()) {
  //  }


  for (const auto& lmi_constraint :
       prog.linear_matrix_inequality_constraints()) {
    const std::vector<Eigen::MatrixXd>& F = lmi_constraint.evaluator()->F();
    const VectorXDecisionVariable& x = lmi_constraint.variables();
    std::vector<int> x_indices = prog.FindDecisionVariableIndices(x);
    auto x_order = sort_indices(x_indices);

    std::vector<Eigen::MatrixXd> A(F.size() - 1);
    Eigen::MatrixXd C = F.at(0);
    for (size_t i = 0;  i < F.size() - 1; i++) {
      A[i] = -F.at(x_order.at(i) + 1);
    }
    sort(x_indices.begin(), x_indices.end());
    conex_prog->AddConstraint(conex::DenseLMIConstraint(A, C), x_indices);
  }
}

void ParseQuadraticCost(const MathematicalProgram& prog,
                        Eigen::VectorXd* linear_cost,
                        conex::Program* conex_prog) {
  int count = 0;

  for (const auto& cost : prog.quadratic_costs()) {
    const VectorXDecisionVariable& z = cost.variables();
    std::vector<int> z_indices = prog.FindDecisionVariableIndices(z);
    for (int i = 0; i < z.rows(); i++) {
      (*linear_cost)(z_indices.at(i)) += cost.evaluator()->b()(i);
    }

    // Set inner-product matrix Q of Lorentz cone L.
    Eigen::MatrixXd Q(z.rows() + 1, z.rows() + 1);
    Q.setZero(); Q(0, 0) = 1;
    Q.bottomRightCorner(z.rows(), z.rows()) << cost.evaluator()->Q();

    // Build (A, b) satisfying b - A(x, t) \in L <=> t >= 1/2 x^T Q x.
    Eigen::MatrixXd A(z.rows() + 2, z.rows() + 1);
    Eigen::MatrixXd b(z.rows() + 2, 1);
    A.setZero(); b.setZero();
    A.topRightCorner(2, 1) << -0.5, -0.5; 
    A.bottomLeftCorner(z.rows(), z.rows()) = Eigen::MatrixXd::Identity(z.rows(), z.rows());
    b(0) = 1; b(1) = -1;

    // (.5 t+1)^2 >= (.5t-1)^2 + x^T Q x.
    // .25 t^2 + t + 1  >= .25 t^2 - t + 1 + x^T Q x
    // => 2t >= x^T Q x.

    z_indices.push_back(prog.num_vars() + count);
    conex_prog->AddConstraint(conex::QuadraticConstraint(Q, A, b), z_indices);
    count++;
  }
}
}  // namespace

void ConexSolver::DoSolve(
    const MathematicalProgram& prog,
    const Eigen::VectorXd& /*initial_guess*/,
    const SolverOptions& /*merged_options*/,
    MathematicalProgramResult* result) const {
  if (!prog.GetVariableScaling().empty()) {
    static const logging::Warn log_once(
      "ConexSolver doesn't support the feature of variable scaling.");
  }

  int num_vars = prog.num_vars();
  int num_epigraph_parameters = prog.quadratic_costs().size();

  conex::Program conex_prog(num_vars + num_epigraph_parameters);

  // Our cost (LinearCost, QuadraticCost, etc) also allows a constant term, we
  // add these constant terms to `cost_constant`.
  double cost_constant{0};
  // Parse linear cost
  Eigen::VectorXd c(num_vars + num_epigraph_parameters); c.setZero();
  Eigen::VectorXd x(num_vars + num_epigraph_parameters);
  ParseLinearCost(prog, &c, &cost_constant);
  if (num_epigraph_parameters > 0) {
    c.tail(num_epigraph_parameters).array() = 1;
  }
  ParseQuadraticCost(prog, &c, &conex_prog);
  ParseLinearEqualityConstraint(prog, &conex_prog);
  ParseBoundingBoxConstraint(prog, &conex_prog); 
  ParseLinearConstraint(prog, &conex_prog);
  ParsePositiveSemidefiniteConstraint(prog, &conex_prog);
  ParseSecondOrderConeConstraints(prog, &conex_prog);

  conex::SolverConfiguration config;
  config.prepare_dual_variables = 1;
  config.max_iterations = 25;
  config.divergence_upper_bound = 1000;
  config.final_centering_steps = 0;
  config.inv_sqrt_mu_max = 100000;
  SolutionResult solution_result{SolutionResult::kSolutionFound};
  if (!conex::Solve(-c, conex_prog, config, x.data())) {
    solution_result = SolutionResult::kInfeasibleConstraints;
  }
  result->set_x_val(x.head(num_vars));
  result->set_solution_result(solution_result);
  result->set_optimal_cost(c.dot(x) + cost_constant);
}

}  // namespace solvers
}  // namespace drake
