#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include <gflags/gflags.h>

#include "drake/common/nice_type_name.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/math/random_rotation.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.h"
#include "drake/multibody/contact_solvers/mp_convex_solver.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/systems/analysis/implicit_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace examples {
namespace mp_convex_solver {
namespace {

DEFINE_double(simulation_time, 1.0, "Simulation duration in seconds");
DEFINE_double(
    mbp_time_step, 1.0E-2,
    "If mbp_time_step > 0, the fixed-time step period (in seconds) of discrete "
    "updates for the plant (modeled as a discrete system). "
    "If mbp_time_step = 0, the plant is modeled as a continuous system "
    "and no contact forces are displayed.  mbp_time_step must be >= 0.");

// The following set of flags are only used in "continuous mode", i.e. when
// mbp_time_step = 0.
DEFINE_string(jacobian_scheme, "forward",
              "Valid Jacobian computation schemes are: "
              "'forward', 'central', or 'automatic'");
DEFINE_bool(use_full_newton, true, "Use full Newton, otherwise quasi-Newton.");
DEFINE_double(penetration_allowance, 1.0E-4, "Allowable penetration (meters).");
DEFINE_double(stiction_tolerance, 1.0E-4,
              "Allowable drift speed during stiction (m/s).");
DEFINE_bool(fixed_step, true, "Use fixed step integration. No error control.");

// Problem parameters.
DEFINE_double(friction_coefficient, 1.0,
              "All friction coefficients have this value.");
DEFINE_bool(only_collision_spheres, true,
            "Use only point contact with spheres");
DEFINE_int32(objects_per_pile, 3, "Number of objects per pile.");
DEFINE_int32(
    num_spheres, 3,
    "Multi-contact emulation. We place num_sphere x num_spheres per face.");

// Visualization.
DEFINE_bool(visualize, true, "Whether to visualize (true) or not (false).");
DEFINE_bool(visualize_forces, false,
            "Whether to visualize forces (true) or not (false).");
DEFINE_double(viz_period, 1.0 / 60.0, "Viz period.");

// Discrete contact solver.
DEFINE_bool(tamsi, false, "Use TAMSI (true) or MpConvexSolver (false).");
// If using Gurobi, compile with: bazel run --config gurobi ....
DEFINE_string(solver, "gurobi", "Underlying solver. 'gurobi', 'scs'");

using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::ContactResults;
using drake::multibody::MultibodyPlant;
using drake::multibody::contact_solvers::internal::MpConvexSolver;
using drake::multibody::contact_solvers::internal::MpConvexSolverParameters;
using drake::multibody::contact_solvers::internal::MpConvexSolverStats;
using Eigen::Translation3d;
using Eigen::Vector3d;
using clock = std::chrono::steady_clock;

// Parameters
const double width(0.8);
const double length(0.8);

std::vector<geometry::GeometryId> box_geometry_ids;

const RigidBody<double>& AddBox(const std::string& name,
                                const Vector3<double>& block_dimensions,
                                double mass, double friction,
                                const Vector4<double>& color,
                                bool add_sphere_collision,
                                bool add_box_collision,
                                MultibodyPlant<double>* plant) {
  // Ensure the block's dimensions are mass are positive.
  const double LBx = block_dimensions.x();
  const double LBy = block_dimensions.y();
  const double LBz = block_dimensions.z();

  // Describe body B's mass, center of mass, and inertia properties.
  const Vector3<double> p_BoBcm_B = Vector3<double>::Zero();
  const UnitInertia<double> G_BBcm_B =
      UnitInertia<double>::SolidBox(LBx, LBy, LBz);
  const SpatialInertia<double> M_BBcm_B(mass, p_BoBcm_B, G_BBcm_B);

  // Create a rigid body B with the mass properties of a uniform solid block.
  const RigidBody<double>& box = plant->AddRigidBody(name, M_BBcm_B);

  // Box's visual.
  // The pose X_BG of block B's geometry frame G is an identity transform.
  const RigidTransform<double> X_BG;  // Identity transform.
  plant->RegisterVisualGeometry(box, X_BG, geometry::Box(LBx, LBy, LBz),
                                name + "_visual", color);

  // Box's collision geometry is a solid box.
  if (add_sphere_collision) {
    const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
    const Vector4<double> red_50(1.0, 0.0, 0.0, 0.5);
    const double radius_x = LBx / FLAGS_num_spheres / 2.0;
    const double radius_y = LBy / FLAGS_num_spheres / 2.0;
    const double radius_z = LBz / FLAGS_num_spheres / 2.0;
    int i = 0;
    std::vector<double> x_range, y_range, z_range;
    double dx = 2 * radius_x;
    double dy = 2 * radius_y;
    double dz = 2 * radius_z;
    for (int j = 0; j < FLAGS_num_spheres; ++j) {
      x_range.push_back(-LBx / 2 + radius_x + j * dx);
      y_range.push_back(-LBy / 2 + radius_y + j * dy);
      z_range.push_back(-LBz / 2 + radius_z + j * dz);
    }
    for (double x_sign : x_range) {
      for (double y_sign : y_range) {
        for (double z_sign : z_range) {
          const std::string name_spherei =
              name + "_sphere" + std::to_string(++i) + "_collision";
          const double x = x_sign;
          const double y = y_sign;
          const double z = z_sign;
          const Vector3<double> p_BoSpherei_B(x, y, z);
          const RigidTransform<double> X_BSpherei(p_BoSpherei_B);
          geometry::Sphere shape(radius_x);
          // Ellipsoid might not be accurate. From console [warning]:
          // "Ellipsoid is primarily for ComputeContactSurfaces in hydroelastic
          // contact model. The accuracy of other collision queries and signed
          // distance queries are not guaranteed."
          // geometry::Ellipsoid shape(radius_x, radius_y, radius_z);
          plant->RegisterCollisionGeometry(
              box, X_BSpherei, shape, name_spherei,
              CoulombFriction<double>(friction, friction));
          plant->RegisterVisualGeometry(box, X_BSpherei, shape, name_spherei,
                                        red);
        }  // z
      }    // y
    }      // x
  }

  if (add_box_collision) {
    auto id = plant->RegisterCollisionGeometry(
        box, X_BG, geometry::Box(LBx, LBy, LBz), name + "_collision",
        CoulombFriction<double>(friction, friction));
    box_geometry_ids.push_back(id);
  }
  return box;
}

void AddSink(MultibodyPlant<double>* plant) {
  DRAKE_THROW_UNLESS(plant != nullptr);

  // Parameters for the sink.
  // const double length = 1.0;
  // const double width = 0.8;
  const double height = 0.4;
  const double wall_thickness = 0.04;
  const double wall_mass = 1.0;
  const double friction_coefficient = FLAGS_friction_coefficient;
  const Vector4<double> light_blue(0.5, 0.8, 1.0, 0.3);

  auto add_wall = [&](const std::string& name, const Vector3d& dimensions,
                      const RigidTransformd& X_WB) -> const RigidBody<double>& {
    const auto& wall = AddBox(name, dimensions, wall_mass, friction_coefficient,
                              light_blue, false, true, plant);
    plant->WeldFrames(plant->world_frame(), wall.body_frame(), X_WB);
    return wall;
  };

  const Vector3d bottom_dimensions(length, width, wall_thickness);
  const Vector3d side_wall_dimensions(height, width, wall_thickness);
  const Vector3d back_front_wall_dimensions(length, wall_thickness, height);

  add_wall("sink_bottom", bottom_dimensions,
           Translation3d(0, 0, -wall_thickness / 2.0));
  add_wall("sink_right", side_wall_dimensions,
           RigidTransformd(RotationMatrixd::MakeYRotation(M_PI_2),
                           Vector3d(length / 2.0, 0.0, height / 2.0)));
  add_wall("sink_left", side_wall_dimensions,
           RigidTransformd(RotationMatrixd::MakeYRotation(M_PI_2),
                           Vector3d(-length / 2.0, 0.0, height / 2.0)));
  add_wall("sink_back", back_front_wall_dimensions,
           Translation3d(0.0, width / 2, height / 2));
  add_wall("sink_front", back_front_wall_dimensions,
           Translation3d(0.0, -width / 2, height / 2));
}

const RigidBody<double>& AddSphere(const std::string& name, const double radius,
                                   double mass, double friction,
                                   const Vector4<double>& color,
                                   MultibodyPlant<double>* plant) {
  const UnitInertia<double> G_Bcm = UnitInertia<double>::SolidSphere(radius);
  const SpatialInertia<double> M_Bcm(mass, Vector3<double>::Zero(), G_Bcm);

  const RigidBody<double>& ball = plant->AddRigidBody(name, M_Bcm);

  // Add collision geometry.
  const RigidTransformd X_BS = RigidTransformd::Identity();
  plant->RegisterCollisionGeometry(ball, X_BS, geometry::Sphere(radius),
                                   name + "_collision",
                                   CoulombFriction<double>(friction, friction));

  // Add visual geometry.
  plant->RegisterVisualGeometry(ball, X_BS, geometry::Sphere(radius),
                                name + "_visual", color);

  // We add a few spots so that we can appreciate the sphere's
  // rotation, colored on red, green, blue according to the body's axes.
  const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
  const Vector4<double> green(0.0, 1.0, 0.0, 1.0);
  const Vector4<double> blue(0.0, 0.0, 1.0, 1.0);
  const double visual_radius = 0.2 * radius;
  const geometry::Cylinder spot(visual_radius, visual_radius);
  // N.B. We do not place the cylinder's cap exactly on the sphere surface to
  // avoid visualization artifacts when the surfaces are kissing.
  const double radial_offset = radius - 0.45 * visual_radius;
  auto spot_pose = [](const Vector3<double>& position) {
    // The cylinder's z-axis is defined as the normalized vector from the
    // sphere's origin to the cylinder's center `position`.
    const Vector3<double> axis = position.normalized();
    return RigidTransformd(
        Eigen::Quaterniond::FromTwoVectors(Vector3<double>::UnitZ(), axis),
        position);
  };
  plant->RegisterVisualGeometry(ball, spot_pose({radial_offset, 0., 0.}), spot,
                                name + "_x+", red);
  plant->RegisterVisualGeometry(ball, spot_pose({-radial_offset, 0., 0.}), spot,
                                name + "_x-", red);
  plant->RegisterVisualGeometry(ball, spot_pose({0., radial_offset, 0.}), spot,
                                name + "_y+", green);
  plant->RegisterVisualGeometry(ball, spot_pose({0., -radial_offset, 0.}), spot,
                                name + "_y-", green);
  plant->RegisterVisualGeometry(ball, spot_pose({0., 0., radial_offset}), spot,
                                name + "_z+", blue);
  plant->RegisterVisualGeometry(ball, spot_pose({0., 0., -radial_offset}), spot,
                                name + "_z-", blue);
  return ball;
}

std::vector<BodyIndex> AddObjects(MultibodyPlant<double>* plant) {
  const double radius = 0.05;
  const double mass = 0.2;
  const double friction = FLAGS_friction_coefficient;
  const Vector4<double> orange(1.0, 0.55, 0.0, 1.0);
  const Vector4<double> purple(204.0 / 255, 0.0, 204.0 / 255, 1.0);
  const Vector4<double> green(0, 153.0 / 255, 0, 1.0);
  const Vector4<double> cyan(51 / 255, 1.0, 1.0, 1.0);
  const Vector4<double> pink(1.0, 204.0 / 255, 204.0 / 255, 1.0);
  std::vector<Vector4<double>> colors;
  colors.push_back(orange);
  colors.push_back(purple);
  colors.push_back(green);
  colors.push_back(cyan);
  colors.push_back(pink);

  const int seed = 41;
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(0, 1);

  auto roll_shape = [&]() { return distribution(generator); };

  const int num_objects = FLAGS_objects_per_pile;
  const int num_bodies = plant->num_bodies();

  // const Vector3d box_size(4*radius, radius/4, radius);
  const Vector3d box_size = 2 * radius * Vector3d::Ones();

  std::vector<BodyIndex> bodies;
  for (int i = 1; i <= num_objects; ++i) {
    const auto& color = colors[(i - 1) % colors.size()];
    const std::string name = "object" + std::to_string(i + num_bodies);
    switch (roll_shape()) {
      case 0:
        bodies.push_back(
            AddSphere(name, radius, mass, friction, color, plant).index());
        break;
      case 1:
        Vector4<double> color50(color);
        color50.z() = 0.5;
        bodies.push_back(AddBox(name, box_size, mass, friction, color50,
                                FLAGS_only_collision_spheres, true, plant)
                             .index());
        break;
    }
  }

  return bodies;
}

void SetObjectsIntoAPile(const MultibodyPlant<double>& plant,
                         const Vector3d& offset,
                         const std::vector<BodyIndex>& bodies,
                         systems::Context<double>* plant_context) {
  const double delta_z = 0.15;  // assume objects have a BB of about 10 cm.

  const int seed = 41;
  std::mt19937 generator(seed);

  double z = delta_z / 2;
  for (auto body_index : bodies) {
    const auto& body = plant.get_body(body_index);
    if (body.is_floating()) {
      const RotationMatrixd R_WB =
          math::UniformlyRandomRotationMatrix<double>(&generator);
      const Vector3d p_WB = offset + Vector3d(0.0, 0.0, z);

      plant.SetFreeBodyPose(plant_context, body, RigidTransformd(R_WB, p_WB));
      z += delta_z;
    }
  }
}

int do_main() {
  // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, FLAGS_mbp_time_step);

  AddSink(&plant);

  // AddSphere("sphere", radius, mass, friction, orange, &plant);
  auto pile1 = AddObjects(&plant);
  auto pile2 = AddObjects(&plant);
  auto pile3 = AddObjects(&plant);
  auto pile4 = AddObjects(&plant);

  // Only box-sphere and sphere-sphere are allowed.
  geometry::GeometrySet all_boxes(box_geometry_ids);
  scene_graph.ExcludeCollisionsWithin(all_boxes);

  plant.Finalize();
  plant.set_penetration_allowance(FLAGS_penetration_allowance);

  // Set the speed tolerance (m/s) for the underlying Stribeck friction model
  // (the allowable drift speed during stiction).  For two points in contact,
  // this is the maximum sliding speed for the points to be regarded as
  // stationary relative to each other (so that static friction is used).
  plant.set_stiction_tolerance(FLAGS_stiction_tolerance);

  // ConvexBarrierSolver<double>* solver{nullptr};
  MpConvexSolver<double>* solver{nullptr};
  if (!FLAGS_tamsi) {
    solver =
        &plant.set_contact_solver(std::make_unique<MpConvexSolver<double>>());
    MpConvexSolverParameters params;
    params.alpha = 0.1;
    params.Rt_factor = 0.1;
    // Opopt: It fails very often.
    // params.solver_id = solvers::IpoptSolver::id();

    // Nlopt: "converges", but analytical ID errors are large.
    // params.solver_id = solvers::NloptSolver::id();

    if (FLAGS_solver == "scs") {
      // ScsSolver: Shows good performance/convergence.
      params.solver_id = solvers::ScsSolver::id();
    } else if (FLAGS_solver == "gurobi") {
      // GurobiSolver.
      // Compile with: bazel run --config gurobi ....
      params.solver_id = solvers::GurobiSolver::id();
    } else {
      throw std::runtime_error("Solver not supported.");
    }
    solver->set_parameters(params);
  }

  fmt::print("Num positions: {:d}\n", plant.num_positions());
  fmt::print("Num velocities: {:d}\n", plant.num_velocities());

  // Publish contact results for visualization.
  if (FLAGS_visualize) {
    geometry::DrakeVisualizerParams viz_params;
    viz_params.publish_period = FLAGS_viz_period;
    geometry::DrakeVisualizer::AddToBuilder(&builder, scene_graph, nullptr,
                                            viz_params);
  }
  if (FLAGS_visualize_forces) {
    ConnectContactResultsToDrakeVisualizer(&builder, plant);
  }
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // In the plant's default context, we assume the state of body B in world W is
  // such that X_WB is an identity transform and B's spatial velocity is zero.
  plant.SetDefaultContext(&plant_context);

  SetObjectsIntoAPile(plant, Vector3d(length / 4, width / 4, 0), pile1,
                      &plant_context);

  SetObjectsIntoAPile(plant, Vector3d(-length / 4, width / 4, 0), pile2,
                      &plant_context);

  SetObjectsIntoAPile(plant, Vector3d(-length / 4, -width / 4, 0), pile3,
                      &plant_context);
  SetObjectsIntoAPile(plant, Vector3d(length / 4, -width / 4, 0), pile4,
                      &plant_context);

  auto simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  systems::IntegratorBase<double>& integrator =
      simulator->get_mutable_integrator();
  auto* implicit_integrator =
      dynamic_cast<systems::ImplicitIntegrator<double>*>(&integrator);
  if (implicit_integrator) {
    if (FLAGS_jacobian_scheme == "forward") {
      implicit_integrator->set_jacobian_computation_scheme(
          systems::ImplicitIntegrator<
              double>::JacobianComputationScheme::kForwardDifference);
    } else if (FLAGS_jacobian_scheme == "central") {
      implicit_integrator->set_jacobian_computation_scheme(
          systems::ImplicitIntegrator<
              double>::JacobianComputationScheme::kCentralDifference);
    } else if (FLAGS_jacobian_scheme == "automatic") {
      implicit_integrator->set_jacobian_computation_scheme(
          systems::ImplicitIntegrator<
              double>::JacobianComputationScheme::kAutomatic);
    } else {
      throw std::runtime_error("Invalid Jacobian computation scheme");
    }
    implicit_integrator->set_use_full_newton(FLAGS_use_full_newton);
  }
  if (integrator.supports_error_estimation())
    integrator.set_fixed_step_mode(FLAGS_fixed_step);

  std::ofstream sol_file("sol.dat");
  simulator->set_monitor([&](const systems::Context<double>& root_context) {
    const systems::Context<double>& ctxt =
        plant.GetMyContextFromRoot(root_context);
    const ContactResults<double>& contact_results =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            ctxt);
    const int nc = contact_results.num_point_pair_contacts();

    const double ke = plant.CalcKineticEnergy(ctxt);

    double vt_rms = 0;
    double vn_rms = 0;
    int num_positive_phi = 0;
    double mean_positive_phi = 0;
    int num_negative_phi = 0;
    double mean_negative_phi = 0;
    for (int ic = 0; ic < nc; ++ic) {
      const PointPairContactInfo<double>& info =
          contact_results.point_pair_contact_info(ic);
      vt_rms += (info.slip_speed() * info.slip_speed());
      vn_rms += (info.separation_speed() * info.separation_speed());
      const drake::geometry::PenetrationAsPointPair<double>& pp =
          info.point_pair();
      if (pp.depth > 0) {
        mean_negative_phi += pp.depth;
        num_negative_phi++;
      } else {
        num_positive_phi++;
        mean_positive_phi -= pp.depth;
      }
    }
    if (nc > 0) vt_rms = std::sqrt(vt_rms / nc);
    if (nc > 0) vn_rms = std::sqrt(vn_rms / nc);
    if (num_positive_phi > 0) mean_positive_phi /= num_positive_phi;
    if (num_negative_phi > 0) mean_negative_phi /= num_negative_phi;

    // time, ke, vt, vn, phi_plus, phi_minus.
    sol_file << fmt::format(
        "{:20.8g} {:d} {:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g}\n",
        ctxt.get_time(), nc, ke, vt_rms, vn_rms, mean_positive_phi,
        mean_negative_phi);
    return systems::EventStatus::Succeeded();
  });

  clock::time_point sim_start_time = clock::now();
  simulator->AdvanceTo(FLAGS_simulation_time);
  clock::time_point sim_end_time = clock::now();
  const double sim_time =
      std::chrono::duration<double>(sim_end_time - sim_start_time).count();
  std::cout << "Sim Time [sec]: " << sim_time << std::endl;
  sol_file.close();

  // Print contact solver stats.
  const std::vector<MpConvexSolverStats>& stats_hist =
      solver->get_stats_history();
  std::ofstream file("log.dat");
  file << fmt::format(
      "{:>18} {:>18} {:>18}  {:>18}  {:>18}  {:>18}  {:>18}  {:>18}\n",
      "num_contacts", "id_rel_err", "id_abs_error", "gamma_norm", "total_time",
      "preproc_time", "mp_setup_time", "sover_time");
  for (const auto& s : stats_hist) {
    file << fmt::format(
        "{:d} {:18.6g} {:18.6g} {:18.6g} {:18.6g} {:18.6g} {:18.6g} {:18.6g}\n",
        s.num_contacts, s.iteration_errors.id_rel_error,
        s.iteration_errors.id_abs_error, s.iteration_errors.gamma_norm,
        s.total_time, s.preproc_time, s.mp_setup_time, s.solver_time);
  }
  file.close();

  PrintSimulatorStatistics(*simulator);

  return 0;
}

}  // namespace
}  // namespace mp_convex_solver
}  // namespace examples
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "\nSimulation of a clutter of objects falling into a box container.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::mp_convex_solver::do_main();
}
