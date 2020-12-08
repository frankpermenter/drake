This directory contains the three test cases run in the accompanying document in
`examples/multibody/mp_convex_solver/doc/convex_solver.pdf`, section 6.

We summarize here how to compile, run and obtain log data for each example.

## Conveyor Belt

Compile and run with:
```
bazel run -j 4 --config gurobi examples/multibody/mp_convex_solver:conveyor_belt -- --time_step=1e-3
```

The option `--time_step=1e-3` sets the discrete time step. In the pdf doc we use
time steps from 1e-4 and 1e-2 in a geometric sequence with common ration
Sqrt(10).

The Bazel compile options are:
- `-j4`: to limit the number of processors used to compile. Especially useful to
  avoid a small laptop run out of memory.
- `--config gurobi`: Only needed if running with the Gurobi solver (providing
  `--solver=gurobi`).

### Solution File
This program logs state and forces along with solver statistics per time step in
a file named `sol.dat`. The file is ASCII with columns for: 
`time, z, zdot, ft, fn, vr, vn, id_rel_err, id_abs_err`.

Within the Bazel environment this file will be found at
`bazel-bin/examples/multibody/mp_convex_solver/conveyor_belt.runfiles/drake/sol.dat`
and can be copied where needed.

## Sliding Box

Compile and run with:
```
bazel run -j 4 --config gurobi examples/multibody/mp_convex_solver:sliding_box -- --time_step=1e-3
```

The option `--time_step=1e-3` sets the discrete time step. In the pdf doc we use
time steps 1e-2 and 1e-3.

The Bazel compile options are:
- `-j4`: to limit the number of processors used to compile. Especially useful to
  avoid a small laptop run out of memory.
- `--config gurobi`: Only needed if running with the Gurobi solver (providing
  `--solver=gurobi`).

### Solution File
This program logs state and forces along with solver statistics per time step in
a file named `sol.dat`. The file is ASCII with columns for: 
`time, x, xdot, z, zdot, ft, fn, vr, vn, id_rel_err, id_abs_err`.

Within the Bazel environment this file will be found at
`bazel-bin/examples/multibody/mp_convex_solver/sliding_box.runfiles/drake/sol.dat`
and can be copied where needed.

## Clutter

Compile and run with:
```
bazel run -j 4 --config gurobi examples/multibody/mp_convex_solver:clutter -- --mbp_time_step=1e-2 --objects_per_pile=10 --simulator_target_realtime_rate=0 --simulation_time=4.0
```

The most relevant options are:
- `mpb_time_step`: specifies the discrete solver time step.
- `objects_per_pile`: an arbitrary assortment of objects are initialized into
  four piles, each having `objects_per_pile` objects.
- `simulation_time`: The duration of the simulation in seconds.
- `visualize_forces`: Set to `true` to enable the visualization of contact forces.

### Solution File

This test case produces two solution files:
- `sol.dat`: records state dependent statistics. These are: `time, ke, vt, vn,
  phi_plus, phi_minus`, time, kinetic energy, RMS tangential velocity, RMS
  normal velocity, mean positive SDF, mean negative SDF.
- `log.dat`: Records solver statistics, such as number of contacts, errors
  relative to the analytical inverse dynamics, and wall clock for different
  phases of the solver.


Within the Bazel environment these files will be found at
`bazel-bin/examples/multibody/mp_convex_solver/clutter.runfiles/drake`.