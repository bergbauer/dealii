// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Regression test for MatrixFreeTools::compute_diagonal() in the portable
// thread-private path. The test forces a multi-cell-per-team launch so that
// the per-thread shared-memory partitioning is exercised independently of the
// backend's Kokkos::AUTO team-size choice.

#include "compute_diagonal_util.h"


template <int dim,
          int fe_degree,
          int n_points     = fe_degree + 1,
          int n_components = 1,
          typename Number  = double>
void
compute_diagonal_forced_thread_private(
  const Portable::MatrixFree<dim, Number>                 &matrix_free,
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
    &diagonal_global)
{
  constexpr int forced_team_size = 2;

  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
  using QuadOp = LaplaceOperatorQuad<dim, fe_degree, n_components, Number>;
  using CellAction = MatrixFreeTools::internal::ComputeDiagonalCellAction<
    dim,
    fe_degree,
    n_points,
    n_components,
    Number,
    QuadOp>;
  using Kernel = Portable::internal::ApplyKernel<dim, Number, CellAction, false>;
  using Selection = typename Kernel::CellSelection;
  using ExecSpace = MemorySpace::Default::kokkos_space::execution_space;

  matrix_free.initialize_dof_vector(diagonal_global);
  diagonal_global = Number();

  VectorType dummy_src;
  matrix_free.initialize_dof_vector(dummy_src);
  dummy_src = Number();

  CellAction cell_action(QuadOp{},
                         EvaluationFlags::gradients,
                         EvaluationFlags::gradients);

  ExecSpace exec;
  const auto &graph = matrix_free.get_colored_graph();

  for (unsigned int color = 0; color < graph.size(); ++color)
    {
      const auto color_data = matrix_free.get_data(color);
      if (color_data.n_cells == 0)
        continue;

      const auto launch_pass = [&](const bool launch_thread_per_cell,
                                   const Selection selection) {
        Kernel apply_kernel(cell_action,
                            color_data,
                            dummy_src,
                            diagonal_global,
                            launch_thread_per_cell,
                            selection);

        int team_size = 1;
        if (launch_thread_per_cell)
          {
            Kokkos::TeamPolicy<ExecSpace> probe_policy(exec,
                                                       color_data.n_cells,
                                                       Kokkos::AUTO);
            const int max_team_size =
              probe_policy.team_size_max(apply_kernel, Kokkos::ParallelForTag());
            team_size = std::max(1, std::min(forced_team_size, max_team_size));
          }

        Kokkos::parallel_for(
          "dealii::tests::compute_diagonal_thread_private_01",
          Kokkos::TeamPolicy<ExecSpace>(exec, color_data.n_cells, team_size),
          apply_kernel);
      };

      if (color_data.has_constrained_dofs)
        {
          launch_pass(true, Selection::unconstrained_only);
          launch_pass(false, Selection::constrained_only);
        }
      else
        launch_pass(true, Selection::all);
    }

  Kokkos::fence();
  matrix_free.set_constrained_values(Number(1.), diagonal_global);
}



template <int dim,
          int fe_degree,
          int n_points     = fe_degree + 1,
          int n_components = 1,
          typename Number  = double>
void
test()
{
  Triangulation<dim> tria;

  GridGenerator::hyper_cube(tria, -numbers::PI / 2, numbers::PI / 2);

  tria.refine_global(2);
  for (auto &cell : tria.active_cell_iterators())
    if (cell->is_active() && cell->center()[0] < 0.0)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  AssertThrow(tria.n_active_cells() > 2, ExcInternalError());

  const FE_Q<dim>     fe_q(fe_degree);
  const FESystem<dim> fe(fe_q, n_components);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(
    dof_handler,
    0,
    Functions::ZeroFunction<dim, Number>(n_components),
    constraints);
  constraints.close();

  typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data(
    update_values | update_gradients,
    false,
    false,
    true);

  MappingQ<dim> mapping(1);
  QGauss<1>     quad(fe_degree + 1);

  Portable::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quad, additional_data);

  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
    diagonal_device;
  compute_diagonal_forced_thread_private<dim,
                                         fe_degree,
                                         n_points,
                                         n_components,
                                         Number>(matrix_free, diagonal_device);

  LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> diagonal_host;
  matrix_free.initialize_dof_vector(diagonal_host);
  LinearAlgebra::ReadWriteVector<Number> rw_vector(
    diagonal_device.get_partitioner()->locally_owned_range());
  rw_vector.import_elements(diagonal_device, VectorOperation::insert);
  diagonal_host.import_elements(rw_vector, VectorOperation::insert);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<Number> reference_matrix;
  reference_matrix.reinit(sparsity_pattern);

  Function<dim, Number> *scaling = nullptr;
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe_degree + 1),
                                       reference_matrix,
                                       scaling,
                                       constraints);

  Number max_error = Number();
  for (unsigned int i = 0; i < diagonal_host.size(); ++i)
    {
      const Number expected =
        constraints.is_constrained(i) ? Number(1.) : reference_matrix(i, i);
      max_error = std::max(max_error,
                           std::abs(expected - diagonal_host.local_element(i)));
    }

  const Number tolerance =
    std::is_same_v<Number, double> ? Number(1e-12) : Number(5e-5);

  AssertThrow(max_error < tolerance,
              ExcMessage("Thread-private diagonal mismatch: " +
                         std::to_string(max_error)));

  deallog << "dim=" << dim << ", fe_degree=" << fe_degree << ", Number="
          << (std::is_same_v<Number, double> ? "double" : "float")
          << ", max_error=" << max_error << std::endl;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();
  deallog << std::scientific << std::setprecision(3);

  test<2, 1, 2, 1>();
  test<2, 1, 2, 1, float>();
  test<3, 1, 2, 1>();
  test<3, 1, 2, 1, float>();
}