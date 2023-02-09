// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

/*
 * Test the NonMatching::MappingInfo class together with FEPointEvaluation and
 * compare to NonMatching::FEValues
 */

#include <deal.II/base/function_signed_distance.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  initlog();

  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 1;

  FE_Q<dim> fe_q(degree);

  Triangulation<dim> tria;

  MappingQ<dim> mapping(degree);

  GridGenerator::subdivided_hyper_cube(tria, 4);

  DoFHandler<dim> dof_handler(tria);

  dof_handler.distribute_dofs(fe_q);

  Functions::SignedDistance::Sphere<dim> level_set;

  Vector<double> level_set_vec(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, level_set, level_set_vec);

  NonMatching::MeshClassifier<dim> mesh_classifier(dof_handler, level_set_vec);
  mesh_classifier.reclassify();

  hp::QCollection<1> q_collection((QGauss<1>(degree)));

  NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
    q_collection, dof_handler, level_set_vec);


  // FEPointEvaluation
  NonMatching::MappingInfo<dim> mapping_info_cell(
    mapping, update_values | update_gradients | update_JxW_values);

  std::vector<Quadrature<dim>> quad_vec;
  for (const auto &cell : tria.active_cell_iterators())
    {
      quadrature_generator.generate(cell);
      quad_vec.push_back(quadrature_generator.get_inside_quadrature());
    }

  mapping_info_cell.reinit_cells(tria.active_cell_iterators(), quad_vec);

  Vector<double> src(dof_handler.n_dofs()), dst(dof_handler.n_dofs());

  for (auto &v : src)
    v = random_value<double>();

  FEPointEvaluation<1, dim, dim, double> fe_point(mapping_info_cell, fe_q);

  std::vector<double> solution_values_in(fe_q.dofs_per_cell);
  std::vector<double> solution_values_out(fe_q.dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_values(src,
                           solution_values_in.begin(),
                           solution_values_in.end());

      fe_point.reinit(cell->active_cell_index());

      fe_point.evaluate(solution_values_in,
                        EvaluationFlags::values | EvaluationFlags::gradients);

      for (unsigned int q = 0; q < fe_point.n_q_points; ++q)
        {
          fe_point.submit_value(fe_point.JxW(q) * fe_point.get_value(q), q);
          fe_point.submit_gradient(fe_point.JxW(q) * fe_point.get_gradient(q),
                                   q);
        }

      fe_point.integrate(solution_values_out,
                         EvaluationFlags::values | EvaluationFlags::gradients);

      cell->distribute_local_to_global(
        Vector<double>(solution_values_out.begin(), solution_values_out.end()),
        dst);
    }


  // FEValues
  const QGauss<1> quadrature_1D(degree);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside = update_values | update_gradients |
                               update_JxW_values | update_quadrature_points;
  region_update_flags.surface = update_values | update_gradients |
                                update_JxW_values | update_quadrature_points |
                                update_normal_vectors;

  hp::FECollection<dim> fe_collection(fe_q);

  NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                    quadrature_1D,
                                                    region_update_flags,
                                                    mesh_classifier,
                                                    dof_handler,
                                                    level_set_vec);

  Vector<double> dst_2(dof_handler.n_dofs());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      non_matching_fe_values.reinit(cell);

      const auto &inside_fe_values =
        non_matching_fe_values.get_inside_fe_values();

      if (inside_fe_values)
        {
          cell->get_dof_values(src,
                               solution_values_in.begin(),
                               solution_values_in.end());

          std::vector<Tensor<1, dim>> solution_gradients(
            inside_fe_values->n_quadrature_points);
          std::vector<double> solution_values(
            inside_fe_values->n_quadrature_points);

          for (const auto q : inside_fe_values->quadrature_point_indices())
            {
              double         values = 0.;
              Tensor<1, dim> gradients;

              for (const auto i : inside_fe_values->dof_indices())
                {
                  gradients +=
                    solution_values_in[i] * inside_fe_values->shape_grad(i, q);
                  values +=
                    solution_values_in[i] * inside_fe_values->shape_value(i, q);
                }
              solution_gradients[q] = gradients * inside_fe_values->JxW(q);
              solution_values[q]    = values * inside_fe_values->JxW(q);
            }

          for (const auto i : inside_fe_values->dof_indices())
            {
              double sum_gradients = 0.;
              double sum_values    = 0.;
              for (const auto q : inside_fe_values->quadrature_point_indices())
                {
                  sum_gradients +=
                    solution_gradients[q] * inside_fe_values->shape_grad(i, q);
                  sum_values +=
                    solution_values[q] * inside_fe_values->shape_value(i, q);
                }

              solution_values_out[i] = sum_gradients + sum_values;
            }

          cell->distribute_local_to_global(
            Vector<double>(solution_values_out.begin(),
                           solution_values_out.end()),
            dst_2);
        }
    }

  deallog << "check difference l2 norm: " << dst.l2_norm() - dst_2.l2_norm()
          << std::endl;
}
