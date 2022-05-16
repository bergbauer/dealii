// ---------------------------------------------------------------------
//
// Copyright (C) 2011 - 2021 by the deal.II authors
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

#ifndef dealii_matrix_free_fe_point_templates_h
#define dealii_matrix_free_fe_point_templates_h


#include <deal.II/base/config.h>

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/polynomials_piecewise.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/fe_q_dg0.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/matrix_free/face_info.h>
#include <deal.II/matrix_free/face_setup_internal.h>
#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/matrix_free_fe_point.h>

#include <fstream>

DEAL_II_NAMESPACE_OPEN



// --------------------- MatrixFree -----------------------------------

template <int dim, typename Number, typename VectorizedArrayType>
MatrixFreeFEPoint<dim, Number, VectorizedArrayType>::MatrixFreeFEPoint()
  : Subscriptor()
  , indices_are_initialized(false)
  , mapping_is_initialized(false)
  , mg_level(numbers::invalid_unsigned_int)
{}



template <int dim, typename Number, typename VectorizedArrayType>
MatrixFreeFEPoint<dim, Number, VectorizedArrayType>::MatrixFreeFEPoint(
  const MatrixFreeFEPoint<dim, Number, VectorizedArrayType> &other)
  : Subscriptor()
{
  copy_from(other);
}

DEAL_II_NAMESPACE_CLOSE

#endif
