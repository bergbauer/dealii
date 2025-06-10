// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_cgal_polygon_h
#define dealii_cgal_polygon_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/cgal/point_conversion.h>
#  include <deal.II/cgal/utilities.h>

#  include <CGAL/Polygon_2.h>
#  include <CGAL/Polygon_with_holes_2.h>


DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  template <typename KernelType>
  void
  dealii_cell_to_cgal_polygon(
    const typename Triangulation<2, 2>::cell_iterator &cell,
    const Mapping<2, 2>                               &mapping,
    CGAL::Polygon_2<KernelType>                       &polygon);



  template <typename KernelType>
  void
  dealii_tria_to_cgal_polygon(const Triangulation<2, 2>   &tria,
                              CGAL::Polygon_2<KernelType> &fitted_2D_mesh);


  template <typename KernelType>
  void
  compute_boolean_operation(
    const CGAL::Polygon_2<KernelType>                   &polygon_1,
    const CGAL::Polygon_2<KernelType>                   &polygon_2,
    const BooleanOperation                              &boolean_operation,
    std::vector<CGAL::Polygon_with_holes_2<KernelType>> &polygon_out);
} // namespace CGALWrappers

DEAL_II_NAMESPACE_CLOSE

#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif
#endif
