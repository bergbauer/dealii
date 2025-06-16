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
  /**
   * Build a CGAL::Polygon_2 from a deal.II cell.
   *
   * The class Polygon_2 is a wrapper around a container of points that can
   * be used to represent polygons.
   * The points must be added in counterclockwise order to a Polygon_2
   *
   * More information on this class is available at
   * https://doc.cgal.org/latest/Polygon/index.html
   *
   * The functions are for two dimensional triangulations in two dimensional
   * space. Projecting 3D points is possible with CGAL but not implemented.
   *
   * The generated boundary representation is useful when performing
   * geometric operations using compute boolean operations.
   *
   * @param[in] cell The input deal.II cell iterator
   * @param[in] mapping The mapping used to map the vertices of the cell
   * @param[out] polygon The output CGAL::Polygon_2
   */
  template <typename KernelType>
  void
  dealii_cell_to_cgal_polygon(
    const typename Triangulation<2, 2>::cell_iterator &cell,
    const Mapping<2, 2>                               &mapping,
    CGAL::Polygon_2<KernelType>                       &polygon);



  /**
   * Convert a deal.II triangulation to a CGAL::Polygon_2.
   *
   * Triangulations that have holes are not supported. The output
   * is a Polygon_2, the function would need to be extended.
   *
   * @param[in] tria The input deal.II triangulation
   * @param[out] polygon The output CGAL::Polygon_2
   */
  template <typename KernelType>
  void
  dealii_tria_to_cgal_polygon(const Triangulation<2, 2>   &tria,
                              CGAL::Polygon_2<KernelType> &polygon);



  /**
   * Perform a BooleanOperation on two CGAL::Polygon_2.
   *
   * The output is a vector of CGAL::Polygon_2_with_holes, since this
   * can generally be the result of a boolean operation.
   *
   * For the union the vector will always have length one.
   *
   * For the difference operation the second polygon is subtracted
   * from the first one.
   *
   * Corefinement is not supported as boolean operation.
   *
   * @param[in] polygon_1 The first input CGAL::Polygon_2
   * @param[in] polygon_2 The second input CGAL::Polygon_2
   * @param[in] boolean_operation The input BooleanOperation
   * @param[out] polygon_out The output CGAL::Polygon_2_with_holes
   */
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
