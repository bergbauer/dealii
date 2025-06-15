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

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/cgal/point_conversion.h>

#  include <CGAL/Polygon_2.h>


// only for compute boolean operation
#  include <deal.II/cgal/utilities.h>

#  include <CGAL/Boolean_set_operations_2.h>
#  include <CGAL/Polygon_with_holes_2.h>


DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  using K = CGAL::Exact_predicates_exact_constructions_kernel;

  using CGALPoint2           = CGAL::Point_2<K>;
  using CGALPolygon          = CGAL::Polygon_2<K>;
  using CGALPolygonWithHoles = CGAL::Polygon_with_holes_2<K>;

  void
  dealii_cell_to_cgal_polygon(
    const typename Triangulation<2, 2>::cell_iterator &cell,
    const Mapping<2, 2>                               &mapping,
    CGALPolygon                                       &polygon)
  {
    const auto &vertices = mapping.get_vertices(cell);
    polygon.clear();
    if (cell->reference_cell() == ReferenceCells::Triangle)
      {
        polygon.push_back(
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[0]));
        polygon.push_back(
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[1]));
        polygon.push_back(
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[2]));
      }
    else if (cell->reference_cell() == ReferenceCells::Quadrilateral)
      {
        polygon.push_back(
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[0]));
        polygon.push_back(
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[1]));
        polygon.push_back(
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[3]));
        polygon.push_back(
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[2]));
      }
    else
      {
        DEAL_II_ASSERT_UNREACHABLE();
      }
  }



  void
  dealii_tria_to_cgal_polygon(const Triangulation<2, 2> &tria,
                              CGALPolygon               &fitted_2D_mesh)
  {
    std::map<unsigned int, unsigned int> face_vertex_indices;

    for (const auto &cell : tria.active_cell_iterators())
      {
        for (const unsigned int i : cell->face_indices())
          {
            const typename Triangulation<2, 2>::face_iterator &face =
              cell->face(i);
            if (face->at_boundary())
              {
                if (cell->reference_cell() == ReferenceCells::Quadrilateral &&
                    (i == 0 || i == 3))
                  {
                    face_vertex_indices[face->vertex_index(1)] =
                      face->vertex_index(0);
                  }
                else
                  {
                    face_vertex_indices[face->vertex_index(0)] =
                      face->vertex_index(1);
                  }
              }
          }
      }

    const auto &vertices = tria.get_vertices();
    fitted_2D_mesh.clear();

    unsigned int current_index = face_vertex_indices.begin()->first;

    for (size_t i = face_vertex_indices.size(); i > 0; --i)
      {
        fitted_2D_mesh.push_back(
          dealii_point_to_cgal_point<CGALPoint2, 2>(vertices[current_index]));
        auto it       = face_vertex_indices.find(current_index);
        current_index = it->second;
        face_vertex_indices.erase(it);
      }
  }



  void
  compute_boolean_operation(const CGALPolygon      &polygon_1,
                            const CGALPolygon      &polygon_2,
                            const BooleanOperation &boolean_operation,
                            std::vector<CGALPolygonWithHoles> &polygon_out)
  {
    Assert(!(boolean_operation == BooleanOperation::compute_corefinement),
           ExcMessage("Corefinement no usecase for 2D polygons"));

    polygon_out.clear();

    if (boolean_operation == BooleanOperation::compute_intersection)
      {
        CGAL::intersection(polygon_1,
                           polygon_2,
                           std::back_inserter(polygon_out));
      }
    else if (boolean_operation == BooleanOperation::compute_difference)
      {
        CGAL::difference(polygon_1, polygon_2, std::back_inserter(polygon_out));
      }
    else if ((boolean_operation == BooleanOperation::compute_union))
      {
        polygon_out.resize(1);
        CGAL::join(polygon_1, polygon_2, polygon_out[0]);
      }
  }
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
