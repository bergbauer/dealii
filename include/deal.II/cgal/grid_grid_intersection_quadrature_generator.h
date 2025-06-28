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

#ifndef dealii_grid_grid_intersection_quadrature_generator_h
#define dealii_grid_grid_intersection_quadrature_generator_h

#include <deal.II/base/config.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping.h> //think of this and mapping header

#include <deal.II/non_matching/immersed_surface_quadrature.h>
#include <deal.II/non_matching/mesh_classifier.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/cgal/surface_mesh.h>
#  include <deal.II/cgal/utilities.h>

#  include <CGAL/Boolean_set_operations_2.h>
#  include <CGAL/Constrained_Delaunay_triangulation_2.h>
#  include <CGAL/Delaunay_triangulation_2.h>
#  include <CGAL/Partition_traits_2.h>
#  include <CGAL/Polygon_mesh_processing/clip.h>
#  include <CGAL/Polygon_with_holes_2.h>
#  include <CGAL/Side_of_triangle_mesh.h>
#  include <CGAL/intersections.h>
#  include <CGAL/partition_2.h>

#  include "polygon.h"

// output
#  include <deal.II/base/timer.h>

#  include <CGAL/IO/VTK.h>
#  include <CGAL/IO/output_to_vtu.h>
#  include <CGAL/boost/graph/IO/polygon_mesh_io.h>


DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  template <int dim>
  class GridGridIntersectionQuadratureGenerator
  {
    using K = CGAL::Exact_predicates_exact_constructions_kernel;

    // 3D
    using CGALPoint         = CGAL::Point_3<K>;
    using CGALTriangulation = CGAL::Triangulation_3<K>;

    // 2D
    using Traits = CGAL::Partition_traits_2<K>;

    using CGALPoint2           = CGAL::Point_2<K>;
    using CGALPolygon          = CGAL::Polygon_2<K>;
    using CGALPolygonWithHoles = CGAL::Polygon_with_holes_2<K>;
    using CGALSegment2         = CGAL::Segment_2<K>;
    using Triangulation2       = CGAL::Constrained_Delaunay_triangulation_2<K>;

  public:
    GridGridIntersectionQuadratureGenerator();

    GridGridIntersectionQuadratureGenerator(
      const Mapping<dim> &mapping_in,
      unsigned int        n_quadrature_points_1D_in,
      BooleanOperation    boolean_operation_in,
      bool                precompute_dg_faces_in = false);

    void
    reinit(const Mapping<dim> &mapping_in,
           unsigned int        n_quadrature_points_1D_in,
           BooleanOperation    boolean_operation_in,
           bool                precompute_dg_faces_in = false);

    void
    set_n_quadrature_points_1D(unsigned int n_quadrature_points_1D_in);

    void
    clear();

    template <typename TriangulationType>
    void
    setup_domain_boundary(const TriangulationType &tria_fitted_in);

    template <typename TriangulationType>
    void
    reclassify(const TriangulationType &tria_unfitted_in);

    void
    generate(const typename Triangulation<dim>::cell_iterator &cell);

    void
    generate_dg_face(const typename Triangulation<dim>::cell_iterator &cell,
                     unsigned int face_index);

    NonMatching::ImmersedSurfaceQuadrature<dim>
    get_surface_quadrature() const;

    Quadrature<dim>
    get_inside_quadrature() const;

    Quadrature<dim>
    get_outside_quadrature() const;

    Quadrature<dim - 1>
    get_inside_quadrature_dg_face() const;

    Quadrature<dim - 1>
    get_outside_quadrature_dg_face() const;

    Quadrature<dim - 1>
    get_inside_quadrature_dg_face(
      unsigned int face_index) const; // precomputed dg faces

    NonMatching::LocationToLevelSet
    location_to_geometry(unsigned int cell_index) const;

    NonMatching::LocationToLevelSet
    location_to_geometry(
      const typename Triangulation<dim>::cell_iterator &cell) const;

    NonMatching::LocationToLevelSet
    location_to_geometry(const typename Triangulation<dim>::cell_iterator &cell,
                         unsigned int face_index) const;

    void
    output_fitted_mesh() const;

  private:
    CGAL::Bounded_side
    side_of_surface_mesh(const Point<dim> &point) const;


    const Mapping<dim> *mapping;
    unsigned int        n_quadrature_points_1D;
    BooleanOperation    boolean_operation;
    bool                precompute_dg_faces;

    CGALPolygon                   surface_mesh_2D;
    CGAL::Surface_mesh<CGALPoint> surface_mesh_3D;
    std::unique_ptr<
      CGAL::Side_of_triangle_mesh<CGAL::Surface_mesh<CGALPoint>, K>>
      side_of_surface_mesh_3D;

    Quadrature<dim>                             quad_cells;
    NonMatching::ImmersedSurfaceQuadrature<dim> quad_surface;
    Quadrature<dim - 1>                         quad_dg_face;
    std::vector<Quadrature<dim - 1>> quad_dg_face_vec; // precomputed dg faces

    std::vector<NonMatching::LocationToLevelSet> cell_locations;
    std::vector<NonMatching::LocationToLevelSet> face_locations;
  };



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
