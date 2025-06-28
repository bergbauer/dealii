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

#include <deal.II/base/config.h>

#include <deal.II/cgal/grid_grid_intersection_quadrature_generator.h>

DEAL_II_NAMESPACE_OPEN

#ifndef DOXYGEN
// Template implementations
namespace CGALWrappers
{
  template <int dim>
  GridGridIntersectionQuadratureGenerator<
    dim>::GridGridIntersectionQuadratureGenerator()
    : mapping(nullptr)
    , n_quadrature_points_1D(0)
    , boolean_operation(BooleanOperation::compute_intersection)
    , precompute_dg_faces(false)
  {
    Assert(
      dim == 2 || dim == 3,
      ExcMessage(
        "GridGridIntersectionQuadratureGenerator only supports 2D and 3D"));
  };

  template <int dim>
  GridGridIntersectionQuadratureGenerator<dim>::
    GridGridIntersectionQuadratureGenerator(
      const Mapping<dim> &mapping_in,
      unsigned int        n_quadrature_points_1D_in,
      BooleanOperation    boolean_operation_in,
      bool                precompute_dg_faces_in)
    : mapping(&mapping_in)
    , n_quadrature_points_1D(n_quadrature_points_1D_in)
    , boolean_operation(boolean_operation_in)
    , precompute_dg_faces(precompute_dg_faces_in)
  {
    Assert(
      dim == 2 || dim == 3,
      ExcMessage(
        "GridGridIntersectionQuadratureGenerator only supports 2D and 3D"));
    Assert(boolean_operation_in == BooleanOperation::compute_intersection ||
             boolean_operation_in == BooleanOperation::compute_difference,
           ExcMessage("Union and corerefinement not implemented"));
  }

  template <int dim>
  void
  GridGridIntersectionQuadratureGenerator<dim>::reinit(
    const Mapping<dim> &mapping_in,
    unsigned int        n_quadrature_points_1D_in,
    BooleanOperation    boolean_operation_in,
    bool                precompute_dg_faces_in)
  {
    mapping                = &mapping_in;
    n_quadrature_points_1D = n_quadrature_points_1D_in;
    boolean_operation      = boolean_operation_in;
    precompute_dg_faces    = precompute_dg_faces_in;
    Assert(boolean_operation_in == BooleanOperation::compute_intersection ||
             boolean_operation_in == BooleanOperation::compute_difference,
           ExcMessage("Union and corerefinement not implemented"));
  }

  template <int dim>
  void
  GridGridIntersectionQuadratureGenerator<dim>::set_n_quadrature_points_1D(
    unsigned int n_quadrature_points_1D_in)
  {
    n_quadrature_points_1D = n_quadrature_points_1D_in;
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<2>::clear()
  {
    quad_cells   = Quadrature<2>();
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<2>();
    cell_locations.clear();
    surface_mesh_3D.clear();
    surface_mesh_2D.clear();
    quad_dg_face_vec.clear();
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::clear()
  {
    quad_cells   = Quadrature<3>();
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<3>();
    cell_locations.clear();
    surface_mesh_3D.clear();
    surface_mesh_2D.clear();
    quad_dg_face_vec.clear();
  }

  template <>
  template <typename TriangulationType>
  void
  GridGridIntersectionQuadratureGenerator<2>::setup_domain_boundary(
    const TriangulationType &tria_fitted_in)
  {
    surface_mesh_2D.clear();
    dealii_tria_to_cgal_polygon(tria_fitted_in, surface_mesh_2D);

    Assert(surface_mesh_2D.is_simple(), ExcMessage("Polygon not simple"));
    Assert(surface_mesh_2D.is_counterclockwise_oriented(),
           ExcMessage("Polygon not oriented"));
  }

  template <>
  template <typename TriangulationType>
  void
  GridGridIntersectionQuadratureGenerator<3>::setup_domain_boundary(
    const TriangulationType &tria_fitted_in)
  {
    surface_mesh_3D.clear();
    dealii_tria_to_cgal_surface_mesh<CGALPoint>(tria_fitted_in,
                                                surface_mesh_3D);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_mesh_3D);

    side_of_surface_mesh_3D = std::make_unique<
      CGAL::Side_of_triangle_mesh<CGAL::Surface_mesh<CGALPoint>, K>>(
      surface_mesh_3D);

    Assert(surface_mesh_3D.is_valid(),
           ExcMessage("The CGAL mesh is not valid"));
    Assert(CGAL::is_closed(surface_mesh_3D),
           ExcMessage("The CGAL mesh is not closed"));
    Assert(CGAL::Polygon_mesh_processing::is_outward_oriented(surface_mesh_3D),
           ExcMessage(
             "The normal vectors of the CGAL mesh are not oriented outwards"));
  }

  template <>
  CGAL::Bounded_side
  GridGridIntersectionQuadratureGenerator<2>::side_of_surface_mesh(
    const Point<2> &point) const
  {
    auto cgal_point = dealii_point_to_cgal_point<CGALPoint2, 2>(point);

    return CGAL::bounded_side_2(surface_mesh_2D.begin(),
                                surface_mesh_2D.end(),
                                cgal_point);
  }

  template <>
  CGAL::Bounded_side
  GridGridIntersectionQuadratureGenerator<3>::side_of_surface_mesh(
    const Point<3> &point) const
  {
    auto cgal_point = dealii_point_to_cgal_point<CGALPoint, 3>(point);
    return (*side_of_surface_mesh_3D)(cgal_point);
  }

  // The classification inside is only valid if no vertex is on the
  // boundary. Technically if one vertex is on the boundary the cell could still
  // be completely inside but a conservatice approach is chosen.
  // Since we only check for edges we guess it might intersect.
  //-> in this case we will generate a volume integral over whole cell!
  template <int dim>
  template <typename TriangulationType>
  void
  GridGridIntersectionQuadratureGenerator<dim>::reclassify(
    const TriangulationType &tria_unfitted)
  {
    quad_dg_face_vec.clear(); // move somewhere else!!
    cell_locations.assign(tria_unfitted.n_active_cells(),
                          NonMatching::LocationToLevelSet::unassigned);
    face_locations.assign(tria_unfitted.n_raw_faces(),
                          NonMatching::LocationToLevelSet::unassigned);

    CGAL::Bounded_side inside_domain;
    if (boolean_operation == BooleanOperation::compute_intersection)
      {
        inside_domain = CGAL::ON_BOUNDED_SIDE;
      }
    else if (boolean_operation == BooleanOperation::compute_difference)
      {
        inside_domain = CGAL::ON_UNBOUNDED_SIDE;
      }
    else
      {
        DEAL_II_ASSERT_UNREACHABLE();
      }

    const auto &all_vertices = tria_unfitted.get_vertices();
    const auto &used_flags   = tria_unfitted.get_used_vertices();
    std::vector<CGAL::Bounded_side> vertex_locations(all_vertices.size(),
                                                     CGAL::ON_BOUNDARY);
    for (size_t i = 0; i < all_vertices.size(); i++)
      {
        if (used_flags[i])
          {
            vertex_locations[i] = side_of_surface_mesh(all_vertices[i]);
          }
      }

    // now find out if inside or not
    for (const auto &cell : tria_unfitted.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        unsigned int inside_count = 0;
        unsigned int total_count  = 0;
        for (size_t f = 0; f < cell->n_faces(); f++)
          {
            unsigned int inside_count_face   = 0;
            unsigned int boundary_count_face = 0;
            total_count += cell->face(f)->n_vertices() + 1;

            // test center of face
            auto result = side_of_surface_mesh(cell->face(f)->center());

            bool center_is_inside = (result == inside_domain);

            inside_count += center_is_inside;
            inside_count_face += center_is_inside;
            boundary_count_face += (result == CGAL::ON_BOUNDARY);

            // test vertices of face
            for (size_t v = 0; v < cell->face(f)->n_vertices(); v++)
              {
                bool v_is_inside =
                  (vertex_locations[cell->face(f)->vertex_index(v)] ==
                   inside_domain);
                inside_count_face += v_is_inside;
                inside_count += v_is_inside;
                boundary_count_face +=
                  (vertex_locations[cell->face(f)->vertex_index(v)] ==
                   CGAL::ON_BOUNDARY);
              }

            if (inside_count_face == 0 &&
                boundary_count_face != cell->n_faces() + 1)
              {
                face_locations[cell->face(f)->index()] =
                  NonMatching::LocationToLevelSet::outside;
              }
            else if (inside_count_face == cell->n_faces() + 1)
              {
                face_locations[cell->face(f)->index()] =
                  NonMatching::LocationToLevelSet::inside;
              }
            else
              {
                face_locations[cell->face(f)->index()] =
                  NonMatching::LocationToLevelSet::intersected;
              }
          }

        if (inside_count == 0) // case 1: all vertices outside or on boundary:
                               // not considered (outside)
          {
            cell_locations[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::outside;
            // Assert(!CGAL::do_intersect(surface_mesh_2D, polygon_cell),
            // ExcMessage("cell classified as outside although intersected"));
          }
        else if (inside_count == total_count) // case 2: all vertices inside:
                                              // considered (inside)
          {
            cell_locations[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::inside;
          }
        else // case 3: at least one vertex inside and at least one on boundary
             // or outside
          {
            cell_locations[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::intersected;
          }
        // Note: construction of case 1 and 3 make sure that if two vertices are
        // on the boundary the cell
        // inside is considered cut and takes account for the face integral. The
        // cell outside also has vertices on the boundary but is ignored
        // because then the boundary integral would be performed twice
      }
    // maybe not appropriate here, should do a destructor
    // but destruct shared pointer before surface_mesh_3D!!
    // without this line-> error in destructor (i think because of that)
    side_of_surface_mesh_3D.reset();
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<2>::generate(
    const typename Triangulation<2>::cell_iterator &cell)
  {
    // generate polygon for current cell
    CGALPolygon polygon_cell;
    dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

    // perform boolean operation on cell and fitted mesh
    // result is a polygon with holes
    std::vector<CGALPolygonWithHoles> polygon_out_vec;
    compute_boolean_operation(polygon_cell,
                              surface_mesh_2D,
                              boolean_operation,
                              polygon_out_vec);

    // quadrature area in a cell could be split into two polygons
    // occurrence is not expected for smooth boundaries
    // -> outer for loop here for eventual extension
    Assert(polygon_out_vec.size() == 1,
           ExcMessage(
             "Not a single polygon with holes, disconnected domain!!"));

    std::vector<std::array<dealii::Point<2>, 3>> vec_of_simplices;
    for (const auto &polygon_out : polygon_out_vec)
      {
        Assert(polygon_out.outer_boundary().is_simple(),
               ExcMessage("The Polygon outer boundary is not simple"));
        // quadrature area in a cell cannot be a polygon with holes
        Assert(!polygon_out.has_holes(), ExcMessage("The Polygon has holes"));

        // partition polygon into convex polygons
        // these can be meshed as convex hull
        // Note: could use CGAL::approx_convex_partition_2
        Traits                       partition_traits;
        std::list<Traits::Polygon_2> convex_polygons;
        CGAL::optimal_convex_partition_2(
          polygon_out.outer_boundary().vertices_begin(),
          polygon_out.outer_boundary().vertices_end(),
          std::back_inserter(convex_polygons),
          partition_traits);

        Triangulation2 tria;
        for (const auto &convex_poly : convex_polygons)
          {
            tria.insert(convex_poly.vertices_begin(),
                        convex_poly.vertices_end());

            tria.insert_constraint(convex_poly.vertices_begin(),
                                   convex_poly.vertices_end(),
                                   true);

            // Extract simplices and construct quadratures
            for (const auto &face : tria.finite_face_handles())
              {
                std::array<dealii::Point<2>, 3> simplex;
                std::array<dealii::Point<2>, 3> unit_simplex;
                for (unsigned int i = 0; i < 3; ++i)
                  {
                    simplex[i] =
                      cgal_point_to_dealii_point<2>(face->vertex(i)->point());
                  }
                mapping->transform_points_real_to_unit_cell(cell,
                                                            simplex,
                                                            unit_simplex);
                vec_of_simplices.push_back(unit_simplex);
              }

            tria.clear();
          }
      }
    quad_cells = QGaussSimplex<2>(n_quadrature_points_1D)
                   .mapped_quadrature(vec_of_simplices);

    // surface quadrature
    std::vector<std::vector<Point<1>>> dg_quadrature_points;
    std::vector<std::vector<double>>   dg_quadrature_weights;
    if (precompute_dg_faces)
      {
        dg_quadrature_points.resize(cell->n_faces());
        dg_quadrature_weights.resize(cell->n_faces());
      }

    std::vector<Point<2>>     quadrature_points;
    std::vector<double>       quadrature_weights;
    std::vector<Tensor<1, 2>> normals;
    for (const auto &polygon_out : polygon_out_vec)
      {
        for (const auto &edge_cut : polygon_out.outer_boundary().edges())
          {
            unsigned int dg_face_index = cell->n_faces() + 1;
            auto         p_cut_1       = edge_cut.source();
            auto         p_cut_2       = edge_cut.target();
            for (const unsigned int i : cell->face_indices())
              {
                const typename Triangulation<2, 2>::face_iterator &face =
                  cell->face(i);
                if (face->at_boundary() ||
                    location_to_geometry(cell->neighbor(i)) ==
                      NonMatching::LocationToLevelSet::outside)
                  {
                    continue;
                  }

                auto p_uncut_1 =
                  dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(0));
                auto p_uncut_2 =
                  dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(1));

                if (CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_1) &&
                    CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_2))
                  {
                    dg_face_index = i;
                    break;
                  }
              }

            if (dg_face_index == cell->n_faces() + 1)
              {
                std::array<dealii::Point<2>, 2> unit_segment;
                mapping->transform_points_real_to_unit_cell(
                  cell,
                  {{cgal_point_to_dealii_point<2>(p_cut_1),
                    cgal_point_to_dealii_point<2>(p_cut_2)}},
                  unit_segment);

                auto quadrature =
                  QGaussSimplex<1>(n_quadrature_points_1D)
                    .compute_affine_transformation(unit_segment);
                auto points  = quadrature.get_points();
                auto weights = quadrature.get_weights();

                // compute normals
                Tensor<1, 2> vector = unit_segment[1] - unit_segment[0];
                Tensor<1, 2> normal;
                if (boolean_operation == BooleanOperation::compute_intersection)
                  {
                    normal[0] = vector[1];
                    normal[1] = -vector[0];
                  }
                else if (boolean_operation ==
                         BooleanOperation::compute_difference)
                  {
                    normal[0] = -vector[1];
                    normal[1] = vector[0];
                  }
                else
                  {
                    DEAL_II_ASSERT_UNREACHABLE();
                  }

                normal /= normal.norm();


                quadrature_points.insert(quadrature_points.end(),
                                         points.begin(),
                                         points.end());
                quadrature_weights.insert(quadrature_weights.end(),
                                          weights.begin(),
                                          weights.end());
                normals.insert(normals.end(), quadrature.size(), normal);
              }
            else if (precompute_dg_faces) // change to true if want to use
                                          // precomputed dg
              {
                auto p_unit_1 =
                  mapping->project_real_point_to_unit_point_on_face(
                    cell,
                    dg_face_index,
                    cgal_point_to_dealii_point<2>(p_cut_1));

                auto p_unit_2 =
                  mapping->project_real_point_to_unit_point_on_face(
                    cell,
                    dg_face_index,
                    cgal_point_to_dealii_point<2>(p_cut_2));

                Quadrature<1> quadrature =
                  QGaussSimplex<1>(n_quadrature_points_1D)
                    .compute_affine_transformation({{p_unit_1, p_unit_2}});

                auto points  = quadrature.get_points();
                auto weights = quadrature.get_weights();
                dg_quadrature_points[dg_face_index].insert(
                  dg_quadrature_points[dg_face_index].end(),
                  points.begin(),
                  points.end());
                dg_quadrature_weights[dg_face_index].insert(
                  dg_quadrature_weights[dg_face_index].end(),
                  weights.begin(),
                  weights.end());
              }
          }
      }
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<2>(quadrature_points,
                                                             quadrature_weights,
                                                             normals);

    if (precompute_dg_faces)
      {
        for (const unsigned int i : cell->face_indices())
          {
            quad_dg_face_vec[i] =
              Quadrature<1>(dg_quadrature_points[i], dg_quadrature_weights[i]);
          }
      }
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::generate(
    const typename Triangulation<3>::cell_iterator &cell)
  {
    CGAL::Surface_mesh<CGALPoint> surface_cell;
    dealii_cell_to_cgal_surface_mesh(cell, *mapping, surface_cell);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_cell);

    CGAL::Surface_mesh<CGALPoint> out_surface;
    compute_boolean_operation(surface_cell,
                              surface_mesh_3D,
                              boolean_operation,
                              out_surface);

    // Fill triangulation with vertices from surface mesh
    CGALTriangulation tria;
    tria.insert(out_surface.points().begin(), out_surface.points().end());

    CGAL::Side_of_triangle_mesh<CGAL::Surface_mesh<CGALPoint>, K> inside_test(
      out_surface);

    // Extract simplices and construct quadratures
    std::vector<std::array<dealii::Point<3>, 4>> vec_of_simplices;
    for (const auto &face : tria.finite_cell_handles())
      {
        std::array<CGALPoint, 4>        simplex_cgal;
        std::array<dealii::Point<3>, 4> simplex;
        std::array<dealii::Point<3>, 4> unit_simplex;
        for (unsigned int i = 0; i < 4; ++i)
          {
            simplex_cgal[i] = face->vertex(i)->point();
            simplex[i] =
              cgal_point_to_dealii_point<3>(face->vertex(i)->point());
          }

        auto centroid = CGAL::centroid(simplex_cgal[0],
                                       simplex_cgal[1],
                                       simplex_cgal[2],
                                       simplex_cgal[3]);
        // dont consider convex hull triangles that are not inside
        if (inside_test(centroid) != CGAL::ON_BOUNDED_SIDE)
          continue;

        mapping->transform_points_real_to_unit_cell(cell,
                                                    simplex,
                                                    unit_simplex);
        vec_of_simplices.push_back(unit_simplex);
      }
    quad_cells = QGaussSimplex<3>(n_quadrature_points_1D)
                   .mapped_quadrature(vec_of_simplices);

    // surface quadrature
    std::vector<Point<3>>     quadrature_points;
    std::vector<double>       quadrature_weights;
    std::vector<Tensor<1, 3>> normals;
    double ref_area = std::pow(cell->minimum_vertex_distance(), 2) * 0.0000001;
    for (const auto &out_surface_face : out_surface.faces())
      {
        if (CGAL::abs(CGAL::Polygon_mesh_processing::face_area(out_surface_face,
                                                               out_surface)) <
            ref_area)
          {
            continue;
          }

        unsigned int             dg_face_index = cell->n_faces() + 1;
        std::array<CGALPoint, 3> simplex;
        int                      i = 0;
        for (const auto &vertex :
             CGAL::vertices_around_face(out_surface.halfedge(out_surface_face),
                                        out_surface))
          {
            simplex[i] = out_surface.point(vertex);
            i += 1;
          }

        for (const unsigned int i_dealii_face : cell->face_indices())
          {
            const typename Triangulation<3, 3>::face_iterator &face =
              cell->face(i_dealii_face);
            if (face->at_boundary() ||
                location_to_geometry(cell->neighbor(i_dealii_face)) ==
                  NonMatching::LocationToLevelSet::outside)
              {
                continue;
              }

            int count = 0;
            for (unsigned int i_dealii_vertex = 0;
                 i_dealii_vertex < face->n_vertices();
                 ++i_dealii_vertex)
              {
                auto cgal_point = dealii_point_to_cgal_point<CGALPoint, 3>(
                  face->vertex(i_dealii_vertex));
                count += CGAL::coplanar(simplex[0],
                                        simplex[1],
                                        simplex[2],
                                        cgal_point);
              }

            if (count >= 3)
              {
                dg_face_index = i_dealii_face;
                break;
              }
          }

        if (dg_face_index == cell->n_faces() + 1)
          {
            std::array<Point<3>, 3> unit_simplex;

            // compute quadrature and fill vectors
            mapping->transform_points_real_to_unit_cell(
              cell,
              {{cgal_point_to_dealii_point<3>(simplex[0]),
                cgal_point_to_dealii_point<3>(simplex[1]),
                cgal_point_to_dealii_point<3>(simplex[2])}},
              unit_simplex);
            auto quadrature = QGaussSimplex<2>(n_quadrature_points_1D)
                                .compute_affine_transformation(unit_simplex);
            auto points  = quadrature.get_points();
            auto weights = quadrature.get_weights();
            quadrature_points.insert(quadrature_points.end(),
                                     points.begin(),
                                     points.end());
            quadrature_weights.insert(quadrature_weights.end(),
                                      weights.begin(),
                                      weights.end());

            const Tensor<1, 3> v1 = cgal_point_to_dealii_point<3>(simplex[2]) -
                                    cgal_point_to_dealii_point<3>(simplex[1]);
            const Tensor<1, 3> v2 = cgal_point_to_dealii_point<3>(simplex[0]) -
                                    cgal_point_to_dealii_point<3>(simplex[1]);
            Tensor<1, 3> normal = cross_product_3d(v1, v2);
            normal /= normal.norm();
            normals.insert(normals.end(), quadrature.size(), normal);
          }
        else if (precompute_dg_faces) // change to true if want to use
                                      // precomputed dg
          {
          }
      }
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<3>(quadrature_points,
                                                             quadrature_weights,
                                                             normals);
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<2>::generate_dg_face(
    const typename Triangulation<2>::cell_iterator &cell,
    unsigned int                                    face_index)
  {
    const typename Triangulation<2, 2>::face_iterator &face =
      cell->face(face_index);
    if (face->at_boundary() ||
        location_to_geometry(cell->neighbor(face_index)) ==
          NonMatching::LocationToLevelSet::outside)
      {
        quad_dg_face = Quadrature<1>();
        return;
      }

    CGALPolygon polygon_cell;
    dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

    std::vector<CGALPolygonWithHoles> polygon_out_vec;

    compute_boolean_operation(polygon_cell,
                              surface_mesh_2D,
                              boolean_operation,
                              polygon_out_vec);

    std::vector<Point<1>> quadrature_points;
    std::vector<double>   quadrature_weights;
    for (const auto &polygon_out : polygon_out_vec)
      {
        for (const auto &edge_cut : polygon_out.outer_boundary().edges())
          {
            bool dg_face = false;
            auto p_cut_1 = edge_cut.source();
            auto p_cut_2 = edge_cut.target();

            auto p_uncut_1 =
              dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(0));
            auto p_uncut_2 =
              dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(1));
            if (CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_1) &&
                CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_2))
              {
                dg_face = true;
              }

            if (dg_face)
              {
                auto p_unit_1 =
                  mapping->project_real_point_to_unit_point_on_face(
                    cell, face_index, cgal_point_to_dealii_point<2>(p_cut_1));

                auto p_unit_2 =
                  mapping->project_real_point_to_unit_point_on_face(
                    cell, face_index, cgal_point_to_dealii_point<2>(p_cut_2));

                Quadrature<1> quadrature =
                  QGaussSimplex<1>(n_quadrature_points_1D)
                    .compute_affine_transformation({{p_unit_1, p_unit_2}});

                auto points  = quadrature.get_points();
                auto weights = quadrature.get_weights();
                quadrature_points.insert(quadrature_points.end(),
                                         points.begin(),
                                         points.end());
                quadrature_weights.insert(quadrature_weights.end(),
                                          weights.begin(),
                                          weights.end());
              }
          }
      }
    quad_dg_face = Quadrature<1>(quadrature_points, quadrature_weights);
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::generate_dg_face(
    const typename Triangulation<3>::cell_iterator &cell,
    unsigned int                                    face_index)
  {
    const typename Triangulation<3, 3>::face_iterator &face =
      cell->face(face_index);
    if (face->at_boundary() ||
        location_to_geometry(cell->neighbor(face_index)) ==
          NonMatching::LocationToLevelSet::outside)
      {
        quad_dg_face = Quadrature<2>();
        return;
      }

    CGAL::Surface_mesh<CGALPoint> surface_cell;
    dealii_cell_to_cgal_surface_mesh(cell, *mapping, surface_cell);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_cell);

    CGAL::Surface_mesh<CGALPoint> out_surface;
    compute_boolean_operation(surface_cell,
                              surface_mesh_3D,
                              boolean_operation,
                              out_surface);

    std::vector<CGALPoint> dealii_vertices;
    for (unsigned int i_dealii_vertex = 0; i_dealii_vertex < face->n_vertices();
         ++i_dealii_vertex)
      {
        dealii_vertices.push_back(dealii_point_to_cgal_point<CGALPoint, 3>(
          face->vertex(i_dealii_vertex)));
      }

    std::vector<Point<2>> quadrature_points;
    std::vector<double>   quadrature_weights;
    for (const auto &face_surface : out_surface.faces())
      {
        int                      i_cgal_vertex = 0;
        std::array<CGALPoint, 3> cgal_vertices;
        for (const auto &vertex :
             CGAL::vertices_around_face(out_surface.halfedge(face_surface),
                                        out_surface))
          {
            cgal_vertices[i_cgal_vertex] = out_surface.point(vertex);
            i_cgal_vertex += 1;
          }

        int count = 0;
        for (auto &vertex : dealii_vertices)
          {
            count += CGAL::coplanar(cgal_vertices[0],
                                    cgal_vertices[1],
                                    cgal_vertices[2],
                                    vertex);
          }

        if (count >= 3)
          {
            std::array<Point<2>, 3> simplex;
            int                     i = 0;
            for (const auto &vertex :
                 CGAL::vertices_around_face(out_surface.halfedge(face_surface),
                                            out_surface))
              {
                auto dealii_point =
                  cgal_point_to_dealii_point<3>(out_surface.point(vertex));
                simplex[i] = mapping->project_real_point_to_unit_point_on_face(
                  cell, face_index, dealii_point);
                i += 1;
              }

            auto quadrature = QGaussSimplex<2>(n_quadrature_points_1D)
                                .compute_affine_transformation(simplex);

            auto points  = quadrature.get_points();
            auto weights = quadrature.get_weights();
            quadrature_points.insert(quadrature_points.end(),
                                     points.begin(),
                                     points.end());
            quadrature_weights.insert(quadrature_weights.end(),
                                      weights.begin(),
                                      weights.end());
          }
      }
    quad_dg_face = Quadrature<2>(quadrature_points, quadrature_weights);
  }

  template <int dim>
  NonMatching::ImmersedSurfaceQuadrature<dim>
  GridGridIntersectionQuadratureGenerator<dim>::get_surface_quadrature() const
  {
    return quad_surface;
  }

  template <int dim>
  Quadrature<dim>
  GridGridIntersectionQuadratureGenerator<dim>::get_inside_quadrature() const
  {
    return quad_cells;
  }

  template <int dim>
  Quadrature<dim>
  GridGridIntersectionQuadratureGenerator<dim>::get_outside_quadrature() const
  {
    AssertThrow(false, ExcNotImplemented());
    return quad_cells;
  }

  template <int dim>
  Quadrature<dim - 1>
  GridGridIntersectionQuadratureGenerator<dim>::get_inside_quadrature_dg_face()
    const
  {
    return quad_dg_face;
  }

  template <int dim>
  Quadrature<dim - 1>
  GridGridIntersectionQuadratureGenerator<dim>::get_outside_quadrature_dg_face()
  const
  {
    AssertThrow(false, ExcNotImplemented());
    return quad_dg_face;
  }

  template <int dim>
  Quadrature<dim - 1>
  GridGridIntersectionQuadratureGenerator<dim>::get_inside_quadrature_dg_face(
    unsigned int face_index) const
  {
    return quad_dg_face_vec.at(face_index);
  }

  template <int dim>
  NonMatching::LocationToLevelSet
  GridGridIntersectionQuadratureGenerator<dim>::location_to_geometry(
    unsigned int cell_index) const
  {
    return cell_locations.at(cell_index);
  }

  template <int dim>
  NonMatching::LocationToLevelSet
  GridGridIntersectionQuadratureGenerator<dim>::location_to_geometry(
    const typename Triangulation<dim>::cell_iterator &cell) const
  {
    return cell_locations.at(cell->active_cell_index());
  }

  template <int dim>
  NonMatching::LocationToLevelSet
  GridGridIntersectionQuadratureGenerator<dim>::location_to_geometry(
    const typename Triangulation<dim>::cell_iterator &cell,
    unsigned int                                      face_index) const
  {
    return face_locations.at(cell->face(face_index)->index());
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<2>::output_fitted_mesh() const
  {
    std::string   filename = "fitted_polygon.vtu";
    std::ofstream file(filename);
    if (!file)
      {
        std::cerr << "Error opening file for writing: " << filename
                  << std::endl;
        return;
      }

    const std::size_t n = surface_mesh_2D.size();

    file << R"(<?xml version="1.0"?>)"
         << "\n";
    file
      << R"(<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">)"
      << "\n";
    file << R"(  <UnstructuredGrid>)"
         << "\n";
    file << R"(    <Piece NumberOfPoints=")" << n << R"(" NumberOfCells="1">)"
         << "\n";

    // Points section
    file << R"(      <Points>)"
         << "\n";
    file
      << R"(        <DataArray type="Float64" NumberOfComponents="3" format="ascii">)"
      << "\n";

    for (const auto &p : surface_mesh_2D.container())
      {
        file << p.x() << " " << p.y() << " 0 ";
      }
    file << "\n";

    file << R"(        </DataArray>)"
         << "\n";
    file << R"(      </Points>)"
         << "\n";

    // Cells section
    // Connectivity: indices of vertices in order
    file << R"(      <Cells>)"
         << "\n";

    // Connectivity
    file
      << R"(        <DataArray type="Int32" Name="connectivity" format="ascii">)";
    for (std::size_t i = 0; i < n; ++i)
      {
        file << i << " ";
      }
    file << R"(</DataArray>)"
         << "\n";

    // Offsets: cumulative count of vertices after each cell
    // Here only one cell with n vertices
    file << R"(        <DataArray type="Int32" Name="offsets" format="ascii">)";
    file << n << R"(</DataArray>)"
         << "\n";

    // Types: VTK cell type for polygon is 7
    // (See https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
    file
      << R"(        <DataArray type="UInt8" Name="types" format="ascii">7</DataArray>)"
      << "\n";

    file << R"(      </Cells>)"
         << "\n";

    file << R"(    </Piece>)"
         << "\n";
    file << R"(  </UnstructuredGrid>)"
         << "\n";
    file << R"(</VTKFile>)"
         << "\n";

    file.close();
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::output_fitted_mesh() const
  {
    CGAL::IO::write_polygon_mesh("surface_mesh_3D.stl", surface_mesh_3D);
  }



// Explicit instantiations.
//
// We don't build the instantiations.inst file if deal.II isn't
// configured with CGAL, but doxygen doesn't know that and tries to
// find that file anyway for parsing -- which then of course it fails
// on. So exclude the following from doxygen consideration.
#  ifndef DOXYGEN
#    include "cgal/grid_grid_intersection_quadrature_generator.inst"
#  endif

} // namespace  CGALWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
