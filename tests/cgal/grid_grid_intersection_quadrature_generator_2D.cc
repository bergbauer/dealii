
#include <deal.II/base/config.h>

#include <deal.II/cgal/grid_grid_intersection_quadrature_generator.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/non_matching/mapping_info.h>

#include "../tests.h"


using namespace CGALWrappers;

using K           = CGAL::Exact_predicates_exact_constructions_kernel;
using CGALPoint2  = CGAL::Point_2<K>;
using CGALPolygon = CGAL::Polygon_2<K>;

void
test(unsigned int refinement_domain, unsigned int refinement_boundary)
{
  Triangulation<2, 2> tria_domain;
  Triangulation<2, 2> tria_boundary;
  CGALPolygon         poly;

  GridGridIntersectionQuadratureGenerator<2> ggi_quadrature_generator;
  MappingQ<2>                                mapping(1);
  unsigned int                               quadrature_degree = 2;

  const FE_Q<2, 2> finite_element(1); // or put FE::nothing

  std::vector<std::pair<std::string, std::string>> names_and_args;
  std::array<BooleanOperation, 2>                  boolean_operations = {
    {BooleanOperation::compute_difference,
                      BooleanOperation::compute_intersection}};

  // only for meshes that have no holes
  // extension to CGAL::Polygon_with_holes possible
  names_and_args = {
    {"hyper_cube",
     "0.0 : 1.0 : false"}, // extreme case grid borders match complete
    {"hyper_cube", "-0.1 : 0.9 : false"}, // concave area for difference case
    {"hyper_rectangle",
     "0.0, -0.1 : 1.0 , 0.5 : false"}, // difficult boundary case
    {"hyper_rectangle",
     "-0.1, -0.1 : 0.5 , 1.1 : false"}, // only partially in domain
    {"hyper_rectangle",
     "-0.1, 0.5 : 0.0 , 1.1 : false"}, // only partially in domain
    {"simplex",
     "0.0, 0.0 ; 1.0 , 0.0 ; 0.0, 1.0"}, // simplex cutting through diagonal
    {"simplex", "-0.1, -0.1 ; 0.9 , -0.1 ; -0.1, 0.9"},
    {"simplex", "-0.5, -0.5 ; 0.5 , 0.5 ; -0.5, 1.5"},
    {"hyper_ball_balanced", "0.0,0.0 : 1.0"},    // centered circle
    {"hyper_ball_balanced", "-0.1 ,1.0 : 0.5"}}; // only partially in domain
  // Notes: cell size of domain must be a order smaller than the size of the
  // domain mesh because of classify
  //  this is in FEM always the case (but keep in mind that cells that contain a
  //  whole boundary mesh  or only have vertices on the edge of it will be
  //  classified as not cut -> change refinement to 1 to see effect)

  // The cell to be cut
  GridGenerator::reference_cell(tria_domain, ReferenceCells::Quadrilateral);
  tria_domain.refine_global(refinement_domain);

  for (const auto &info_pair : names_and_args)
    {
      auto name = info_pair.first;
      auto args = info_pair.second;
      deallog << "name: " << name << std::endl;
      GridGenerator::generate_from_name_and_arguments(tria_boundary,
                                                      name,
                                                      args);
      tria_boundary.refine_global(refinement_boundary);

      for (const auto &boolean_operation : boolean_operations)
        {
          double measure = 0;
          double surface = 0;
          ggi_quadrature_generator.reinit(mapping,
                                          quadrature_degree,
                                          boolean_operation);
          ggi_quadrature_generator.setup_domain_boundary(tria_boundary);
          ggi_quadrature_generator.reclassify(tria_domain);

          std::vector<Quadrature<2>> vector_quadratures;
          std::vector<NonMatching::ImmersedSurfaceQuadrature<2>>
            vector_quadratures_surface;
          std::vector<typename Triangulation<2, 2>::cell_iterator>
            vector_accessors;
          for (const auto &cell : tria_domain.active_cell_iterators())
            {
              auto classification =
                ggi_quadrature_generator.location_to_geometry(cell);
              if (classification ==
                  NonMatching::LocationToLevelSet::intersected)
                {
                  ggi_quadrature_generator.generate(cell);
                  vector_quadratures.push_back(
                    ggi_quadrature_generator.get_inside_quadrature());
                  vector_quadratures_surface.push_back(
                    ggi_quadrature_generator.get_surface_quadrature());
                  vector_accessors.push_back(cell);
                }
              else if (classification ==
                       NonMatching::LocationToLevelSet::inside)
                {
                  measure += cell->measure();
                  for (const auto &face : cell->face_iterators())
                    {
                      if (face->at_boundary())
                        {
                          surface += face->measure();
                        }
                    }
                }
            }
          // For volume integration
          NonMatching::MappingInfo mapping_info(
            mapping,
            UpdateFlags::update_quadrature_points |
              UpdateFlags::update_JxW_values);

          FEPointEvaluation<1, 2, 2, double> point_evaluation(mapping_info,
                                                              finite_element);
          mapping_info.reinit_cells(vector_accessors, vector_quadratures);

          // For area integration
          NonMatching::MappingInfo mapping_info_surface(
            mapping,
            UpdateFlags::update_quadrature_points |
              UpdateFlags::update_JxW_values);

          FEPointEvaluation<1, 2, 2, double> point_evaluation_surface(
            mapping_info_surface, finite_element);
          mapping_info_surface.reinit_surface(vector_accessors,
                                              vector_quadratures_surface);

          for (size_t i = 0; i < vector_accessors.size(); i++)
            {
              point_evaluation.reinit(i);
              for (unsigned int q : point_evaluation.quadrature_point_indices())
                {
                  measure += point_evaluation.JxW(q);
                }
              point_evaluation_surface.reinit(i);
              for (unsigned int q :
                   point_evaluation_surface.quadrature_point_indices())
                {
                  surface += point_evaluation_surface.JxW(q);
                }
            }
          deallog << "Area :" << measure << " , surface :" << surface
                  << std::endl;

          ggi_quadrature_generator.clear();
          deallog << std::endl;
        }
      tria_boundary.clear();
    }
}


int
main()
{
  initlog();
  // test(0, 0); //limit case for classification function: domain scale in size
  // of cell dimension
  test(1, 1);
}
