#include <deal.II/base/function_restriction.h>
#include <deal.II/base/function_tools.h>

#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_element_access.h>

#include <deal.II/non_matching/cut_mesh_classifier.h>
#include <deal.II/non_matching/restriction.h>



DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  namespace internal
  {
    namespace CutMeshClassifierImplementation
    {
      template <int dim>
      CellAndFaceLocations<dim>::CellAndFaceLocations()
        : cell_location(LocationToLevelSet::UNASSIGNED)
      {
        face_locations.fill(LocationToLevelSet::UNASSIGNED);
      }



      LocationToLevelSet
      location_from_min_max_levelset_values(const double levelset_min,
                                            const double levelset_max)
      {
        if (levelset_max < 0)
          return LocationToLevelSet::INSIDE;
        if (0 < levelset_min)
          return LocationToLevelSet::OUTSIDE;

        return LocationToLevelSet::INTERSECTED;
      }



      /**
       * Checks the values of the level set function at the vertices of the
       * incoming function and returns the relation to the level set function
       * based on the values.
       */
      template <class VECTOR>
      LocationToLevelSet
      determine_position_based_on_values(const VECTOR &local_levelset_values)
      {
        const double levelset_min =
          *std::min_element(local_levelset_values.begin(),
                            local_levelset_values.end());

        const double levelset_max =
          *std::max_element(local_levelset_values.begin(),
                            local_levelset_values.end());

        return location_from_min_max_levelset_values(levelset_min,
                                                     levelset_max);
      }



      template <int dim>
      void
      create_lagrange_to_bernstein_cell(
        const FiniteElement<dim> &                 element,
        std::unique_ptr<LAPACKFullMatrix<double>> &lagrange_to_bernstein_cell)
      {
        const FE_Q<dim> *fe_q = dynamic_cast<const FE_Q<dim> *>(&element);
        Assert(fe_q != nullptr, ExcNotImplemented());

        const FE_Bernstein<dim>        fe_bernstein(fe_q->get_degree());
        const std::vector<Point<dim>> &points = fe_q->get_unit_support_points();

        Assert(points.size() == fe_bernstein.n_dofs_per_cell(),
               ExcInternalError());

        lagrange_to_bernstein_cell.reset(
          new LAPACKFullMatrix<double>(points.size(),
                                       fe_bernstein.n_dofs_per_cell()));

        for (unsigned int i = 0; i < points.size(); ++i)
          for (unsigned int j = 0; j < fe_bernstein.n_dofs_per_cell(); ++j)
            {
              (*lagrange_to_bernstein_cell)(i, j) =
                fe_bernstein.shape_value(j, points[i]);
            }

        lagrange_to_bernstein_cell->compute_lu_factorization();
      }



      template <int dim>
      void
      create_lagrange_to_bernstein_face(
        const FiniteElement<dim> &                 element,
        std::unique_ptr<LAPACKFullMatrix<double>> &lagrange_to_bernstein_face)
      {
        const FE_Q<dim> *fe_q = dynamic_cast<const FE_Q<dim> *>(&element);
        Assert(fe_q != nullptr, ExcNotImplemented());

        const FE_Bernstein<dim - 1>        fe_bernstein(fe_q->get_degree());
        const std::vector<Point<dim - 1>> &points =
          fe_q->get_unit_face_support_points();

        Assert(points.size() == fe_bernstein.n_dofs_per_cell(),
               ExcInternalError());

        lagrange_to_bernstein_face.reset(
          new LAPACKFullMatrix<double>(points.size(),
                                       fe_bernstein.n_dofs_per_cell()));

        for (unsigned int i = 0; i < points.size(); ++i)
          for (unsigned int j = 0; j < fe_bernstein.n_dofs_per_cell(); ++j)
            {
              (*lagrange_to_bernstein_face)(i, j) =
                fe_bernstein.shape_value(j, points[i]);
            }

        lagrange_to_bernstein_face->compute_lu_factorization();
      }



      void
      create_lagrange_to_bernstein_face(
        const FiniteElement<1> &,
        std::unique_ptr<LAPACKFullMatrix<double>> &lagrange_to_bernstein_face)
      {
        lagrange_to_bernstein_face.reset(new LAPACKFullMatrix<double>(1, 1));
        (*lagrange_to_bernstein_face)(0, 0) = 1;
        lagrange_to_bernstein_face->compute_lu_factorization();
      }



      /**
       * This class classifies an incoming cell/face into one of the categories
       * of LocationToLevelset based on the values of a discrete level set
       * function at the cell/face local degrees of freedom.
       *
       * This class is used by CutMeshClassifier when the level set function is
       * described as a vector associated with a DoFHandler.
       */
      template <int dim, class VECTOR>
      class DofBasedClassifier : public CellFaceClassifier<dim>
      {
      public:
        /**
         * Constructor, the incoming parameters describe a level set function
         * in a discrete space.
         */
        DofBasedClassifier(const DoFHandler<dim> &dof_handler,
                           const VECTOR &         level_set);


        /**
         * Returns how the incoming cell is located relative to the level set
         * function.
         */
        LocationToLevelSet
        determine_face_location_to_levelset(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const unsigned int face_index) override;
        /**
         * Returns how a face of the incoming cell is located relative to the
         * level set function.
         */
        LocationToLevelSet
        determine_cell_location_to_levelset(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const std::array<LocationToLevelSet,
                           GeometryInfo<dim>::faces_per_cell> &face_locations)
          override;

      private:
        /**
         * Checks the values of the level set function at the incoming dofs and
         * returns the relation to the level set function based on the values.
         */
        LocationToLevelSet
        determine_position_based_on_dofs(
          const std::vector<types::global_dof_index> &dof_indices) const;

        const SmartPointer<const DoFHandler<dim>> dof_handler;

        const SmartPointer<const VECTOR> level_set;

        std::unique_ptr<LAPACKFullMatrix<double>> lagrange_to_bernstein_cell;
        std::unique_ptr<LAPACKFullMatrix<double>> lagrange_to_bernstein_face;
      };



      template <int dim, class VECTOR>
      DofBasedClassifier<dim, VECTOR>::DofBasedClassifier(
        const DoFHandler<dim> &dof_handler,
        const VECTOR &         level_set)
        : dof_handler(&dof_handler)
        , level_set(&level_set)
      {}



      template <int dim, class VECTOR>
      LocationToLevelSet
      DofBasedClassifier<dim, VECTOR>::determine_face_location_to_levelset(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        const unsigned int                                       face_index)
      {
        if (!lagrange_to_bernstein_face)
          create_lagrange_to_bernstein_face(dof_handler->get_fe(),
                                            lagrange_to_bernstein_face);

        typename DoFHandler<dim>::active_cell_iterator cell_with_dofs(
          &dof_handler->get_triangulation(),
          cell->level(),
          cell->index(),
          dof_handler);

        // Get the dofs indices associated with the face.
        const unsigned int n_dofs_per_face =
          dof_handler->get_fe().n_dofs_per_face();
        std::vector<types::global_dof_index> dof_indices(n_dofs_per_face);
        cell_with_dofs->face(face_index)->get_dof_indices(dof_indices);

        Vector<double> local_levelset_values(dof_indices.size());

        for (unsigned int i = 0; i < dof_indices.size(); i++)
          local_levelset_values[i] =
            dealii::internal::ElementAccess<VECTOR>::get(*level_set,
                                                         dof_indices[i]);

        lagrange_to_bernstein_face->solve(local_levelset_values);

        return determine_position_based_on_values(local_levelset_values);
      }



      template <int dim, class VECTOR>
      LocationToLevelSet
      DofBasedClassifier<dim, VECTOR>::determine_cell_location_to_levelset(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        const std::array<LocationToLevelSet, GeometryInfo<dim>::faces_per_cell>
          &)
      {
        if (!lagrange_to_bernstein_cell)
          create_lagrange_to_bernstein_cell(dof_handler->get_fe(),
                                            lagrange_to_bernstein_cell);

        typename DoFHandler<dim>::active_cell_iterator cell_with_dofs(
          &dof_handler->get_triangulation(),
          cell->level(),
          cell->index(),
          dof_handler);

        // Get the dofs indices associated with the face.
        const unsigned int n_dofs_per_cell =
          dof_handler->get_fe().n_dofs_per_cell();
        std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
        cell_with_dofs->get_dof_indices(dof_indices);

        Vector<double> local_levelset_values(dof_indices.size());

        for (unsigned int i = 0; i < dof_indices.size(); i++)
          local_levelset_values[i] =
            dealii::internal::ElementAccess<VECTOR>::get(*level_set,
                                                         dof_indices[i]);

        lagrange_to_bernstein_cell->solve(local_levelset_values);

        return determine_position_based_on_values(local_levelset_values);
      }



      template <int dim, class VECTOR>
      LocationToLevelSet
      DofBasedClassifier<dim, VECTOR>::determine_position_based_on_dofs(
        const std::vector<types::global_dof_index> &dof_indices) const
      {
        std::vector<double> local_levelset_values(dof_indices.size());
        local_levelset_values.resize(dof_indices.size());

        for (unsigned int i = 0; i < dof_indices.size(); i++)
          local_levelset_values[i] =
            dealii::internal::ElementAccess<VECTOR>::get(*level_set,
                                                         dof_indices[i]);

        return determine_position_based_on_values(local_levelset_values);
      }



      /**
       * This class classifies an incoming cell/face into one of the categories
       * of LocationToLevelset based on the level set values at the vertices of
       * the cell/face.
       *
       * This class is used by CutMeshClassifier when the level set function is
       * described as a Function<dim>.
       */
      template <int dim>
      class FunctionBasedClassifier : public CellFaceClassifier<dim>
      {
      public:
        /**
         * Constructor, the incoming function describes the geometry.
         */
        FunctionBasedClassifier(const Function<dim> &level_set);

        /**
         * Returns how the incoming cell is located relative to the level set
         * function.
         */
        LocationToLevelSet
        determine_face_location_to_levelset(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const unsigned int face_index) override;

        /**
         * Returns how a face of the incoming cell is located relative to the
         * level set function.
         */
        LocationToLevelSet
        determine_cell_location_to_levelset(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const std::array<LocationToLevelSet,
                           GeometryInfo<dim>::faces_per_cell> &face_locations)
          override;

      private:
        /**
         * The level set function that describes the geometry, in reference
         * space.
         */
        std::shared_ptr<FunctionUnitToReal<dim>> level_set;
      };



      template <int dim>
      FunctionBasedClassifier<dim>::FunctionBasedClassifier(
        const Function<dim> &level_set)
        : level_set(new FunctionUnitToReal<dim>(level_set))
      {}



      template <>
      LocationToLevelSet
      FunctionBasedClassifier<1>::determine_face_location_to_levelset(
        const typename Triangulation<1>::active_cell_iterator &cell,
        const unsigned int                                     face_index)
      {
        level_set->set_active_cell(cell);

        const Point<1> vertex = GeometryInfo<1>::unit_cell_vertex(face_index);

        const double level_set_value = level_set->value(vertex);

        return location_from_min_max_levelset_values(level_set_value,
                                                     level_set_value);
      }



      template <int dim>
      inline void
      take_min_max_at_vertices(const Function<dim> &      function,
                               const BoundingBox<dim> &   box,
                               std::pair<double, double> &value_bounds)
      {
        for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
          {
            const double vertex_value = function.value(box.vertex(i));

            value_bounds.first  = std::min(value_bounds.first, vertex_value);
            value_bounds.second = std::max(value_bounds.second, vertex_value);
          }
      }



      template <int dim>
      LocationToLevelSet
      FunctionBasedClassifier<dim>::determine_face_location_to_levelset(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        const unsigned int                                       face_index)
      {
        level_set->set_active_cell(cell);
        const unsigned int restricted_coordinate =
          GeometryInfo<dim>::unit_normal_direction[face_index];

        const Point<dim> vertex0_on_face = GeometryInfo<dim>::unit_cell_vertex(
          GeometryInfo<dim>::face_to_cell_vertices(face_index, 0));
        const double restricted_coordinate_value =
          vertex0_on_face(restricted_coordinate);

        Functions::CoordinateRestriction<dim - 1> face_restriction(
          *level_set, restricted_coordinate, restricted_coordinate_value);

        const BoundingBox<dim - 1> box = create_unit_bounding_box<dim - 1>();

        std::pair<double, double>                      value_bounds;
        std::array<std::pair<double, double>, dim - 1> gradient_bounds;

        FunctionTools::taylor_estimate_function_bounds<dim - 1>(
          face_restriction, box, value_bounds, gradient_bounds);
        take_min_max_at_vertices(face_restriction, box, value_bounds);


        return location_from_min_max_levelset_values(value_bounds.first,
                                                     value_bounds.second);
      }



      template <int dim>
      LocationToLevelSet
      FunctionBasedClassifier<dim>::determine_cell_location_to_levelset(
        const typename Triangulation<dim>::active_cell_iterator &,
        const std::array<LocationToLevelSet, GeometryInfo<dim>::faces_per_cell>
          &face_locations)
      {
        // If any of the faces have different LocationToLevelset the cell is
        // intersected.
        for (unsigned int face = 1; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          if (face_locations[face] != face_locations[0])
            return LocationToLevelSet::INTERSECTED;

        // If we got here all functions have the same sign.
        return face_locations[0];
      }

    } // namespace CutMeshClassifierImplementation
  }   // namespace internal



  using namespace internal::CutMeshClassifierImplementation;

  template <int dim>
  template <class VECTOR>
  CutMeshClassifier<dim>::CutMeshClassifier(
    const Triangulation<dim> &triangulation,
    const DoFHandler<dim> &   level_set_dof_handler,
    const VECTOR &            level_set)
    : triangulation(&triangulation)
    , cell_face_classifier(
        new DofBasedClassifier<dim, VECTOR>(level_set_dof_handler, level_set))
  {
    Assert(&triangulation == &level_set_dof_handler.get_triangulation(),
           ExcMessage(
             "The DoFHandler of the level set function must use the same "
             "triangulation as the one that should be classified."));
  }



  template <int dim>
  CutMeshClassifier<dim>::CutMeshClassifier(
    const Triangulation<dim> &triangulation,
    const Function<dim> &     level_set)
    : triangulation(&triangulation)
    , cell_face_classifier(new FunctionBasedClassifier<dim>(level_set))
  {}



  template <int dim>
  void
  CutMeshClassifier<dim>::reclassify()
  {
    // Loop over all cells and determine category of all locally owned
    // cells and faces.
    for (const auto &cell : triangulation->active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            CellAndFaceLocations<dim> &locations = categories[cell];
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                locations.face_locations[f] =
                  cell_face_classifier->determine_face_location_to_levelset(
                    cell, f);
              }

            locations.cell_location =
              cell_face_classifier->determine_cell_location_to_levelset(
                cell, locations.face_locations);
          }
      }
  }



  template <int dim>
  LocationToLevelSet
  CutMeshClassifier<dim>::location_to_level_set(
    const typename Triangulation<dim>::cell_iterator &cell) const
  {
    return categories.at(cell).cell_location;
  }



  template <int dim>
  LocationToLevelSet
  CutMeshClassifier<dim>::location_to_level_set(
    const typename Triangulation<dim>::cell_iterator &cell,
    const unsigned int                                face_index) const
  {
    AssertIndexRange(face_index, GeometryInfo<dim>::faces_per_cell);

    const CellAndFaceLocations<dim> &data = categories.at(cell);
    return data.face_locations[face_index];
  }

} // namespace NonMatching

#include "cut_mesh_classifier.inst"

DEAL_II_NAMESPACE_CLOSE
