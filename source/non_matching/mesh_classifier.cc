// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2021 by the deal.II authors
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

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_element_access.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <algorithm>

DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  namespace internal
  {
    namespace MeshClassifierImplementation
    {
      /**
       * Return LocationToLevelSet::inside/outside if all values in incoming
       * vector are negative/positive, otherwise return
       * LocationToLevelSet::intersected.
       */
      template <class VECTOR>
      LocationToLevelSet
      location_from_dof_signs(const VECTOR &local_levelset_values)
      {
        const auto min_max_element =
          std::minmax_element(local_levelset_values.begin(),
                              local_levelset_values.end());

        if (*min_max_element.second < 0)
          return LocationToLevelSet::inside;
        if (0 < *min_max_element.first)
          return LocationToLevelSet::outside;

        return LocationToLevelSet::intersected;
      }



      /**
       * The concrete LevelSetDescription used when the level set function is
       * described as a (DoFHandler, Vector)-pair.
       */
      template <int dim, class VECTOR>
      class DiscreteLevelSetDescription : public LevelSetDescription<dim>
      {
      public:
        /**
         * Constructor.
         */
        DiscreteLevelSetDescription(const DoFHandler<dim> &dof_handler,
                                    const VECTOR &         level_set);

        /**
         * Return the FECollection of the DoFHandler passed to the constructor.
         */
        const hp::FECollection<dim> &
        get_fe_collection() const override;

        /**
         * Return the active_fe_index of the DoFCellAccessor associated with the
         * DoFHandler and the the incoming cell in the triangulation.
         */
        unsigned int
        active_fe_index(const typename Triangulation<dim>::active_cell_iterator
                          &cell) const override;

        /**
         * Writes the local cell dofs of the global level set vector to
         * @p local_levelset_values.
         */
        void
        get_local_level_set_values(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          Vector<double> &local_levelset_values) override;

        /**
         * Writes the local face dofs of the global level set vector to
         * @p local_levelset_values.
         */
        void
        get_local_level_set_values(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const unsigned int                                       face_index,
          Vector<double> &local_levelset_values) override;

      private:
        /**
         * Pointer to the DoFHandler associated with the level set function.
         */
        const SmartPointer<const DoFHandler<dim>> dof_handler;

        /**
         * Pointer to the vector global dof values of the level set function.
         */
        const SmartPointer<const VECTOR> level_set;
      };



      template <int dim, class VECTOR>
      DiscreteLevelSetDescription<dim, VECTOR>::DiscreteLevelSetDescription(
        const DoFHandler<dim> &dof_handler,
        const VECTOR &         level_set)
        : dof_handler(&dof_handler)
        , level_set(&level_set)
      {}



      template <int dim, class VECTOR>
      void
      DiscreteLevelSetDescription<dim, VECTOR>::get_local_level_set_values(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        Vector<double> &local_levelset_values)
      {
        typename DoFHandler<dim>::active_cell_iterator cell_with_dofs(
          &dof_handler->get_triangulation(),
          cell->level(),
          cell->index(),
          dof_handler);

        // Get the dofs indices associated with the face.
        const unsigned int n_dofs_per_cell =
          cell_with_dofs->get_fe().n_dofs_per_cell();
        std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
        cell_with_dofs->get_dof_indices(dof_indices);

        local_levelset_values.reinit(dof_indices.size());

        for (unsigned int i = 0; i < dof_indices.size(); i++)
          local_levelset_values[i] =
            dealii::internal::ElementAccess<VECTOR>::get(*level_set,
                                                         dof_indices[i]);
      }



      template <int dim, class VECTOR>
      const hp::FECollection<dim> &
      DiscreteLevelSetDescription<dim, VECTOR>::get_fe_collection() const
      {
        return dof_handler->get_fe_collection();
      }



      template <int dim, class VECTOR>
      void
      DiscreteLevelSetDescription<dim, VECTOR>::get_local_level_set_values(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        const unsigned int                                       face_index,
        Vector<double> &local_levelset_values)
      {
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

        local_levelset_values.reinit(dof_indices.size());

        for (unsigned int i = 0; i < dof_indices.size(); i++)
          local_levelset_values[i] =
            dealii::internal::ElementAccess<VECTOR>::get(*level_set,
                                                         dof_indices[i]);
      }



      template <int dim, class VECTOR>
      unsigned int
      DiscreteLevelSetDescription<dim, VECTOR>::active_fe_index(
        const typename Triangulation<dim>::active_cell_iterator &cell) const
      {
        typename DoFHandler<dim>::active_cell_iterator cell_with_dofs(
          &dof_handler->get_triangulation(),
          cell->level(),
          cell->index(),
          dof_handler);

        return cell_with_dofs->active_fe_index();
      }


      /**
       * The concrete LevelSetDescription used when the level set function is
       * described by a Function.
       */
      template <int dim>
      class AnalyticLevelSetDescription : public LevelSetDescription<dim>
      {
      public:
        /**
         * Constructor. Takes the Function that describes the geometry and the
         * element that this function should be interpolated to.
         */
        AnalyticLevelSetDescription(const Function<dim> &     level_set,
                                    const FiniteElement<dim> &element);

        /**
         * Returns the finite element passed to the constructor wrapped in a
         * collection.
         */
        const hp::FECollection<dim> &
        get_fe_collection() const override;

        /**
         * Returns 0, since there is always a single element in the
         * FECollection.
         */
        unsigned int
        active_fe_index(const typename Triangulation<dim>::active_cell_iterator
                          &cell) const override;

        /**
         * Return the level set function evaluated at the real space support
         * points of the finite element passed to the constructor.
         */
        void
        get_local_level_set_values(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          Vector<double> &local_levelset_values) override;

        /**
         * Return the level set function evaluated at the real space face
         * support points of the finite element passed to the constructor.
         */
        void
        get_local_level_set_values(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const unsigned int                                       face_index,
          Vector<double> &local_levelset_values) override;

      private:
        /**
         * Pointer to the level set function.
         */
        const SmartPointer<const Function<dim>> level_set;

        /**
         * Collection containing the single element which we locally interpolate
         * the level set function onto.
         */
        const hp::FECollection<dim> fe_collection;

        /**
         * FEValues object used to transform the support points on a cell to
         * real space.
         */
        FEValues<dim> fe_values;

        /**
         * FEFaceValues object used to transform the support points on a face to
         * real space.
         */
        FEFaceValues<dim> fe_face_values;
      };



      template <int dim>
      AnalyticLevelSetDescription<dim>::AnalyticLevelSetDescription(
        const Function<dim> &     level_set,
        const FiniteElement<dim> &element)
        : level_set(&level_set)
        , fe_collection(element)
        , fe_values(element,
                    Quadrature<dim>(element.get_unit_support_points()),
                    update_quadrature_points)
        , fe_face_values(element,
                         Quadrature<dim - 1>(
                           element.get_unit_face_support_points()),
                         update_quadrature_points)
      {}



      template <int dim>
      void
      AnalyticLevelSetDescription<dim>::get_local_level_set_values(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        Vector<double> &local_levelset_values)
      {
        fe_values.reinit(cell);
        const std::vector<Point<dim>> &points =
          fe_values.get_quadrature_points();

        for (unsigned int i = 0; i < points.size(); i++)
          local_levelset_values[i] = level_set->value(points[i]);
      }



      template <int dim>
      void
      AnalyticLevelSetDescription<dim>::get_local_level_set_values(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        const unsigned int                                       face_index,
        Vector<double> &local_levelset_values)
      {
        fe_face_values.reinit(cell, face_index);
        const std::vector<Point<dim>> &points =
          fe_face_values.get_quadrature_points();

        for (unsigned int i = 0; i < points.size(); i++)
          local_levelset_values[i] = level_set->value(points[i]);
      }



      template <int dim>
      const hp::FECollection<dim> &
      AnalyticLevelSetDescription<dim>::get_fe_collection() const
      {
        return fe_collection;
      }



      template <int dim>
      unsigned int
      AnalyticLevelSetDescription<dim>::active_fe_index(
        const typename Triangulation<dim>::active_cell_iterator &) const
      {
        return 0;
      }
    } // namespace MeshClassifierImplementation
  }   // namespace internal



  using namespace internal::MeshClassifierImplementation;

  template <int dim>
  template <class VECTOR>
  MeshClassifier<dim>::MeshClassifier(const DoFHandler<dim> &dof_handler,
                                      const VECTOR &         level_set)
    : triangulation(&dof_handler.get_triangulation())
    , level_set_description(
        std::make_unique<DiscreteLevelSetDescription<dim, VECTOR>>(dof_handler,
                                                                   level_set))
  {}



  template <int dim>
  MeshClassifier<dim>::MeshClassifier(const Triangulation<dim> &triangulation,
                                      const Function<dim> &     level_set,
                                      const FiniteElement<dim> &element)
    : triangulation(&triangulation)
    , level_set_description(
        std::make_unique<AnalyticLevelSetDescription<dim>>(level_set, element))
  {
    // The level set function must be scalar.
    AssertDimension(level_set.n_components, 1);
    AssertDimension(element.n_components(), 1);
  }



  template <int dim>
  void
  MeshClassifier<dim>::reclassify()
  {
    initialize();

    // Loop over all cells and determine category of all locally owned
    // cells and faces.
    for (const auto &cell : triangulation->active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                face_locations[cell->face(f)] =
                  determine_face_location_to_levelset(cell, f);
              }

            cell_locations[cell] = determine_cell_location_to_levelset(cell);
          }
      }
  }



  template <int dim>
  LocationToLevelSet
  MeshClassifier<dim>::determine_cell_location_to_levelset(
    const typename Triangulation<dim>::active_cell_iterator &cell)
  {
    const unsigned int fe_index = level_set_description->active_fe_index(cell);

    const unsigned int n_local_dofs = lagrange_to_bernstein_cell[fe_index].m();

    Vector<double> local_levelset_values(n_local_dofs);
    level_set_description->get_local_level_set_values(cell,
                                                      local_levelset_values);

    lagrange_to_bernstein_cell[fe_index].solve(local_levelset_values);

    return location_from_dof_signs(local_levelset_values);
  }



  template <int dim>
  LocationToLevelSet
  MeshClassifier<dim>::determine_face_location_to_levelset(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index)
  {
    const unsigned int fe_index = level_set_description->active_fe_index(cell);

    const unsigned int n_local_dofs =
      lagrange_to_bernstein_face[fe_index][face_index].m();

    Vector<double> local_levelset_values(n_local_dofs);
    level_set_description->get_local_level_set_values(cell,
                                                      face_index,
                                                      local_levelset_values);

    lagrange_to_bernstein_face[fe_index][face_index].solve(
      local_levelset_values);

    return location_from_dof_signs(local_levelset_values);
  }



  template <int dim>
  LocationToLevelSet
  MeshClassifier<dim>::location_to_level_set(
    const typename Triangulation<dim>::cell_iterator &cell) const
  {
    return cell_locations.at(cell);
  }



  template <int dim>
  LocationToLevelSet
  MeshClassifier<dim>::location_to_level_set(
    const typename Triangulation<dim>::cell_iterator &cell,
    const unsigned int                                face_index) const
  {
    AssertIndexRange(face_index, GeometryInfo<dim>::faces_per_cell);

    return face_locations.at(cell->face(face_index));
  }



  template <int dim>
  void
  MeshClassifier<dim>::initialize()
  {
    const hp::FECollection<dim> &fe_collection =
      level_set_description->get_fe_collection();

    // The level set function must be scalar.
    AssertDimension(fe_collection.n_components(), 1);

    lagrange_to_bernstein_cell.resize(fe_collection.size());
    lagrange_to_bernstein_face.resize(fe_collection.size());

    for (unsigned int i = 0; i < fe_collection.size(); i++)
      {
        const FiniteElement<dim> &element = fe_collection[i];
        const FE_Q<dim> *fe_q = dynamic_cast<const FE_Q<dim> *>(&element);
        Assert(fe_q != nullptr, ExcNotImplemented());

        const FE_Bernstein<dim> fe_bernstein(fe_q->get_degree());

        const unsigned int dofs_per_cell = fe_q->dofs_per_cell;
        FullMatrix<double> interpolation_matrix(dofs_per_cell, dofs_per_cell);

        fe_q->get_interpolation_matrix(fe_bernstein, interpolation_matrix);

        lagrange_to_bernstein_cell[i].reinit(dofs_per_cell);
        lagrange_to_bernstein_cell[i] = interpolation_matrix;
        lagrange_to_bernstein_cell[i].compute_lu_factorization();

        const unsigned int dofs_per_face = fe_q->dofs_per_face;
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
          {
            FullMatrix<double> face_interpolation_matrix(dofs_per_face,
                                                         dofs_per_face);

            fe_bernstein.get_face_interpolation_matrix(
              *fe_q, face_interpolation_matrix, f);
            lagrange_to_bernstein_face[i][f].reinit(dofs_per_face);
            lagrange_to_bernstein_face[i][f] = face_interpolation_matrix;
            lagrange_to_bernstein_face[i][f].compute_lu_factorization();
          }
      }
  }

} // namespace NonMatching

#include "mesh_classifier.inst"

DEAL_II_NAMESPACE_CLOSE
