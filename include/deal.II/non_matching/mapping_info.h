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


#ifndef dealii_non_matching_mapping_info_h
#define dealii_non_matching_mapping_info_h


#include <deal.II/base/config.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_related_data.h>

#include <memory>


DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  /**
   * This class provides the mapping information computation and mapping data
   * storage to be used together with FEPointEvaluation.
   */
  template <int dim, int spacedim = dim>
  class MappingInfo : public Subscriptor
  {
  public:
    /**
     * Constructor.
     *
     * @param mapping The Mapping class describing the geometry of a cell.
     *
     * @param update_flags Specify the quantities to be computed by the mapping
     * during the call of reinit(). These update flags are also handed to a
     * FEEvaluation object if you construct it with this MappingInfo object.
     */
    MappingInfo(const Mapping<dim> &mapping, const UpdateFlags update_flags);

    /**
     * Reinitialize the mapping information for the incoming cell and unit
     * points.
     */
    void
    reinit(const typename Triangulation<dim, spacedim>::cell_iterator &cell,
           const ArrayView<const Point<dim>> &unit_points);

    /**
     * Reinitialize the mapping information for the incoming vector of cells and
     * corresponding vector of unit points.
     */
    void
    reinit_cells(
      const std::vector<typename Triangulation<dim, spacedim>::cell_iterator>
                                                 &cell_iterator_vector,
      const std::vector<std::vector<Point<dim>>> &unit_points_vector);

    /**
     * Reinitialize the mapping information for all faces of the incoming vector
     * of cells and corresponding vector of unit points.
     */
    void
    reinit_faces(
      const std::vector<typename Triangulation<dim, spacedim>::cell_iterator>
        cell_iterator_vector,
      const std::vector<std::vector<std::vector<Point<dim>>>>
        &unit_points_vector);

    /**
     * Getter function for current unit points.
     */
    const std::vector<Point<dim>> &
    get_unit_points(const unsigned int active_cell_index,
                    const unsigned int face_number) const;

    /**
     * Getter function for computed mapping data. This function accesses
     * internal data and is therefore not a stable interface.
     */
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &
      get_mapping_data(const unsigned int active_cell_index,
                       const unsigned int face_number) const;

    /**
     * Getter function for underlying mapping.
     */
    const Mapping<dim, spacedim> &
    get_mapping() const;

    /**
     * Getter function for the update flags.
     */
    UpdateFlags
    get_update_flags() const;

    using MappingData =
      dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                   spacedim>;
    using FaceData = std::vector<MappingData>;
    using CellData = std::vector<FaceData>;

    /**
     * Compute the mapping related data for the given @p mapping,
     * @p cell and @p unit_points that is required by the FEPointEvaluation
     * class.
     */
    void
    compute_mapping_data_for_generic_points(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const ArrayView<const Point<dim>>                          &unit_points,
      MappingData                                                &mapping_data);

    /**
     * The reference points specified at reinit().
     */
    using UnitPoints = std::vector<Point<dim>>;
    using FaceVector = std::vector<UnitPoints>;
    using CellVector = std::vector<FaceVector>;
    CellVector unit_points;

    /**
     * A pointer to the underlying mapping.
     */
    const SmartPointer<const Mapping<dim, spacedim>> mapping;

    /**
     * The desired update flags for the evaluation.
     */
    const UpdateFlags update_flags;

    /**
     * The update flags for the desired mapping information.
     */
    UpdateFlags update_flags_mapping;

    /**
     * The internal data container for mapping information. The implementation
     * is subject to future changes.
     */
    CellData mapping_data;

    /**
     * A map from the active_cell_index of a CellAccessor to the index where the
     * mapping is stores in the CellData vector.
     */
    std::map<unsigned int, unsigned int> cell_index_to_mapping_info_cell_id;
  };

  // ----------------------- template functions ----------------------


  template <int dim, int spacedim>
  MappingInfo<dim, spacedim>::MappingInfo(const Mapping<dim> &mapping,
                                          const UpdateFlags   update_flags)
    : mapping(&mapping)
    , update_flags(update_flags)
  {
    update_flags_mapping = update_default;
    // translate update flags
    if (update_flags & update_jacobians)
      update_flags_mapping |= update_jacobians;
    if (update_flags & update_gradients ||
        update_flags & update_inverse_jacobians)
      update_flags_mapping |= update_inverse_jacobians;

    // always save quadrature points for now
    update_flags_mapping |= update_quadrature_points;
  }



  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::reinit(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const ArrayView<const Point<dim>>                          &unit_points_in)
  {
    FaceVector dummy(1);
    unit_points.resize(1, dummy);
    unit_points[0][0] =
      std::vector<Point<dim>>(unit_points_in.begin(), unit_points_in.end());

    FaceData dummy_2(1);
    mapping_data.resize(1, dummy_2);
    compute_mapping_data_for_generic_points(cell,
                                            unit_points_in,
                                            mapping_data[0][0]);
  }


  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::reinit_cells(
    const std::vector<typename Triangulation<dim, spacedim>::cell_iterator>
                                               &cell_vector,
    const std::vector<std::vector<Point<dim>>> &unit_points_vector)
  {
    Assert(cell_vector.size() == unit_points_vector.size(),
           ExcDimensionMismatch(cell_vector.size(), unit_points_vector.size()));

    const unsigned int n_cells = cell_vector.size();

    FaceVector dummy(1);
    unit_points.resize(n_cells, dummy);
    FaceData dummy_2(1);
    mapping_data.resize(n_cells, dummy_2);

    for (unsigned int i = 0; i < n_cells; ++i)
      {
        unit_points[i][0] =
          std::vector<Point<dim>>(unit_points_vector[i].begin(),
                                  unit_points_vector[i].end());

        compute_mapping_data_for_generic_points(cell_vector[i],
                                                unit_points[i][0],
                                                mapping_data[i][0]);

        cell_index_to_mapping_info_cell_id.insert(
          {cell_vector[i]->active_cell_index(), i});
      }
  }


  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::reinit_faces(
    const std::vector<typename Triangulation<dim, spacedim>::cell_iterator>
      cell_iterator_vector,
    const std::vector<std::vector<std::vector<Point<dim>>>> &unit_points_vector)
  {
    Assert(cell_iterator_vector.size() == unit_points_vector.size(),
           ExcDimensionMismatch(cell_iterator_vector.size(),
                                unit_points_vector.size()));

    const unsigned int n_cells = cell_iterator_vector.size();

    unit_points.resize(n_cells);
    mapping_data.resize(n_cells);

    for (unsigned int i = 0; i < n_cells; ++i)
      {
        const auto &cell = cell_iterator_vector[i];

        Assert(unit_points_vector[i].size() == cell->n_faces(),
               ExcDimensionMismatch(unit_points_vector[i].size(),
                                    cell->n_faces()));

        cell_index_to_mapping_info_cell_id.insert(
          {cell->active_cell_index(), i});

        unit_points[i].resize(cell->n_faces());
        mapping_data[i].resize(cell->n_faces());
        for (const auto &f : cell->face_indices())
          {
            unit_points[i][f] =
              std::vector<Point<dim>>(unit_points_vector[i][f].begin(),
                                      unit_points_vector[i][f].end());

            compute_mapping_data_for_generic_points(cell,
                                                    unit_points[i][f],
                                                    mapping_data[i][f]);
          }
      }
  }



  template <int dim, int spacedim>
  const std::vector<Point<dim>> &
  MappingInfo<dim, spacedim>::get_unit_points(
    const unsigned int active_cell_index,
    const unsigned int face_number) const
  {
    if (active_cell_index == numbers::invalid_unsigned_int &&
        face_number == numbers::invalid_unsigned_int)
      return unit_points[0][0];
    else if (face_number == numbers::invalid_unsigned_int)
      return unit_points
        [cell_index_to_mapping_info_cell_id.find(active_cell_index)->second][0];
    else if (active_cell_index != numbers::invalid_unsigned_int)
      return unit_points[cell_index_to_mapping_info_cell_id
                           .find(active_cell_index)
                           ->second][face_number];
    else
      AssertThrow(
        false,
        ExcMessage(
          "active_cell_index has to be specified if face number is specified"));
  }



  template <int dim, int spacedim>
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                     spacedim> &
  MappingInfo<dim, spacedim>::get_mapping_data(
    const unsigned int active_cell_index,
    const unsigned int face_number) const
  {
    if (active_cell_index == numbers::invalid_unsigned_int &&
        face_number == numbers::invalid_unsigned_int)
      return mapping_data[0][0];
    else if (face_number == numbers::invalid_unsigned_int)
      return mapping_data
        [cell_index_to_mapping_info_cell_id.find(active_cell_index)->second][0];
    else if (active_cell_index != numbers::invalid_unsigned_int)
      return mapping_data[cell_index_to_mapping_info_cell_id
                            .find(active_cell_index)
                            ->second][face_number];
    else
      AssertThrow(
        false,
        ExcMessage(
          "active_cell_index has to be specified if face number is specified"));
  }



  template <int dim, int spacedim>
  const Mapping<dim, spacedim> &
  MappingInfo<dim, spacedim>::get_mapping() const
  {
    return *mapping;
  }



  template <int dim, int spacedim>
  UpdateFlags
  MappingInfo<dim, spacedim>::get_update_flags() const
  {
    return update_flags;
  }



  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::compute_mapping_data_for_generic_points(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const ArrayView<const Point<dim>>                          &unit_points,
    MappingData                                                &mapping_data)
  {
    if (const MappingQ<dim, spacedim> *mapping_q =
          dynamic_cast<const MappingQ<dim, spacedim> *>(&(*mapping)))
      {
        mapping_q->fill_mapping_data_for_generic_points(cell,
                                                        unit_points,
                                                        update_flags_mapping,
                                                        mapping_data);
      }
    else if (const MappingCartesian<dim, spacedim> *mapping_cartesian =
               dynamic_cast<const MappingCartesian<dim, spacedim> *>(
                 &(*mapping)))
      {
        mapping_cartesian->fill_mapping_data_for_generic_points(
          cell, unit_points, update_flags_mapping, mapping_data);
      }
    else
      {
        FE_DGQ<dim, spacedim>           dummy_fe(1);
        dealii::FEValues<dim, spacedim> fe_values(
          *mapping,
          dummy_fe,
          Quadrature<dim>(
            std::vector<Point<dim>>(unit_points.begin(), unit_points.end())),
          update_flags_mapping);
        fe_values.reinit(cell);
        mapping_data.initialize(unit_points.size(), update_flags_mapping);
        if (update_flags_mapping & update_jacobians)
          for (unsigned int q = 0; q < unit_points.size(); ++q)
            mapping_data.jacobians[q] = fe_values.jacobian(q);
        if (update_flags_mapping & update_inverse_jacobians)
          for (unsigned int q = 0; q < unit_points.size(); ++q)
            mapping_data.inverse_jacobians[q] = fe_values.inverse_jacobian(q);
        if (update_flags_mapping & update_quadrature_points)
          for (unsigned int q = 0; q < unit_points.size(); ++q)
            mapping_data.quadrature_points[q] = fe_values.quadrature_point(q);
      }
  }
} // namespace NonMatching

DEAL_II_NAMESPACE_CLOSE

#endif
