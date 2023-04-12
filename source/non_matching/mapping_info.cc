// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
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

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/fe/mapping_q_internal.h>

DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::compute_mapping_data_for_face_quadrature(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const Quadrature<dim - 1> &                                 quadrature,
    MappingData &                                               mapping_data)
  {
    if (const MappingQ<dim, spacedim> *mapping_q =
          dynamic_cast<const MappingQ<dim, spacedim> *>(&(*mapping)))
      {
        if (update_flags == update_default)
          return;
        
        if (quadrature.get_points().empty())
          return;

        Assert(update_flags & (update_inverse_jacobians | update_jacobians |
                               update_quadrature_points | update_normal_vectors |
                               update_JxW_values),
               ExcNotImplemented());

        update_flags_mapping |=
          mapping->requires_update_flags(update_flags_mapping);

        mapping_data.initialize(quadrature.size(), update_flags_mapping);

        std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase> data_ptr =
          std::make_unique<typename MappingQ<dim, spacedim>::InternalData>(mapping_q->get_degree());
        auto &data = dynamic_cast<typename MappingQ<dim, spacedim>::InternalData &>(*data_ptr);
        data.initialize_face(update_flags_mapping,
                             QProjector<dim>::project_to_oriented_face(
                               ReferenceCells::get_hypercube<dim>(),
                               quadrature,
                               face_no,
                               cell->face_orientation(face_no),
                               cell->face_flip(face_no),
                               cell->face_rotation(face_no)),
                             quadrature.size());

        data.mapping_support_points = mapping_q->compute_mapping_support_points(cell);

        internal::MappingQImplementation::do_fill_fe_face_values(
          mapping_q,
          cell,
          face_no,
          numbers::invalid_unsigned_int,
          QProjector<dim>::DataSetDescriptor::cell(),
          quadrature,
          data,
          mapping_q->polynomials_1d,
          mapping_q->polynomial_degree,
          mapping_q->renumber_lexicographic_to_hierarchic,
          mapping_data);
      }
    else
      {
        update_flags_mapping |=
          mapping->requires_update_flags(update_flags_mapping);
        
        mapping_data.initialize(quadrature.get_points().size(),
                                update_flags_mapping);

        auto internal_mapping_data =
          mapping->get_face_data(update_flags_mapping,
                                 hp::QCollection<dim - 1>(quadrature));

        mapping->fill_fe_face_values(cell,
                                     face_no,
                                     hp::QCollection<dim - 1>(quadrature),
                                     *internal_mapping_data,
                                     mapping_data);
      }
  }


} // namespace NonMatching
#include "mapping_info.inst"
DEAL_II_NAMESPACE_CLOSE
