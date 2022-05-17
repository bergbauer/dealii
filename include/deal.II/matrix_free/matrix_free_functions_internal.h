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

#ifndef dealii_matrix_free_functions_internal_h
#define dealii_matrix_free_functions_internal_h


DEAL_II_NAMESPACE_OPEN

namespace internal
{
  namespace MatrixFreeFunctions
  {
    // steps through all children and adds the active cells recursively
    template <typename InIterator>
    void
    resolve_cell(const InIterator &                                  cell,
                 std::vector<std::pair<unsigned int, unsigned int>> &cell_its)
    {
      if (cell->has_children())
        for (unsigned int child = 0; child < cell->n_children(); ++child)
          resolve_cell(cell->child(child), cell_its);
      else if (cell->is_locally_owned())
        {
          Assert(cell->is_active(), ExcInternalError());
          cell_its.emplace_back(cell->level(), cell->index());
        }
    }
  } // namespace MatrixFreeFunctions

  namespace MatrixFreeImplementation
  {
    template <int dim, int spacedim>
    inline std::vector<IndexSet>
    extract_locally_owned_index_sets(
      const std::vector<const ::dealii::DoFHandler<dim, spacedim> *> &dofh,
      const unsigned int                                              level)
    {
      std::vector<IndexSet> locally_owned_set;
      locally_owned_set.reserve(dofh.size());
      for (unsigned int j = 0; j < dofh.size(); ++j)
        if (level == numbers::invalid_unsigned_int)
          locally_owned_set.push_back(dofh[j]->locally_owned_dofs());
        else
          locally_owned_set.push_back(dofh[j]->locally_owned_mg_dofs(level));
      return locally_owned_set;
    }
  } // namespace MatrixFreeImplementation
} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
