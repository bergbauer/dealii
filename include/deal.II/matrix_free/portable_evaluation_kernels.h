// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2017 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#ifndef dealii__evaluation_kernels_h
#define dealii__evaluation_kernels_h

#include <deal.II/base/config.h>

#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_tensor_product_kernels.h>

#include <Kokkos_Core.hpp>


DEAL_II_NAMESPACE_OPEN


namespace Portable
{
  namespace internal
  {
    /**
     * Helper function to specify whether a transformation to collocation should
     * be used: It should give correct results (first condition), we need to be
     * able to initialize the fields in shape_info.templates.h from the
     * polynomials (second condition), and it should be the most efficient
     * choice in terms of operation counts (third condition).
     */
    constexpr bool
    use_collocation_evaluation(const unsigned int fe_degree,
                               const unsigned int n_q_points_1d)
    {
      // TODO: are the conditions suit for GPU parallelization?
      return (n_q_points_1d > fe_degree) && (n_q_points_1d < 200) &&
             (n_q_points_1d <= 3 * fe_degree / 2 + 1);
    }



    template <int dim, typename Number>
    DEAL_II_HOST_DEVICE auto
    get_shape_values_view(const typename MatrixFree<dim, Number>::Data *data)
    {
      using ShapeView = decltype(data->precomputed_data->shape_values);

      if (data->shared_data->use_shared_shape_data)
        return ShapeView(data->shared_data->shape_values.data(),
                         data->shared_data->shape_values.extent(0));

      return data->precomputed_data->shape_values;
    }



    template <int dim, typename Number>
    DEAL_II_HOST_DEVICE auto
    get_shape_gradients_view(const typename MatrixFree<dim, Number>::Data *data)
    {
      using ShapeView = decltype(data->precomputed_data->shape_gradients);

      if (data->shared_data->use_shared_shape_data)
        return ShapeView(data->shared_data->shape_gradients.data(),
                         data->shared_data->shape_gradients.extent(0));

      return data->precomputed_data->shape_gradients;
    }



    /**
     * This struct performs the evaluation of function values and gradients for
     * tensor-product finite elements. There are two specialized implementation
     * classes FEEvaluationImplCollocation (for Gauss-Lobatto elements where the
     * nodal points and the quadrature points coincide and the 'values'
     * operation is identity) and FEEvaluationImplTransformToCollocation (which
     * can be transformed to a collocation space and can then use the identity
     * in these spaces), which both allow for shorter code.
     */
    template <int dim, int fe_degree, int n_q_points_1d, typename Number>
    struct FEEvaluationImpl
    {
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;
      using SharedView = Kokkos::View<Number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      template <typename EvalType, typename UViewType, typename GradViewType>
      DEAL_II_HOST_DEVICE static void
      do_evaluate(EvalType                                     &eval,
                  const EvaluationFlags::EvaluationFlags        evaluation_flag,
                  const typename MatrixFree<dim, Number>::Data *data,
                  const UViewType                              &u,
                  const GradViewType                           &grad_u)
      {
        if constexpr (dim == 1)
          {
            const auto evaluate_dim1 = [&](const auto &temp,
                                           const bool  in_place) {
              if (evaluation_flag & EvaluationFlags::gradients)
                eval.template gradients<0, true, false, false>(
                  u, Kokkos::subview(grad_u, Kokkos::ALL, 0));
              if (evaluation_flag & EvaluationFlags::values)
                {
                  eval.template values<0, true, false, false>(u, temp);
                  populate_view<false>(
                    data->team_member, u, temp, n_q_points_1d, in_place);
                }
            };

            if (data->thread_per_cell)
              {
                Number     temp_storage[n_q_points_1d];
                SharedView temp(temp_storage, n_q_points_1d);

                evaluate_dim1(temp, true);
              }
            else
              {
                auto temp =
                  Kokkos::subview(data->shared_data->scratch_pad,
                                  Kokkos::make_pair(0, n_q_points_1d));

                evaluate_dim1(temp, false);
              }
          }
        else if constexpr (dim == 2)
          {
            constexpr int temp_size = (fe_degree + 1) * n_q_points_1d;

            const auto evaluate_dim2 = [&](const auto &temp) {
              // grad x
              if (evaluation_flag & EvaluationFlags::gradients)
                {
                  eval.template gradients<0, true, false, false>(u, temp);
                  eval.template values<1, true, false, false>(
                    temp, Kokkos::subview(grad_u, Kokkos::ALL, 0));
                }

              // grad y
              eval.template values<0, true, false, false>(u, temp);
              if (evaluation_flag & EvaluationFlags::gradients)
                eval.template gradients<1, true, false, false>(
                  temp, Kokkos::subview(grad_u, Kokkos::ALL, 1));

              // val: can use values applied in x
              if (evaluation_flag & EvaluationFlags::values)
                eval.template values<1, true, false, false>(temp, u);
            };

            if (data->thread_per_cell)
              {
                Number     temp_storage[temp_size];
                SharedView temp(temp_storage, temp_size);

                evaluate_dim2(temp);
              }
            else
              {
                auto temp = Kokkos::subview(data->shared_data->scratch_pad,
                                            Kokkos::make_pair(0, temp_size));

                evaluate_dim2(temp);
              }
          }
        else if constexpr (dim == 3)
          {
            constexpr int temp1_size =
                            Utilities::pow(fe_degree + 1, 2) * n_q_points_1d,
                          temp2_size =
                            Utilities::pow(n_q_points_1d, 2) * (fe_degree + 1);

            const auto evaluate_dim3 = [&](const auto &temp1,
                                           const auto &temp2) {
              if (evaluation_flag & EvaluationFlags::gradients)
                {
                  // grad x
                  eval.template gradients<0, true, false, false>(u, temp1);
                  eval.template values<1, true, false, false>(temp1, temp2);
                  eval.template values<2, true, false, false>(
                    temp2, Kokkos::subview(grad_u, Kokkos::ALL, 0));
                }

              // grad y
              eval.template values<0, true, false, false>(u, temp1);
              if (evaluation_flag & EvaluationFlags::gradients)
                {
                  eval.template gradients<1, true, false, false>(temp1, temp2);
                  eval.template values<2, true, false, false>(
                    temp2, Kokkos::subview(grad_u, Kokkos::ALL, 1));
                }

              // grad z: can use the values applied in x direction stored
              // in temp1
              eval.template values<1, true, false, false>(temp1, temp2);
              if (evaluation_flag & EvaluationFlags::gradients)
                eval.template gradients<2, true, false, false>(
                  temp2, Kokkos::subview(grad_u, Kokkos::ALL, 2));

              // val: can use the values applied in x & y direction
              // stored in temp2
              if (evaluation_flag & EvaluationFlags::values)
                eval.template values<2, true, false, false>(temp2, u);
            };

            if (data->thread_per_cell)
              {
                Number     temp1_storage[temp1_size];
                Number     temp2_storage[temp2_size];
                SharedView temp1(temp1_storage, temp1_size);
                SharedView temp2(temp2_storage, temp2_size);

                evaluate_dim3(temp1, temp2);
              }
            else
              {
                auto temp1 = Kokkos::subview(data->shared_data->scratch_pad,
                                             Kokkos::make_pair(0, temp1_size));
                auto temp2 =
                  Kokkos::subview(data->shared_data->scratch_pad,
                                  Kokkos::make_pair(temp1_size,
                                                    temp1_size + temp2_size));

                evaluate_dim3(temp1, temp2);
              }
          }
        else
          Assert(false, ExcMessage("dim must not exceed 3!"));
      }


      template <typename EvalType, typename UViewType, typename GradViewType>
      DEAL_II_HOST_DEVICE static void
      do_integrate(EvalType                              &eval,
                   const EvaluationFlags::EvaluationFlags integration_flag,
                   const typename MatrixFree<dim, Number>::Data *data,
                   const UViewType                              &u,
                   const GradViewType                           &grad_u)
      {
        if constexpr (dim == 1)
          {
            const auto integrate_dim1 = [&](const auto &temp,
                                            const bool  in_place) {
              if ((integration_flag & EvaluationFlags::values) &&
                  !(integration_flag & EvaluationFlags::gradients))
                {
                  eval.template values<0, false, false, false>(u, temp);
                  populate_view<false>(
                    data->team_member, u, temp, fe_degree + 1, in_place);
                }
              if (integration_flag & EvaluationFlags::gradients)
                {
                  if (integration_flag & EvaluationFlags::values)
                    {
                      eval.template values<0, false, false, false>(u, temp);
                      eval.template gradients<0, false, true, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 0), temp);
                      populate_view<false>(
                        data->team_member, u, temp, fe_degree + 1, in_place);
                    }
                  else
                    eval.template gradients<0, false, false, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
                }
            };

            if (data->thread_per_cell)
              {
                Number     temp_storage[fe_degree + 1];
                SharedView temp(temp_storage, fe_degree + 1);

                integrate_dim1(temp, true);
              }
            else
              {
                auto temp =
                  Kokkos::subview(data->shared_data->scratch_pad,
                                  Kokkos::make_pair(0, fe_degree + 1));

                integrate_dim1(temp, false);
              }
          }
        else if constexpr (dim == 2)
          {
            constexpr int temp_size = (fe_degree + 1) * n_q_points_1d;

            const auto integrate_dim2 = [&](const auto &temp) {
              if ((integration_flag & EvaluationFlags::values) &&
                  !(integration_flag & EvaluationFlags::gradients))
                {
                  eval.template values<1, false, false, false>(u, temp);
                  eval.template values<0, false, false, false>(temp, u);
                }
              if (integration_flag & EvaluationFlags::gradients)
                {
                  eval.template gradients<1, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 1), temp);
                  if (integration_flag & EvaluationFlags::values)
                    eval.template values<1, false, true, false>(u, temp);
                  eval.template values<0, false, false, false>(temp, u);
                  eval.template values<1, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 0), temp);
                  eval.template gradients<0, false, true, false>(temp, u);
                }
            };

            if (data->thread_per_cell)
              {
                Number     temp_storage[temp_size];
                SharedView temp(temp_storage, temp_size);

                integrate_dim2(temp);
              }
            else
              {
                auto temp = Kokkos::subview(data->shared_data->scratch_pad,
                                            Kokkos::make_pair(0, temp_size));

                integrate_dim2(temp);
              }
          }
        else if constexpr (dim == 3)
          {
            constexpr int temp1_size =
                            Utilities::pow(n_q_points_1d, 2) * (fe_degree + 1),
                          temp2_size =
                            Utilities::pow(fe_degree + 1, 2) * n_q_points_1d;

            const auto integrate_dim3 = [&](const auto &temp1,
                                            const auto &temp2) {
              if ((integration_flag & EvaluationFlags::values) &&
                  !(integration_flag & EvaluationFlags::gradients))
                {
                  eval.template values<2, false, false, false>(u, temp1);
                  eval.template values<1, false, false, false>(temp1, temp2);
                  eval.template values<0, false, false, false>(temp2, u);
                }
              if (integration_flag & EvaluationFlags::gradients)
                {
                  eval.template gradients<2, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 2), temp1);
                  if (integration_flag & EvaluationFlags::values)
                    eval.template values<2, false, true, false>(u, temp1);
                  eval.template values<1, false, false, false>(temp1, temp2);
                  eval.template values<2, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 1), temp1);
                  eval.template gradients<1, false, true, false>(temp1, temp2);
                  eval.template values<0, false, false, false>(temp2, u);
                  eval.template values<2, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 0), temp1);
                  eval.template values<1, false, false, false>(temp1, temp2);
                  eval.template gradients<0, false, true, false>(temp2, u);
                }
            };

            if (data->thread_per_cell)
              {
                Number     temp1_storage[temp1_size];
                Number     temp2_storage[temp2_size];
                SharedView temp1(temp1_storage, temp1_size);
                SharedView temp2(temp2_storage, temp2_size);

                integrate_dim3(temp1, temp2);
              }
            else
              {
                auto temp1 = Kokkos::subview(data->shared_data->scratch_pad,
                                             Kokkos::make_pair(0, temp1_size));
                auto temp2 =
                  Kokkos::subview(data->shared_data->scratch_pad,
                                  Kokkos::make_pair(temp1_size,
                                                    temp1_size + temp2_size));

                integrate_dim3(temp1, temp2);
              }
          }
        else
          Assert(false, ExcMessage("dim must not exceed 3!"));
      }

      DEAL_II_HOST_DEVICE static void
      evaluate(const unsigned int                            n_components,
               const EvaluationFlags::EvaluationFlags        evaluation_flag,
               const typename MatrixFree<dim, Number>::Data *data)
      {
        if (evaluation_flag == EvaluationFlags::nothing)
          return;

        // the evaluator does not need temporary storage since no in-place
        // operation takes place in this function
        auto scratch_for_eval = Kokkos::subview(data->shared_data->scratch_pad,
                                                Kokkos::make_pair(0, 0));
        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree + 1,
                               n_q_points_1d,
                               Number>
          eval(data->team_member,
               get_shape_values_view<dim, Number>(data),
               get_shape_gradients_view<dim, Number>(data),
               data->precomputed_data->co_shape_gradients,
               scratch_for_eval,
               data->thread_per_cell);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            auto u = Kokkos::subview(data->shared_data->values, Kokkos::ALL, c);
            auto grad_u = Kokkos::subview(data->shared_data->gradients,
                                          Kokkos::ALL,
                                          Kokkos::ALL,
                                          c);

            do_evaluate(eval, evaluation_flag, data, u, grad_u);
          }
      }



      DEAL_II_HOST_DEVICE static void
      integrate(const unsigned int                            n_components,
                const EvaluationFlags::EvaluationFlags        integration_flag,
                const typename MatrixFree<dim, Number>::Data *data)
      {
        if (integration_flag == EvaluationFlags::nothing)
          return;

        // the evaluator does not need temporary storage since no in-place
        // operation takes place in this function
        auto scratch_for_eval = Kokkos::subview(data->shared_data->scratch_pad,
                                                Kokkos::make_pair(0, 0));
        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree + 1,
                               n_q_points_1d,
                               Number>
          eval(data->team_member,
               get_shape_values_view<dim, Number>(data),
               get_shape_gradients_view<dim, Number>(data),
               data->precomputed_data->co_shape_gradients,
               scratch_for_eval,
               data->thread_per_cell);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            auto u = Kokkos::subview(data->shared_data->values, Kokkos::ALL, c);
            auto grad_u = Kokkos::subview(data->shared_data->gradients,
                                          Kokkos::ALL,
                                          Kokkos::ALL,
                                          c);

            do_integrate(eval, integration_flag, data, u, grad_u);
          }
      }
    };



    template <int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename QuadOp>
    DEAL_II_HOST_DEVICE inline void
    apply_fused_thread_per_cell(
      const typename MatrixFree<dim, Number>::Data *data,
      const DeviceVector<Number>                   &src,
      DeviceVector<Number>                         &dst,
      const QuadOp                                 &quad_op,
      const EvaluationFlags::EvaluationFlags        evaluate_flag,
      const EvaluationFlags::EvaluationFlags        integrate_flag)
    {
      static_assert(
        dim >= 1 && dim <= 3,
        "apply_fused_thread_per_cell currently supports dim in [1,3].");

      constexpr int n_dofs_1d   = fe_degree + 1;
      constexpr int n_q_1d      = n_q_points_1d;
      constexpr int n_local_dof = Utilities::pow(n_dofs_1d, dim);
      constexpr int n_q_points  = Utilities::pow(n_q_1d, dim);
      constexpr int n_u_entries =
        (n_local_dof > n_q_points ? n_local_dof : n_q_points);

      constexpr int temp1_size =
        dim == 1 ? 0 :
        dim == 2 ? n_dofs_1d * n_q_1d :
                   Utilities::pow(n_dofs_1d, 2) * n_q_1d;
      constexpr int temp2_size =
        dim == 3 ? Utilities::pow(n_q_1d, 2) * n_dofs_1d : 0;
      constexpr int n_work_entries =
        temp1_size + temp2_size + (dim + 1) * n_q_points * n_components;

      const int cell = data->cell_index;

      auto local_to_global = data->precomputed_data->local_to_global;
      auto shape_values    = get_shape_values_view<dim, Number>(data);
      auto shape_gradients = get_shape_gradients_view<dim, Number>(data);
      auto inv_jacobian    = data->precomputed_data->inv_jacobian;
      auto jxw             = data->precomputed_data->JxW;

      struct QuadEvaluator
      {
        DEAL_II_HOST_DEVICE int
        get_current_cell_index() const
        {
          return cell;
        }

        DEAL_II_HOST_DEVICE const typename MatrixFree<dim, Number>::Data *
        get_matrix_free_data() const
        {
          return data;
        }

        using value_type = std::conditional_t<(n_components == 1),
                                              Number,
                                              Tensor<1, n_components, Number>>;

        using gradient_type = std::conditional_t<
          n_components == 1,
          Tensor<1, dim, Number>,
          std::conditional_t<n_components == dim,
                             Tensor<2, dim, Number>,
                             Tensor<1, n_components, Tensor<1, dim, Number>>>>;

        DEAL_II_HOST_DEVICE value_type
        get_value(const int q_point) const
        {
          if constexpr (n_components == 1)
            return val_q[q_point];
          else
            {
              value_type values;
              for (unsigned int c = 0; c < n_components; ++c)
                values[c] = val_q[c * n_q_points + q_point];
              return values;
            }
        }

        DEAL_II_HOST_DEVICE gradient_type
        get_gradient(const int q_point) const
        {
          if constexpr (n_components == 1)
            {
              Tensor<1, dim, Number> grad;
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  Number tmp = 0.;
                  for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                    tmp += inv_jacobian_view(q_point, cell, d_2, d_1) *
                           grad_ref_q[d_2][q_point];
                  grad[d_1] = tmp;
                }
              return grad;
            }
          else
            {
              gradient_type grad;
              for (unsigned int c = 0; c < n_components; ++c)
                for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                  {
                    Number tmp = 0.;
                    for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                      tmp += inv_jacobian_view(q_point, cell, d_2, d_1) *
                             grad_ref_q[c * dim + d_2][q_point];

                    if constexpr (n_components == dim)
                      grad[c][d_1] = tmp;
                    else
                      grad[c][d_1] = tmp;
                  }
              return grad;
            }
        }

        DEAL_II_HOST_DEVICE void
        submit_value(const value_type &val_in, const int q_point)
        {
          if constexpr (n_components == 1)
            val_q[q_point] = val_in * jxw_view(q_point, cell);
          else
            for (unsigned int c = 0; c < n_components; ++c)
              val_q[c * n_q_points + q_point] =
                val_in[c] * jxw_view(q_point, cell);
        }

        DEAL_II_HOST_DEVICE void
        submit_gradient(const gradient_type &grad_in, const int q_point)
        {
          if constexpr (n_components == 1)
            {
              // Transform real gradients back to reference space and apply
              // JxW.
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  Number tmp = 0.;
                  for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                    tmp +=
                      inv_jacobian_view(q_point, cell, d_1, d_2) * grad_in[d_2];
                  grad_ref_q[d_1][q_point] = tmp * jxw_view(q_point, cell);
                }
            }
          else
            {
              for (unsigned int c = 0; c < n_components; ++c)
                for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                  {
                    Number tmp = 0.;
                    for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                      {
                        if constexpr (n_components == dim)
                          tmp += inv_jacobian_view(q_point, cell, d_1, d_2) *
                                 grad_in[c][d_2];
                        else
                          tmp += inv_jacobian_view(q_point, cell, d_1, d_2) *
                                 grad_in[c][d_2];
                      }
                    grad_ref_q[c * dim + d_1][q_point] =
                      tmp * jxw_view(q_point, cell);
                  }
            }
        }

        const typename MatrixFree<dim, Number>::Data *data;
        decltype(inv_jacobian)                        inv_jacobian_view;
        decltype(jxw)                                 jxw_view;
        int                                           cell;
        Number                                       *val_q;
        Number                                      **grad_ref_q;
      };

      using LocalView = Kokkos::View<Number *,
                                     MemorySpace::Default::kokkos_space::
                                       execution_space::scratch_memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using LocalGradView =
        Kokkos::View<Number **,
                     Kokkos::LayoutLeft,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      Number u_buffer[n_u_entries];
      Number work[n_work_entries];

      Number *temp_1 = work;
      Number *temp_2 = temp_1 + temp1_size;
      Number *val_q  = temp_2 + temp2_size;

      Number *grad_ref_q[n_components * dim];
      for (int c = 0; c < n_components; ++c)
        for (int d = 0; d < dim; ++d)
          grad_ref_q[c * dim + d] =
            val_q + n_components * n_q_points + (c * dim + d) * n_q_points;

      // the evaluator does not need temporary storage since no in-place
      // operation takes place in do_evaluate itself
      auto scratch_for_eval = Kokkos::subview(data->shared_data->scratch_pad,
                                              Kokkos::make_pair(0, 0));
      EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                             dim,
                             fe_degree + 1,
                             n_q_points_1d,
                             Number>
        eval(data->team_member,
             shape_values,
             shape_gradients,
             data->precomputed_data->co_shape_gradients,
             scratch_for_eval,
             data->thread_per_cell);

      for (int c = 0; c < n_components; ++c)
        {
          const LocalView u_view(u_buffer, n_u_entries);
          const LocalView val_q_view(val_q + c * n_q_points, n_q_points);

          // gather
          for (int i = 0; i < n_local_dof; ++i)
            u_view(i) = src[local_to_global(i + c * n_local_dof, cell)];

          const LocalGradView grad_ref_q_view(grad_ref_q[c * dim],
                                              n_q_points,
                                              dim);

          // evaluate
          FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::do_evaluate(
            eval, evaluate_flag, data, u_view, grad_ref_q_view);

          // store quadrature values for this component
          for (int q = 0; q < n_q_points; ++q)
            val_q_view(q) = u_view(q);
        }

      // apply quadrature point operator
      QuadEvaluator quad_eval{data, inv_jacobian, jxw, cell, val_q, grad_ref_q};

      for (int q = 0; q < n_q_points; ++q)
        quad_op(&quad_eval, q);

      for (int c = 0; c < n_components; ++c)
        {
          const LocalView     u_view(u_buffer, n_u_entries);
          const LocalView     val_q_view(val_q + c * n_q_points, n_q_points);
          const LocalGradView grad_ref_q_view(grad_ref_q[c * dim],
                                              n_q_points,
                                              dim);

          // load quadrature values for this component into u_dof_view for
          // do_integrate
          for (int q = 0; q < n_q_points; ++q)
            u_view(q) = val_q_view(q);

          // integrate
          FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::do_integrate(
            eval, integrate_flag, data, u_view, grad_ref_q_view);

          // scatter
          if (data->precomputed_data->use_coloring)
            {
              for (int i = 0; i < n_local_dof; ++i)
                dst[local_to_global(i + c * n_local_dof, cell)] += u_view(i);
            }
          else
            {
              for (int i = 0; i < n_local_dof; ++i)
                Kokkos::atomic_add(
                  &dst[local_to_global(i + c * n_local_dof, cell)], u_view(i));
            }
        }
    }



    /**
     * This struct performs the evaluation of function values and gradients for
     * tensor-product finite elements. This is a specialization for elements
     * where the nodal points coincide with the quadrature points like FE_Q
     * shape functions on Gauss-Lobatto elements integrated with Gauss-Lobatto
     * quadrature. The assumption of this class is that the shape 'values'
     * operation is identity, which allows us to write shorter code.
     *
     * In literature, this form of evaluation is often called spectral
     * evaluation, spectral collocation or simply collocation, meaning the same
     * location for shape functions and evaluation space (quadrature points).
     */
    template <int dim, int fe_degree, typename Number>
    struct FEEvaluationImplCollocation
    {
      using SharedView = Kokkos::View<Number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      DEAL_II_HOST_DEVICE static void
      evaluate(const unsigned int                            n_components,
               const EvaluationFlags::EvaluationFlags        evaluation_flag,
               const typename MatrixFree<dim, Number>::Data *data)
      {
        // since the dof values have already been stored in
        // shared_data->values, there is nothing to do if the gradients are
        // not required
        if (!(evaluation_flag & EvaluationFlags::gradients))
          return;

        constexpr int n_points = Utilities::pow(fe_degree + 1, dim);
        Number        scratch_for_eval_storage[n_points];
        SharedView    scratch_for_eval;
        if (data->thread_per_cell)
          scratch_for_eval = SharedView(scratch_for_eval_storage, n_points);
        else
          scratch_for_eval = Kokkos::subview(data->shared_data->scratch_pad,
                                             Kokkos::make_pair(0, n_points));

        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree + 1,
                               fe_degree + 1,
                               Number>
          eval(data->team_member,
               get_shape_values_view<dim, Number>(data),
               get_shape_gradients_view<dim, Number>(data),
               data->precomputed_data->co_shape_gradients,
               scratch_for_eval,
               data->thread_per_cell);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            auto u = Kokkos::subview(data->shared_data->values, Kokkos::ALL, c);
            auto grad_u = Kokkos::subview(data->shared_data->gradients,
                                          Kokkos::ALL,
                                          Kokkos::ALL,
                                          c);

            eval.template co_gradients<0, true, false, false>(
              u, Kokkos::subview(grad_u, Kokkos::ALL, 0));
            if constexpr (dim > 1)
              eval.template co_gradients<1, true, false, false>(
                u, Kokkos::subview(grad_u, Kokkos::ALL, 1));
            if constexpr (dim > 2)
              eval.template co_gradients<2, true, false, false>(
                u, Kokkos::subview(grad_u, Kokkos::ALL, 2));
          }
      }


      DEAL_II_HOST_DEVICE static void
      integrate(const unsigned int                            n_components,
                const EvaluationFlags::EvaluationFlags        integration_flag,
                const typename MatrixFree<dim, Number>::Data *data)
      {
        // since the quad values have already been stored in
        // shared_data->values, there is nothing to do if the gradients are
        // not required
        if (!(integration_flag & EvaluationFlags::gradients))
          return;

        constexpr int n_points = Utilities::pow(fe_degree + 1, dim);
        Number        scratch_for_eval_storage[n_points];
        SharedView    scratch_for_eval;
        if (data->thread_per_cell)
          scratch_for_eval = SharedView(scratch_for_eval_storage, n_points);
        else
          scratch_for_eval = Kokkos::subview(data->shared_data->scratch_pad,
                                             Kokkos::make_pair(0, n_points));

        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree + 1,
                               fe_degree + 1,
                               Number>
          eval(data->team_member,
               get_shape_values_view<dim, Number>(data),
               get_shape_gradients_view<dim, Number>(data),
               data->precomputed_data->co_shape_gradients,
               scratch_for_eval,
               data->thread_per_cell);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            auto u = Kokkos::subview(data->shared_data->values, Kokkos::ALL, c);
            auto grad_u = Kokkos::subview(data->shared_data->gradients,
                                          Kokkos::ALL,
                                          Kokkos::ALL,
                                          c);

            if constexpr (dim == 1)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<0, false, true, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
                else
                  eval.template co_gradients<2, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 2), u);
              }
            else if constexpr (dim == 2)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<1, false, true, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 1), u);
                else
                  eval.template co_gradients<1, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 1), u);
                eval.template co_gradients<0, false, true, false>(
                  Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
              }
            else if constexpr (dim == 3)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<2, false, true, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 2), u);
                else
                  eval.template co_gradients<2, false, false, false>(
                    Kokkos::subview(grad_u, Kokkos::ALL, 2), u);
                eval.template co_gradients<1, false, true, false>(
                  Kokkos::subview(grad_u, Kokkos::ALL, 1), u);
                eval.template co_gradients<0, false, true, false>(
                  Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
              }
            else
              Assert(false, ExcMessage("dim must not exceed 3!"));
          }
      }
    };



    /**
     * This struct performs the evaluation of function values and gradients for
     * tensor-product finite elements. This is a specialization for symmetric
     * basis functions about the mid point 0.5 of the unit interval with the
     * same number of quadrature points as degrees of freedom. In that case, we
     * can first transform the basis to one that has the nodal points in the
     * quadrature points (i.e., the collocation space) and then perform the
     * evaluation of the first and second derivatives in this transformed space,
     * using the identity operation for the shape values.
     */
    template <int dim, int fe_degree, int n_q_points_1d, typename Number>
    struct FEEvaluationImplTransformToCollocation
    {
      using SharedView = Kokkos::View<Number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      DEAL_II_HOST_DEVICE static void
      evaluate(const unsigned int                            n_components,
               const EvaluationFlags::EvaluationFlags        evaluation_flag,
               const typename MatrixFree<dim, Number>::Data *data)
      {
        constexpr int scratch_size = Utilities::pow(n_q_points_1d, dim);
        Number        scratch_for_eval_storage[scratch_size];
        SharedView    scratch_for_eval;
        if (data->thread_per_cell)
          scratch_for_eval = SharedView(scratch_for_eval_storage, scratch_size);
        else
          scratch_for_eval =
            Kokkos::subview(data->shared_data->scratch_pad,
                            Kokkos::make_pair(0, scratch_size));

        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree + 1,
                               n_q_points_1d,
                               Number>
          eval(data->team_member,
               get_shape_values_view<dim, Number>(data),
               get_shape_gradients_view<dim, Number>(data),
               data->precomputed_data->co_shape_gradients,
               scratch_for_eval,
               data->thread_per_cell);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            auto u = Kokkos::subview(data->shared_data->values, Kokkos::ALL, c);
            auto grad_u = Kokkos::subview(data->shared_data->gradients,
                                          Kokkos::ALL,
                                          Kokkos::ALL,
                                          c);

            eval.template values<0, true, false, true>(u, u);
            if constexpr (dim > 1)
              eval.template values<1, true, false, true>(u, u);
            if constexpr (dim > 2)
              eval.template values<2, true, false, true>(u, u);

            if (evaluation_flag & EvaluationFlags::gradients)
              {
                eval.template co_gradients<0, true, false, false>(
                  u, Kokkos::subview(grad_u, Kokkos::ALL, 0));
                if constexpr (dim > 1)
                  eval.template co_gradients<1, true, false, false>(
                    u, Kokkos::subview(grad_u, Kokkos::ALL, 1));
                if constexpr (dim > 2)
                  eval.template co_gradients<2, true, false, false>(
                    u, Kokkos::subview(grad_u, Kokkos::ALL, 2));
              }
          }
      }


      DEAL_II_HOST_DEVICE static void
      integrate(const unsigned int                            n_components,
                const EvaluationFlags::EvaluationFlags        integration_flag,
                const typename MatrixFree<dim, Number>::Data *data)
      {
        constexpr int scratch_size = Utilities::pow(n_q_points_1d, dim);
        Number        scratch_for_eval_storage[scratch_size];
        SharedView    scratch_for_eval;
        if (data->thread_per_cell)
          scratch_for_eval = SharedView(scratch_for_eval_storage, scratch_size);
        else
          scratch_for_eval =
            Kokkos::subview(data->shared_data->scratch_pad,
                            Kokkos::make_pair(0, scratch_size));

        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree + 1,
                               n_q_points_1d,
                               Number>
          eval(data->team_member,
               get_shape_values_view<dim, Number>(data),
               get_shape_gradients_view<dim, Number>(data),
               data->precomputed_data->co_shape_gradients,
               scratch_for_eval,
               data->thread_per_cell);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            auto u = Kokkos::subview(data->shared_data->values, Kokkos::ALL, c);
            auto grad_u = Kokkos::subview(data->shared_data->gradients,
                                          Kokkos::ALL,
                                          Kokkos::ALL,
                                          c);

            // apply derivatives in collocation space
            if (integration_flag & EvaluationFlags::gradients)
              {
                if constexpr (dim == 1)
                  {
                    if (integration_flag & EvaluationFlags::values)
                      eval.template co_gradients<0, false, true, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
                    else
                      eval.template co_gradients<2, false, false, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 2), u);
                  }
                else if constexpr (dim == 2)
                  {
                    if (integration_flag & EvaluationFlags::values)
                      eval.template co_gradients<1, false, true, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 1), u);
                    else
                      eval.template co_gradients<1, false, false, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 1), u);
                    eval.template co_gradients<0, false, true, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
                  }
                else if constexpr (dim == 3)
                  {
                    if (integration_flag & EvaluationFlags::values)
                      eval.template co_gradients<2, false, true, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 2), u);
                    else
                      eval.template co_gradients<2, false, false, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 2), u);
                    eval.template co_gradients<1, false, true, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 1), u);
                    eval.template co_gradients<0, false, true, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
                  }
                else
                  Assert(false, ExcMessage("dim must not exceed 3!"));
              }

            // transform back to the original space
            if constexpr (dim > 2)
              eval.template values<2, false, false, true>(u, u);
            if constexpr (dim > 1)
              eval.template values<1, false, false, true>(u, u);
            eval.template values<0, false, false, true>(u, u);
          }
      }
    };
  } // end of namespace internal
} // end of namespace Portable


DEAL_II_NAMESPACE_CLOSE

#endif
