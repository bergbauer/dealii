#include <deal.II/base/bounding_box.h>

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

#include <deal.II/non_matching/restriction.h>

DEAL_II_NAMESPACE_OPEN
namespace NonMatching
{
  template <int dim>
  FunctionUnitToReal<dim>::FunctionUnitToReal(const Function<dim> &function)
    : function(&function)
  {
    cell_index = numbers::invalid_unsigned_int;
    cell_level = numbers::invalid_unsigned_int;
  }



  template <int dim>
  void
  FunctionUnitToReal<dim>::set_active_cell(
    const typename Triangulation<dim>::active_cell_iterator &cell)
  {
    cell_level    = cell->level();
    cell_index    = cell->index();
    triangulation = &(cell->get_triangulation());
  }



  template <int dim>
  double
  FunctionUnitToReal<dim>::value(const Point<dim> & point,
                                 const unsigned int component) const
  {
    Assert(component == 0, ExcInternalError());
    (void)component;

    typename Triangulation<dim>::active_cell_iterator cell(triangulation,
                                                           cell_level,
                                                           cell_index);

    FEValues<dim> fe_values(mapping,
                            dummy_element,
                            Quadrature<dim>(point),
                            update_quadrature_points);
    fe_values.reinit(cell);

    const Point<dim> &point_in_real_space = fe_values.quadrature_point(0);

    return function->value(point_in_real_space);
  }



  template <int dim>
  Tensor<1, dim>
  FunctionUnitToReal<dim>::gradient(const Point<dim> & point,
                                    const unsigned int component) const
  {
    Assert(component == 0, ExcInternalError());
    (void)component;

    typename Triangulation<dim>::active_cell_iterator cell(triangulation,
                                                           cell_level,
                                                           cell_index);

    const UpdateFlags update_flags =
      update_jacobians | update_quadrature_points;

    FEValues<dim> fe_values(mapping,
                            dummy_element,
                            Quadrature<dim>(point),
                            update_flags);
    fe_values.reinit(cell);

    const Point<dim> &point_in_real_space = fe_values.quadrature_point(0);

    const Tensor<2, dim> &jacobian = fe_values.jacobian(0);

    const Tensor<1, dim> real_gradient =
      function->gradient(point_in_real_space);

    return contract<0, 0>(jacobian, real_gradient);
  }



  template <int dim>
  SymmetricTensor<2, dim>
  FunctionUnitToReal<dim>::hessian(const Point<dim> & point,
                                   const unsigned int component) const
  {
    Assert(component == 0, ExcInternalError());
    (void)component;

    typename Triangulation<dim>::active_cell_iterator cell(triangulation,
                                                           cell_level,
                                                           cell_index);

    const UpdateFlags update_flags =
      update_jacobians | update_jacobian_grads | update_quadrature_points;
    FEValues<dim> fe_values(mapping,
                            dummy_element,
                            Quadrature<dim>(point),
                            update_flags);
    fe_values.reinit(cell);

    const Point<dim> &point_in_real_space = fe_values.quadrature_point(0);

    const Tensor<2, dim> &jacobian      = fe_values.jacobian(0);
    const Tensor<3, dim> &jacobian_grad = fe_values.jacobian_grad(0);

    const Tensor<1, dim> real_gradient =
      function->gradient(point_in_real_space);
    const SymmetricTensor<2, dim> real_hessian =
      function->hessian(point_in_real_space);

    Tensor<2, dim> ref_space_hessian =
      contract<0, 0>(jacobian, real_hessian * jacobian);
    ref_space_hessian += contract<1, 0>(jacobian_grad, real_gradient);

    // The result should already be symmetric up to floating point.
    return symmetrize(ref_space_hessian);
  }

} // namespace NonMatching
#include "restriction.inst"
DEAL_II_NAMESPACE_CLOSE
