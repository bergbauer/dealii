/*
 * Restriction.h
 *
 *  Created on: Jul 1, 2015
 *      Author: simon
 */
#ifndef __deal2___restriction
#define __deal2___restriction

//#include <boost/math/tools/tuple.hpp>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <boost/math/special_functions.hpp>

#include <memory>
#include <vector>



DEAL_II_NAMESPACE_OPEN
namespace NonMatching
{
  template <int dim>
  class FunctionUnitToReal : public Function<dim>
  {
  public:
    FunctionUnitToReal(const Function<dim> &function);

    void
    set_active_cell(
      const typename Triangulation<dim>::active_cell_iterator &cell);

    double
    value(const Point<dim> & point,
          const unsigned int component = 0) const override;

    Tensor<1, dim>
    gradient(const Point<dim> & point,
             const unsigned int component = 0) const override;

    SymmetricTensor<2, dim>
    hessian(const Point<dim> & point,
            const unsigned int component = 0) const override;

  private:
    const SmartPointer<const Function<dim>> function;

    unsigned int cell_level, cell_index;

    SmartPointer<const Triangulation<dim>> triangulation;
    const MappingQ1<dim>                   mapping;

    /**
     * We use a FEValues object to compute the Jacobian and its derivatives. To
     * use it we need an element.
     */
    const FE_Nothing<dim> dummy_element;
  };

} // namespace NonMatching
DEAL_II_NAMESPACE_CLOSE

#endif
