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

#include <deal.II/base/function_restriction.h>
#include <deal.II/base/function_tools.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <boost/math/special_functions/sign.hpp>
#include <boost/math/tools/roots.hpp>

#include <algorithm>
#include <vector>

DEAL_II_NAMESPACE_OPEN
namespace NonMatching
{
  namespace internal
  {
    namespace QuadratureGeneratorImplementation
    {
      template <int dim>
      void
      quadrature_from_point(const Quadrature<1> &      quadrature1D,
                            const Point<dim - 1> &     point,
                            const double               weight,
                            const unsigned int         direction_index,
                            const double               start,
                            const double               end,
                            ExtendableQuadrature<dim> &quadrature)
      {
        const double length = end - start;
        for (unsigned int j = 0; j < quadrature1D.size(); ++j)
          {
            const double x = start + (end - start) * quadrature1D.point(j)[0];
            quadrature.push_back(dealii::internal::create_higher_dim_point(
                                   point, direction_index, x),
                                 length * weight * quadrature1D.weight(j));
          }
      }



      /**
       * For each (point, weight) in lower create a dim-dimensional quadrature
       * using quadrature_from_point and add the results to @p quadrature.
       */
      template <int dim>
      void
      create_by_tensor(const Quadrature<1> &      quadrature1D,
                       const double               start,
                       const double               end,
                       const Quadrature<dim - 1> &lower,
                       const unsigned int         direction_index,
                       ExtendableQuadrature<dim> &quadrature)
      {
        Assert(start < end,
               ExcMessage("Interval start must be less than interval end."));

        for (unsigned int j = 0; j < lower.size(); ++j)
          {
            quadrature_from_point(quadrature1D,
                                  lower.point(j),
                                  lower.weight(j),
                                  direction_index,
                                  start,
                                  end,
                                  quadrature);
          }
      }



      template <int dim>
      Definiteness
      pointwise_definiteness(
        const std::vector<std::shared_ptr<Function<dim>>> &functions,
        const Point<dim> &                                 point)
      {
        Assert(functions.size() > 0,
               ExcMessage(
                 "The incoming vector must contain at least one function."));

        const int sign_of_first = boost::math::sign(functions[0]->value(point));

        if (sign_of_first == 0)
          return Definiteness::INDEFINITE;

        for (unsigned int j = 1; j < functions.size(); ++j)
          {
            const int sign = boost::math::sign(functions[j]->value(point));

            if (sign != sign_of_first)
              return Definiteness::INDEFINITE;
          }
        // If we got here all functions have the same sign.
        if (sign_of_first < 0)
          return Definiteness::NEGATIVE;
        else
          return Definiteness::POSITIVE;
      }



      template <int dim>
      void
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


      /**
       * Estimate bounds on each of the Functions in the incoming vector over
       * the incoming box.
       *
       * Bounds on the functions value and the gradient components are first
       * computed using FunctionTools::taylor_estimate_function_bounds.
       * In addition, the function value is checked for min/max at the at
       * the vertices of the box. The gradient is not checked at the box
       * vertices.
       */
      template <int dim>
      void
      estimate_function_bounds(
        const std::vector<std::shared_ptr<Function<dim>>> &functions,
        const BoundingBox<dim> &                           box,
        std::vector<FunctionBounds<dim>> &                 all_function_bounds)
      {
        all_function_bounds.clear();
        all_function_bounds.reserve(functions.size());
        for (unsigned int i = 0; i < functions.size(); ++i)
          {
            const Function<dim> &function = *(functions.at(i));

            FunctionBounds<dim> bounds;
            FunctionTools::taylor_estimate_function_bounds<dim>(
              function, box, bounds.value, bounds.gradient);
            take_min_max_at_vertices(function, box, bounds.value);

            all_function_bounds.push_back(bounds);
          }
      }



      template <int dim>
      std::pair<double, double>
      find_extreme_values(const std::vector<FunctionBounds<dim>> &bounds)
      {
        Assert(bounds.size() > 0, ExcMessage("The incoming vector is empty."));

        std::pair<double, double> extremes;

        extremes.first  = bounds[0].value.first;
        extremes.second = bounds[0].value.second;
        for (unsigned int i = 1; i < bounds.size(); ++i)
          {
            extremes.first  = std::min(extremes.first, bounds[i].value.first);
            extremes.second = std::max(extremes.second, bounds[i].value.second);
          }

        return extremes;
      }



      /*
       * Return true if the incoming function bounds correspond to a function
       * which is indefinite, i.e., that is not negative or positive definite.
       */
      inline bool
      is_indefinite(const std::pair<double, double> &function_bounds)
      {
        if (function_bounds.first > 0)
          return false;
        if (function_bounds.second < 0)
          return false;
        return true;
      }



      /*
       * Return a lower bound, $L_a$, on the absolute value of a function,
       * $f(x)$:
       *
       * $L_a \leq |f(x)|$,
       *
       * by estimating it from the incoming lower and upper bounds:
       * $L \leq f(x) \leq U$.
       *
       * By rewriting the lower and upper bounds as
       * $F - C \leq f(x) \leq F + C$,
       * where $L = F - C$, $U = F + C$ (or $F = (U + L)/2$, $C = (U - L)/2$),
       * we get $|f(x) - F| \leq C$.
       * Using the inverse triangle inequality gives
       * $|F| - |f(x)| \leq |f(x) - F| \leq C$.
       * Thus, $L_a = |F| - C$.
       * If this is negative, $L_a$ is set to 0, since the absolute value can't
       * be negative.
       */
      inline double
      lower_bound_on_abs(const std::pair<double, double> &function_bounds)
      {
        Assert(function_bounds.first <= function_bounds.second,
               ExcMessage("Function bounds reversed, max < min."));

        const double estimate =
          0.5 * (std::abs(function_bounds.second + function_bounds.first) -
                 (function_bounds.second - function_bounds.first));

        // |f| Can't be negative.
        return std::max(0.0, estimate);
      }



      template <int dim>
      std_cxx17::optional<Tensor<1, dim>>
      min_of_all_min_abs_grad(
        const std::vector<FunctionBounds<dim>> &all_function_bounds)
      {
        Assert(all_function_bounds.size() > 0,
               ExcMessage("The incoming vector is empty."));

        std_cxx17::optional<Tensor<1, dim>> min_existance_distances;

        for (unsigned int i = 0; i < all_function_bounds.size(); ++i)
          {
            const FunctionBounds<dim> &bounds = all_function_bounds.at(i);

            // If a restriction is strictly positive or negative, there is no
            // need to consider it when we determine a height direction. Note
            // that this check wouldn't be needed if we pruned before we tried
            // to find a height direction.
            if (is_indefinite(bounds.value))
              {
                if (!min_existance_distances)
                  {
                    min_existance_distances.emplace();
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        (*min_existance_distances)[d] =
                          lower_bound_on_abs(bounds.gradient[d]);
                      }
                  }
                else
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        (*min_existance_distances)[d] =
                          std::min((*min_existance_distances)[d],
                                   lower_bound_on_abs(bounds.gradient[d]));
                      }
                  }
              }
          }

        return min_existance_distances;
      }



      HeightDirectionData::HeightDirectionData()
      {
        direction    = numbers::invalid_unsigned_int;
        min_abs_dfdx = 0;
      }



      template <int dim>
      std_cxx17::optional<HeightDirectionData>
      find_best_height_direction(
        const std::vector<FunctionBounds<dim>> &all_function_bounds)
      {
        std_cxx17::optional<HeightDirectionData> data;

        const std_cxx17::optional<Tensor<1, dim>> min_abs_grad =
          min_of_all_min_abs_grad(all_function_bounds);

        if (min_abs_grad)
          {
            data.emplace();

            const double *max_element =
              std::max_element(min_abs_grad->begin_raw(),
                               min_abs_grad->end_raw());

            data->direction    = max_element - min_abs_grad->begin_raw();
            data->min_abs_dfdx = *max_element;
          }

        return data;
      }



      /*
       * Return true if there are exactly two incoming FunctionBounds and
       * they corresponds to one function being positive definite and
       * one being negative definite. Return false otherwise.
       */
      template <int dim>
      inline bool
      one_positive_one_negative_definite(
        const std::vector<FunctionBounds<dim>> &all_function_bounds)
      {
        if (all_function_bounds.size() != 2)
          return false;
        else
          {
            const FunctionBounds<dim> &bounds0 = all_function_bounds.at(0);
            const FunctionBounds<dim> &bounds1 = all_function_bounds.at(1);

            if (bounds0.value.first > 0 && bounds1.value.second < 0)
              return true;
            if (bounds1.value.first > 0 && bounds0.value.second < 0)
              return true;
            return false;
          }
      }



      /*
       * Transform the points and weights of the incoming quadrature,
       * unit_quadrature, from unit space to the incoming box and add these to
       * quadrature.
       *
       * Note that unit_quadrature should be a quadrature over [0,1]^dim.
       */
      template <int dim>
      void
      map_quadrature_to_box(const Quadrature<dim> &    unit_quadrature,
                            const BoundingBox<dim> &   box,
                            ExtendableQuadrature<dim> &quadrature)
      {
        for (unsigned int i = 0; i < unit_quadrature.size(); i++)
          {
            const Point<dim> point = box.unit_to_real(unit_quadrature.point(i));
            const double     weight = unit_quadrature.weight(i) * box.volume();

            quadrature.push_back(point, weight);
          }
      }



      /**
       * For each of the incoming dim-dimensional functions, create the
       * restriction to the top and bottom of the incoming BoundingBox and add
       * these two
       * (dim-1)-dimensional functions to @param restrictions. Here, top and bottom is
       * meant with respect to the incoming @param direction. For each function, the
       * "bottom-restriction" will be added before the "top-restriction"
       *
       * @note @param restrictions will be cleared, so after this function
       * restrictions.size() == 2 * functions.size().
       */
      template <int dim>
      void
      restrict_to_top_and_bottom(
        const std::vector<std::shared_ptr<Function<dim>>> &functions,
        const BoundingBox<dim> &                           box,
        const unsigned int                                 direction,
        std::vector<std::shared_ptr<Function<dim - 1>>> &  restrictions)
      {
        AssertIndexRange(direction, dim);

        restrictions.clear();
        const double bottom = box.lower_bound(direction);
        const double top    = box.upper_bound(direction);

        for (const auto &function : functions)
          {
            restrictions.push_back(std::shared_ptr<Function<dim - 1>>(
              new Functions::CoordinateRestriction<dim - 1>(*function,
                                                            direction,
                                                            bottom)));
            restrictions.push_back(std::shared_ptr<Function<dim - 1>>(
              new Functions::CoordinateRestriction<dim - 1>(*function,
                                                            direction,
                                                            top)));
          }
      }



      /**
       * Restrict each of the incoming @param functions to @param point,
       * while keeping the coordinate direction @param open_direction open,
       * and add the restriction to @param restrictions.
       *
       * @note @param restrictions will be cleared, so after this function
       * restrictions.size() == functions.size().
       */
      template <int dim>
      void
      restrict_to_point(
        const std::vector<std::shared_ptr<Function<dim>>> &functions,
        const Point<dim - 1> &                             point,
        const unsigned int                                 open_direction,
        std::vector<std::shared_ptr<Function<1>>> &        restrictions)
      {
        AssertIndexRange(open_direction, dim);

        restrictions.clear();
        restrictions.reserve(functions.size());
        for (const auto &function : functions)
          {
            restrictions.push_back(std::shared_ptr<Function<1>>(
              new Functions::PointRestriction<dim - 1>(*function,
                                                       open_direction,
                                                       point)));
          }
      }



      template <int dim>
      void
      distribute_points_between_roots(
        const Quadrature<1> &      quadrature1D,
        const BoundingBox<1> &     box,
        const std::vector<double> &roots,
        const Point<dim - 1> &     point,
        const double               weight,
        const unsigned int         height_function_direction,
        const std::vector<std::shared_ptr<Function<1>>> &restrictions1D,
        const AdditionalQGeneratorData &                 additional_data,
        ImmersedQuadratures<dim> &                       immersed_quadratures)
      {
        // Make this int to avoid a warning signed/unsigned comparision.
        const int n_roots = roots.size();

        // The number of intervals are roots.size() + 1
        for (int i = -1; i < n_roots; ++i)
          {
            // Start and end point of the interval.
            const double start = i < 0 ? box.lower_bound(0) : roots[i];
            const double end =
              i + 1 < n_roots ? roots[i + 1] : box.upper_bound(0);

            const double length = end - start;
            // It might be that the end points of the box are roots.
            // If this is the case then the interval has length zero.
            // Don't distribute points on the interval if it is shorter than
            // some tolerance.
            if (length > additional_data.min_distance_between_roots)
              {
                const Point<1> center(start + 0.5 * length);
                // Determine what type of interval we are dealing with.
                const Definiteness definiteness =
                  pointwise_definiteness(restrictions1D, center);
                ExtendableQuadrature<dim> &target_quad =
                  immersed_quadratures.quadrature_by_definiteness(definiteness);
                quadrature_from_point(quadrature1D,
                                      point,
                                      weight,
                                      height_function_direction,
                                      start,
                                      end,
                                      target_quad);
              }
          }
      }



      RootFinder::AdditionalData::AdditionalData(
        const double       tolerance,
        const unsigned int max_recursion_depth,
        const unsigned int max_iterations)
        : tolerance(tolerance)
        , max_recursion_depth(max_recursion_depth)
        , max_iterations(max_iterations)
      {}



      RootFinder::RootFinder(const AdditionalData &data)
        : additional_data(data)
      {}



      void
      RootFinder::find_roots(
        const std::vector<std::shared_ptr<Function<1>>> &functions,
        const BoundingBox<1> &                           interval,
        std::vector<double> &                            roots)
      {
        for (unsigned int j = 0; j < functions.size(); ++j)
          {
            const unsigned int recursion_depth = 0;
            find_roots(*functions.at(j), interval, recursion_depth, roots);
          }
        // Sort and make sure no roots are duplicated
        std::sort(roots.begin(), roots.end());

        const auto roots_are_equal = [this](const double &a, const double &b) {
          return std::abs(a - b) < additional_data.tolerance;
        };
        roots.erase(unique(roots.begin(), roots.end(), roots_are_equal),
                    roots.end());
      }



      void
      RootFinder::find_roots(const Function<1> &   function,
                             const BoundingBox<1> &interval,
                             const unsigned int    recursion_depth,
                             std::vector<double> & roots)
      {
        // Compute function values at end points.
        const double left_value  = function.value(interval.vertex(0));
        const double right_value = function.value(interval.vertex(1));

        // If we have a sign change we solve for the root.
        if (boost::math::sign(left_value) != boost::math::sign(right_value))
          {
            const auto lambda = [&function](const double x) {
              return function.value(Point<1>(x));
            };

            const auto stopping_criteria = [this](const double &a,
                                                  const double &b) {
              return std::abs(a - b) < additional_data.tolerance;
            };

            boost::uintmax_t iterations = additional_data.max_iterations;

            const std::pair<double, double> root_bracket =
              boost::math::tools::toms748_solve(lambda,
                                                interval.lower_bound(0),
                                                interval.upper_bound(0),
                                                left_value,
                                                right_value,
                                                stopping_criteria,
                                                iterations);

            const double root = .5 * (root_bracket.first + root_bracket.second);
            roots.push_back(root);
          }
        else
          {
            // Compute bounds on the incoming function to check if there are
            // roots. If the function is positive or negative on the whole
            // interval we do nothing.
            std::pair<double, double>                value_bounds;
            std::array<std::pair<double, double>, 1> gradient_bounds;
            FunctionTools::taylor_estimate_function_bounds<1>(function,
                                                              interval,
                                                              value_bounds,
                                                              gradient_bounds);

            // Since we already know the function values at the interval ends we
            // might as well check these for min/max too.
            const double function_min =
              std::min(std::min(left_value, right_value), value_bounds.first);

            // If the functions is positive there are no roots.
            if (function_min > 0)
              return;

            const double function_max =
              std::max(std::max(left_value, right_value), value_bounds.second);

            // If the functions is negative there are no roots.
            if (function_max < 0)
              return;

            // If we can't say that the function is strictly positive/negative
            // we split the interval if half. We can't split forever, so if we
            // have reached the max recursion, we stop looking for roots.
            if (recursion_depth < additional_data.max_recursion_depth)
              {
                find_roots(function,
                           interval.child(0),
                           recursion_depth + 1,
                           roots);
                find_roots(function,
                           interval.child(1),
                           recursion_depth + 1,
                           roots);
              }
          }
      }



      template <int dim>
      ExtendableQuadrature<dim>::ExtendableQuadrature(
        const Quadrature<dim> &quadrature)
        : Quadrature<dim>(quadrature)
      {}



      template <int dim>
      void
      ExtendableQuadrature<dim>::push_back(const Point<dim> &point,
                                           const double      weight)
      {
        this->quadrature_points.push_back(point);
        this->weights.push_back(weight);
      }



      template <int dim>
      ExtendableQuadrature<dim> &
      ImmersedQuadratures<dim>::quadrature_by_definiteness(
        const Definiteness definiteness)
      {
        switch (definiteness)
          {
            case Definiteness::NEGATIVE:
              return inside;
            case Definiteness::POSITIVE:
              return outside;
            default:
              return indefinite;
          }
      }



      /**
       * Takes a (dim-1)-dimensional point from the cross-section (orthogonal
       * to @param direction) of the @param box. Creates the two
       * dim-dimensional points, which are the projections from the cross
       * section to the faces of the box and returns the point closest to the
       * zero-contour of the incoming level set function.
       */
      template <int dim>
      Point<dim>
      face_projection_closest_zero_contour(const Point<dim - 1> &  point,
                                           const unsigned int      direction,
                                           const BoundingBox<dim> &box,
                                           const Function<dim> &   level_set)
      {
        const Point<dim> bottom_point =
          dealii::internal::create_higher_dim_point(point,
                                                    direction,
                                                    box.lower_bound(direction));
        const double bottom_value = level_set.value(bottom_point);

        const Point<dim> top_point =
          dealii::internal::create_higher_dim_point(point,
                                                    direction,
                                                    box.upper_bound(direction));
        const double top_value = level_set.value(top_point);

        // The end point closest to the zero-contour is the one with smallest
        // absolute value.
        return std::abs(bottom_value) < std::abs(top_value) ? bottom_point :
                                                              top_point;
      }



      template <int dim, int spacedim>
      UpThroughDimensionCreator<dim, spacedim>::UpThroughDimensionCreator(
        ImmersedQuadratures<dim> &      immersed_quadratures,
        const hp::QCollection<1> &      q_collection1D,
        const AdditionalQGeneratorData &additional_data)
        : immersed_quadratures(&immersed_quadratures)
        , q_collection1D(&q_collection1D)
        , additional_data(additional_data)
        , root_finder(
            RootFinder::AdditionalData(additional_data.root_finder_tolerance,
                                       additional_data.n_allowed_splits))
      {
        height_function_direction = numbers::invalid_unsigned_int;

        q_index = 0;
      }



      template <int dim, int spacedim>
      void
      UpThroughDimensionCreator<dim, spacedim>::create_quadrature(
        const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
        const BoundingBox<dim> &                           box,
        const Quadrature<dim - 1> &                        low_dim_quadrature,
        const unsigned int height_function_direction)
      {
        this->height_function_direction = height_function_direction;

        const Quadrature<1> &quadrature1D = (*q_collection1D)[q_index];
        for (unsigned int q = 0; q < low_dim_quadrature.size(); ++q)
          {
            const Point<dim - 1> &point  = low_dim_quadrature.point(q);
            const double          weight = low_dim_quadrature.weight(q);
            restrict_to_point(level_sets,
                              point,
                              height_function_direction,
                              restrictions1D);

            const BoundingBox<1> bounds_in_direction =
              box.bounds(height_function_direction);

            roots.clear();
            root_finder.find_roots(restrictions1D, bounds_in_direction, roots);

            distribute_points_between_roots(quadrature1D,
                                            bounds_in_direction,
                                            roots,
                                            point,
                                            weight,
                                            height_function_direction,
                                            restrictions1D,
                                            additional_data,
                                            *immersed_quadratures);

            if (dim == spacedim)
              create_surface_point(point, weight, level_sets, box);
          }

        restrictions1D.clear();
      }



      template <int dim, int spacedim>
      void
      UpThroughDimensionCreator<dim, spacedim>::create_surface_point(
        const Point<dim - 1> &                             point,
        const double                                       weight,
        const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
        const BoundingBox<dim> &                           box)
      {
        AssertIndexRange(roots.size(), 2);
        Assert(level_sets.size() == 1, ExcInternalError());


        const Function<dim> &level_set = *level_sets.at(0);

        Point<dim> surface_point;
        if (roots.size() == 1)
          {
            surface_point = dealii::internal::create_higher_dim_point(
              point, height_function_direction, roots[0]);
          }
        else
          {
            // If we got here we have missed roots of the level set functions
            // when we created the lower dimensional quadratures. This can
            // happen if the level set function oscillates rapidly. The best
            // thing we can do is to choose the surface point as the end point
            // closest to the zero contour.
            surface_point = face_projection_closest_zero_contour(
              point, height_function_direction, box, level_set);
          }

        const Tensor<1, dim> gradient = level_set.gradient(surface_point);
        Tensor<1, dim>       normal   = gradient;
        normal *= 1. / normal.norm();

        const double surface_weight =
          weight * gradient.norm() /
          std::abs(gradient[height_function_direction]);
        immersed_quadratures->surface.push_back(surface_point,
                                                surface_weight,
                                                normal);
      }



      template <int dim, int spacedim>
      void
      UpThroughDimensionCreator<dim, spacedim>::set_1D_quadrature(
        unsigned int q_index)
      {
        AssertIndexRange(q_index, q_collection1D->size());
        this->q_index = q_index;
      }



      template <int dim, int spacedim>
      QGenerator<dim, spacedim>::QGenerator(
        const hp::QCollection<1> &      q_collection1D,
        const AdditionalQGeneratorData &additional_data)
        : additional_data(additional_data)
        , low_dim_algorithm(q_collection1D, additional_data)
        , up_through_dimension_creator(this->immersed_quadratures,
                                       q_collection1D,
                                       additional_data)
        , q_collection1D(&q_collection1D)
      {
        Assert(q_collection1D.size() > 0,
               ExcMessage("Incoming quadrature collection is empty."));

        q_index = 0;

        for (unsigned int i = 0; i < q_collection1D.size(); i++)
          tensor_products.push_back(Quadrature<dim>(q_collection1D[i]));
      }



      template <int dim, int spacedim>
      void
      QGeneratorBase<dim, spacedim>::clear_quadratures()
      {
        immersed_quadratures = ImmersedQuadratures<dim>();
      }



      template <int dim, int spacedim>
      void
      QGenerator<dim, spacedim>::clear_quadratures()
      {
        this->immersed_quadratures = ImmersedQuadratures<dim>();
        low_dim_algorithm.clear_quadratures();
      }



      template <int dim, int spacedim>
      const ImmersedQuadratures<dim> &
      QGeneratorBase<dim, spacedim>::get_quadratures() const
      {
        return immersed_quadratures;
      }



      template <int dim, int spacedim>
      bool
      QGenerator<dim, spacedim>::allowed_to_split_cube(
        const BoundingBox<dim> &box) const
      {
        // Each time, i, the box is split it's side-length is 1 / 2^i
        const double smallest_allowed_side_length =
          1. / std::pow(2, additional_data.n_allowed_splits);

        // We know that the bounding box is a hypercube so the side-length is
        // the same in all directions.
        return box.side_length(0) > smallest_allowed_side_length;
      }



      template <int dim, int spacedim>
      void
      QGenerator<dim, spacedim>::create_quadrature(
        const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
        const BoundingBox<dim> &                           box)
      {
        std::vector<FunctionBounds<dim>> all_function_bounds;
        estimate_function_bounds(level_sets, box, all_function_bounds);

        const std::pair<double, double> extreme_values =
          find_extreme_values(all_function_bounds);

        if (extreme_values.first > additional_data.limit_to_be_definite)
          {
            map_quadrature_to_box(tensor_products[q_index],
                                  box,
                                  this->immersed_quadratures.outside);
          }
        else if (extreme_values.second < -additional_data.limit_to_be_definite)
          {
            map_quadrature_to_box(tensor_products[q_index],
                                  box,
                                  this->immersed_quadratures.inside);
          }
        else if (one_positive_one_negative_definite(all_function_bounds))
          {
            map_quadrature_to_box(tensor_products[q_index],
                                  box,
                                  this->immersed_quadratures.indefinite);
          }
        else
          {
            const std_cxx17::optional<HeightDirectionData> data =
              find_best_height_direction(all_function_bounds);

            // Check larger than a constant to avoid that min_abs_dfdx is only
            // larger by 0 by floating point precision.
            if (data && data->min_abs_dfdx >
                          additional_data.lower_bound_implicit_function)
              {
                create_low_dim_quadratures(data->direction, level_sets, box);
                create_high_dim_quadratures(data->direction, level_sets, box);
              }
            else if (allowed_to_split_cube(box))
              {
                split_box_and_recurse(level_sets, box);
              }
            else
              {
                // We can't split the box recursively forever. Use the midpoint
                // method as a last resort.
                use_midpoint_method(level_sets, box);
              }
          }
      }



      template <int dim, int spacedim>
      void
      QGenerator<dim, spacedim>::split_box_and_recurse(
        const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
        const BoundingBox<dim> &                           box)
      {
        for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell;
             ++i)
          {
            this->create_quadrature(level_sets, box.child(i));
          }
      }



      template <int dim, int spacedim>
      void
      QGenerator<dim, spacedim>::create_low_dim_quadratures(
        const unsigned int height_function_direction,
        const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
        const BoundingBox<dim> &                           box)
      {
        std::vector<std::shared_ptr<Function<dim - 1>>> restrictions;
        restrictions.reserve(2 * level_sets.size());

        restrict_to_top_and_bottom(level_sets,
                                   box,
                                   height_function_direction,
                                   restrictions);

        const BoundingBox<dim - 1> cross_section =
          box.cross_section(height_function_direction);

        low_dim_algorithm.clear_quadratures();
        low_dim_algorithm.create_quadrature(restrictions, cross_section);
      }



      template <int dim, int spacedim>
      void
      QGenerator<dim, spacedim>::create_high_dim_quadratures(
        const unsigned int height_function_direction,
        const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
        const BoundingBox<dim> &                           box)
      {
        const ImmersedQuadratures<dim - 1> &low_dim_quadratures =
          low_dim_algorithm.get_quadratures();

        const Quadrature<1> &quadrature1D = (*q_collection1D)[q_index];

        create_by_tensor(quadrature1D,
                         box.lower_bound(height_function_direction),
                         box.upper_bound(height_function_direction),
                         low_dim_quadratures.inside,
                         height_function_direction,
                         this->immersed_quadratures.inside);

        create_by_tensor(quadrature1D,
                         box.lower_bound(height_function_direction),
                         box.upper_bound(height_function_direction),
                         low_dim_quadratures.outside,
                         height_function_direction,
                         this->immersed_quadratures.outside);

        up_through_dimension_creator.create_quadrature(
          level_sets,
          box,
          low_dim_quadratures.indefinite,
          height_function_direction);
      }



      template <int dim, int spacedim>
      void
      QGenerator<dim, spacedim>::use_midpoint_method(
        const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
        const BoundingBox<dim> &                           box)
      {
        const Point<dim>   center = box.center();
        const Definiteness definiteness =
          pointwise_definiteness(level_sets, center);

        ExtendableQuadrature<dim> &quadrature =
          this->immersed_quadratures.quadrature_by_definiteness(definiteness);

        quadrature.push_back(center, box.volume());
      }



      template <int dim, int spacedim>
      void
      QGenerator<dim, spacedim>::set_1D_quadrature(const unsigned int q_index)
      {
        AssertIndexRange(q_index, q_collection1D->size());

        low_dim_algorithm.set_1D_quadrature(q_index);
        up_through_dimension_creator.set_1D_quadrature(q_index);
      }



      template <int spacedim>
      QGenerator<1, spacedim>::QGenerator(
        const hp::QCollection<1> &      q_collection1D,
        const AdditionalQGeneratorData &additional_data)
        : additional_data(additional_data)
        , q_collection1D(&q_collection1D)
        , root_finder(
            RootFinder::AdditionalData(additional_data.root_finder_tolerance,
                                       additional_data.n_allowed_splits))
      {
        Assert(q_collection1D.size() > 0,
               ExcMessage("Incoming quadrature collection is empty."));
        q_index = 0;
      }



      template <int spacedim>
      void
      QGenerator<1, spacedim>::create_quadrature(
        const std::vector<std::shared_ptr<Function<1>>> &restrictions,
        const BoundingBox<1> &                           box)
      {
        roots.clear();
        root_finder.find_roots(restrictions, box, roots);

        const Quadrature<1> &quadrature1D = (*q_collection1D)[q_index];

        distribute_points_between_roots(quadrature1D,
                                        box,
                                        roots,
                                        zero_dim_point,
                                        unit_weight,
                                        direction,
                                        restrictions,
                                        additional_data,
                                        this->immersed_quadratures);

        if (spacedim == 1)
          this->create_surface_points(restrictions);
      }



      template <int spacedim>
      void
      QGenerator<1, spacedim>::create_surface_points(
        const std::vector<std::shared_ptr<Function<1>>> &restrictions)
      {
        Assert(restrictions.size() == 1, ExcInternalError());

        for (unsigned int i = 0; i < roots.size(); ++i)
          {
            // A surface integral in 1D is just a point evaluation,
            // so the weight is always 1.
            const double   weight = 1;
            const Point<1> point(roots[i]);

            Tensor<1, 1> normal        = restrictions[0]->gradient(point);
            const double gradient_norm = normal.norm();
            Assert(
              gradient_norm > 1e-11,
              ExcMessage(
                "The level set function has a gradient almost equal to 0."));
            normal *= 1. / gradient_norm;

            this->immersed_quadratures.surface.push_back(point, weight, normal);
          }
      }



      template <int spacedim>
      void
      QGenerator<1, spacedim>::set_1D_quadrature(const unsigned int q_index)
      {
        AssertIndexRange(q_index, q_collection1D->size());
        this->q_index = q_index;
      }
    } // namespace QuadratureGeneratorImplementation
  }   // namespace internal

  using namespace internal::QuadratureGeneratorImplementation;



  AdditionalQGeneratorData::AdditionalQGeneratorData(
    const unsigned int n_allowed_splits,
    const double       lower_bound_implicit_function,
    const double       min_distance_between_roots,
    const double       limit_to_be_definite,
    const double       root_finder_tolerance,
    const unsigned int n_allowed_root_finder_splits)
    : n_allowed_splits(n_allowed_splits)
    , lower_bound_implicit_function(lower_bound_implicit_function)
    , min_distance_between_roots(min_distance_between_roots)
    , limit_to_be_definite(limit_to_be_definite)
    , root_finder_tolerance(root_finder_tolerance)
    , n_allowed_root_finder_splits(n_allowed_root_finder_splits)
  {}



  template <int dim>
  QuadratureGenerator<dim>::QuadratureGenerator(
    const hp::QCollection<1> &quadratures1D,
    const AdditionalData &    additional_data)
    : q_generator(quadratures1D, additional_data)
  {}



  template <int dim>
  void
  QuadratureGenerator<dim>::create_quadrature(
    const std::shared_ptr<Function<dim>> &level_set,
    const BoundingBox<dim> &              box)
  {
    q_generator.clear_quadratures();

    std::vector<std::shared_ptr<Function<dim>>> restrictions;
    restrictions.push_back(level_set);

    q_generator.create_quadrature(restrictions, box);

    // The "indefinite" quadrature is only used for building the
    // inside and outside quadratures when going from the lower to the higher
    // dimension. When we get to this point it should be empty.
    Assert(q_generator.get_quadratures().indefinite.size() == 0,
           ExcInternalError());
  }



  template <int dim>
  const Quadrature<dim> &
  QuadratureGenerator<dim>::get_inside_quadrature() const
  {
    const ImmersedQuadratures<dim> &quadratures = q_generator.get_quadratures();

    return quadratures.inside;
  }



  template <int dim>
  const Quadrature<dim> &
  QuadratureGenerator<dim>::get_outside_quadrature() const
  {
    const ImmersedQuadratures<dim> &quadratures = q_generator.get_quadratures();

    return quadratures.outside;
  }



  template <int dim>
  const ImmersedSurfaceQuadrature<dim> &
  QuadratureGenerator<dim>::get_surface_quadrature() const
  {
    const ImmersedQuadratures<dim> &quadratures = q_generator.get_quadratures();

    return quadratures.surface;
  }


  template <int dim>
  void
  QuadratureGenerator<dim>::set_1D_quadrature(const unsigned int q_index)
  {
    q_generator.set_1D_quadrature(q_index);
  }

} // namespace NonMatching
#include "quadrature_generator.inst"
DEAL_II_NAMESPACE_CLOSE
